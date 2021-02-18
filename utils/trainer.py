from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.saver import Saver
from utils.model import Model

class Trainer:

    def __init__(self, args):
        # Store args
        self.args = args
        
        # Setup saver
        self.saver = Saver(Path(self.args['log_dir']), self.args, sub_dirs=list(self.args['datasets'].keys()), tag=self.args['tag'])

        # Setup model
        self.net = Model(data_len = int(self.args['chunk_len'] / self.args['chunk_linear_subsample']),
                            data_channels = self.args['data_channels'],
                            layers_base = self.args['layers_base'],
                            channels_base = self.args['channels_base'],
                            min_spatial_size = self.args['min_spatial_size'],
                            start_dilation = self.args['start_dilation'],
                            min_sig_dil_ratio = self.args['min_sig_dil_ratio'],
                            max_channels = self.args['max_channels'],
                            h_size = self.args['h_size'],
                            enable_variational = self.args['enable_variational'])

        # Check resume
        if self.args['resume'] is not None:
            self.net.load_state_dict(Saver.load_state_dict(self.args['resume']))

        # Move to device
        self.net.to(self.args['device'])

        # Add network to params
        self.args['net'] = str(self.net)
        self.args[f'{self.net}_parameters'] = np.sum([p.numel() for p in self.net.parameters()])

        # Store args
        self.ch_list = self.args['channels_list']
        self.plot_every = self.args['plot_every']

        # Optimizer params
        optim_params = {'lr': self.args['lr'], 'weight_decay': self.args['weight_decay']}
        if self.args['optim'] == 'Adam':
            optim_params = {**optim_params, 'betas': (0.5, 0.999)}
        elif self.args['optim'] == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}

        # Create optimizer
        optim_class = getattr(torch.optim, self.args['optim'])
        self.optim = optim_class(params=self.net.parameters(), **optim_params)

        # Configure LR scheduler
        if self.args['optim'] == 'SGD' and self.args['reduce_lr_every'] is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args['reduce_lr_every'], self.args['reduce_lr_factor'])
        else:
            self.scheduler = None

        # Compute splits names
        self.splits = list(self.args['datasets'].keys())

        # Setup data loader
        self.loaders = {s: DataLoader(self.args['datasets'][s], batch_size=self.args['batch_size'], shuffle=(s == list(self.args['datasets'].keys())[0]), num_workers= 0, drop_last=True) for s in self.splits}

    def train(self):
        # Initialize output metrics
        result_metrics = {s: {} for s in self.splits}

        # Process each epoch
        for epoch in range(1, self.args['epochs']):
            try:
                # Define stuff
                train_rec_dists = None
                dist_thresholds = None

                # Process each split
                for split in self.splits:
                    # Epoch metrics
                    epoch_metrics = {}

                    # Train: Initialize training reconstruction distances
                    if split == list(self.args['datasets'].keys())[0]: # Training Set
                        train_rec_dists = []

                    # Process each batch
                    dl = self.loaders[split]
                    for batch_idx, (x, label, _) in enumerate(tqdm(dl, desc=f'{split}, {epoch}')):
                        # Initialize trainer args
                        args_trainer = {}

                        # Validation: add distance and label
                        if dist_thresholds is not None:
                            args_trainer['dist_thresholds'] = dist_thresholds

                        # Forward batch
                        x = x.to(self.args['device'])

                        args_trainer['step'] = (epoch * len(dl)) + batch_idx
                        args_trainer['split'] = split
                        out,metrics = self.__forward_batch(x, label, args_trainer)

                        # Check NaN
                        if torch.isnan(out[0]).all():
                            if split == list(self.args['datasets'].keys())[0]:
                                raise FloatingPointError('Found NaN values')
                            else:
                                print('Warning: Found NaN values')

                        # Training
                        if split == list(self.args['datasets'].keys())[0]: # Training Set
                            # Keep track of reconstruction distance
                            train_rec_dists.append(metrics['rec_dist'])

                        # Log metrics 
                        for k, v in metrics.items():
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

                            # Check tensorboard
                            if args_trainer['step'] % self.args['log_every'] == 0 and not isinstance(v, torch.Tensor):
                                self.saver.dump_metric(v, args_trainer['step'], split, k, 'batch')

                    # End epoch, training: estimate thresholds from reconstruction distance
                    self.__end_epoch()
                    if split == list(self.args['datasets'].keys())[0]: # Training Set
                        # Get stats
                        min_rec_dist = min(train_rec_dists)
                        max_rec_dist = max(train_rec_dists)

                        # Compute threshold range
                        dist_thresholds = torch.linspace(0, 2*max_rec_dist, 64).unsqueeze(0).to(self.args['device']) # or (min_rec_dist, max_rec_dist + 2*(max_rec_dist - min_rec_dist), 20)
                        
                        # Log epoch metrics 
                        for k, v in epoch_metrics.items():
                            # Compute epoch average
                            avg_v = sum(v)/len(v)
                            # Dump to saver
                            self.saver.dump_metric(avg_v, epoch, split, k, 'epoch')
                            # Add to output results
                            result_metrics[split][k] = result_metrics[split][k] + [avg_v] if k in result_metrics[split] else [avg_v]

                    # End epoch, val: compute TPR and FPR
                    elif split == list(self.args['datasets'].keys())[1]: # Validation Set
                        # Compute TPR and FPR
                        tpr_num = torch.cat([x.unsqueeze(0) for x in epoch_metrics['tp']], 0).sum(0).float()
                        tpr_den = torch.cat([x.unsqueeze(0) for x in epoch_metrics['tp'] + epoch_metrics['fn']], 0).sum(0).float()
                        tpr = tpr_num/tpr_den
                        tpr[tpr_den == 0] = 0
                        fpr_num = torch.cat([x.unsqueeze(0) for x in epoch_metrics['fp']], 0).sum(0).float()
                        fpr_den = torch.cat([x.unsqueeze(0) for x in epoch_metrics['fp'] + epoch_metrics['tn']], 0).sum(0).float()
                        fpr = fpr_num/fpr_den
                        fpr[fpr_den == 0] = 0
                        # Add ROC curve
                        self.saver.dump_line((fpr, tpr), epoch, split, 'ROC', 'o')

                # Save checkpoint
                if epoch % self.args['save_every'] == 0:
                    self.saver.save_checkpoint(self.net, metrics, "Model", epoch)

            except KeyboardInterrupt:
                print('Caught keyboard interrupt: saving checkpoint...')
                self.saver.save_checkpoint(self.net, metrics, "Model", epoch)
                break

            except FloatingPointError as err:
                print(f'Error: {err}')
                break

        # Save last checkpoint
        self.saver.save_checkpoint(self.net, metrics, "Model", epoch)

        # Terminate saver
        self.saver.close()

        # Return
        return self.net, result_metrics

    def __forward_batch(self, x, label, args):
        # Set network mode
        if args['split'] == list(self.args['datasets'].keys())[0]: # Training Set
            self.net.train()
            torch.set_grad_enabled(True)
        else:
            self.net.eval()
            torch.set_grad_enabled(False)
        
        # Compute loss
        x_rec, mu, logvar = self.net(x)
        mse_loss = self.net.loss(x_rec, x, mu, logvar)

        # Optimize
        if args['split'] == list(self.args['datasets'].keys())[0]: # Training Set
            self.optim.zero_grad()
            mse_loss.backward()
            self.optim.step()

        # Log outputs and gradients
        if args['split'] == list(self.args['datasets'].keys())[0] and args['step'] % self.plot_every == 0:
            # Log output histograms
            for name,module in self.net.named_modules():
                if name != '' and hasattr(module, 'last_output'):
                    # Log histogram
                    self.saver.dump_histogram(module.last_output, args['step'], 'output ' + name)
            # Log parameters and gradients 
            for name,param in self.net.named_parameters():
                self.saver.dump_histogram(param.grad, args['step'], 'grad ' + name)
                self.saver.dump_histogram(param.data, args['step'], 'param ' + name)
        
        # Compute MAE
        mae = (x - x_rec).abs().mean()

        # Compute reconstruction distance
        rec_dist = (x - x_rec).pow(2).sum(2).sum(1).sqrt()
        avg_rec_dist = rec_dist.mean()

        # Initialize metrics
        metrics = {'mse_loss': mse_loss.item(),
                   'mae': mae.item(),
                   'rec_dist': avg_rec_dist.item()}

        # Check thresholds are availables
        if 'dist_thresholds' in args:
            # Read data
            dist_thresholds = args['dist_thresholds']
            # Add dimensions for broadcast
            rec_dist = rec_dist.unsqueeze(1)
            # Compute predictions at different thresholds
            preds = (rec_dist > dist_thresholds).type(label.type())
            # Expand label for broadcast
            label = label.unsqueeze(1).expand_as(preds) 
            # Compute TP, FP, TN, FN for each threshold
            metrics['tp'] = (preds*label).sum(0)
            metrics['fp'] = (preds*(1-label)).sum(0)
            metrics['tn'] = ((1-preds)*(1-label)).sum(0)
            metrics['fn'] = ((1-preds)*label).sum(0)

        # Plot
        if args['step'] % self.plot_every == 0:
            for ch in self.ch_list:
                self.saver.dump_line(x[0,ch,:], args['step'], args['split'], 'CH_'+str(ch.item()))
                self.saver.dump_line(x_rec[0,ch,:], args['step'], args['split'], 'CH_'+str(ch.item())+'_reconstruction')
                
        # Return metrics
        return (x_rec, mu, logvar), metrics

    def __end_epoch(self):
        # Optimizer scheduler step
        if self.scheduler is not None:
            self.scheduler.step()