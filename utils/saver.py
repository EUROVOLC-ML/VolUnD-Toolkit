from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import os
from os.path import getmtime
import sys
import ast
import torch
from pathlib import Path
from time import time
from typing import Union
from torch.utils.tensorboard import SummaryWriter
import threading
import webbrowser
import matplotlib
matplotlib.use('Agg')


class Saver(object):
    """
    Saver allows for saving and restore networks.
    """

    def __init__(self, base_output_dir: Path, args: dict, sub_dirs=('trainingSet', 'validationSet'), tag=''):

        # Create experiment directory
        timestamp_str = datetime.fromtimestamp(
            time()).strftime('%Y-%m-%d_%H-%M-%S')
        if isinstance(tag, str) and len(tag) > 0:
            # Append tag
            timestamp_str += f"_{tag}"
        self.path = base_output_dir / f'{timestamp_str}'
        self.path.mkdir(parents=True, exist_ok=True)

        # TB logs
        self.args = args
        self.writer = SummaryWriter(str(self.path))

        # Create checkpoint sub-directory
        self.ckpt_path = self.path / 'ckpt'
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # Create output sub-directories
        self.sub_dirs = sub_dirs
        self.output_path = {}

        for s in self.sub_dirs:
            self.output_path[s] = self.path / 'output' / s

        for d in self.output_path.values():
            d.mkdir(parents=True, exist_ok=False)

        # Dump experiment hyper-params
        with open(self.path / 'hyperparams.txt', mode='wt') as f:
            args_str = [f'{a}: {v}\n' for a, v in self.args.items()]
            args_str.append(f'exp_name: {timestamp_str}\n')
            f.writelines(sorted(args_str))

        # Dump command
        with open(self.path / 'command.txt', mode='wt') as f:
            cmd_args = ' '.join(sys.argv)
            f.write(cmd_args)
            f.write('\n')

        # Start TensorBoard Daemon to visualize data
        if args['tensorboard_enable']:
            self.tensorboard_port = args['tensorboard_port']
            t = threading.Thread(target=lambda: os.system('tensorboard --logdir=' + str(
                self.path) + ' --port=' + str(self.tensorboard_port) + ' --bind_all'))
            t.start()
            webbrowser.open('http://localhost:' +
                            str(self.tensorboard_port) + '/', new=1)

    def save_checkpoint(self, net: torch.nn.Module, stats: dict, name: str, epoch: int):
        """
        Save model and optimizer parameters in the checkpoint directory.
        """
        # Get state dict
        model_state_dict = net.state_dict()
        # Copy to CPU
        for k, v in model_state_dict.items():
            model_state_dict[k] = v.cpu()
        # Save
        torch.save({'model_state_dict': model_state_dict, 'stats': stats,
                    'epoch': epoch}, self.ckpt_path / f'{name}_{epoch:05d}.pth')

    def dump_line(self, line, step, split, name, fmt=''):
        """
        Dump line as matplotlib figure into folder and tb

        """
        assert split in self.sub_dirs
        # Plot line
        fig = plt.figure()
        if isinstance(line, tuple):
            line_x, line_y = line
            plt.plot(line_x.cpu().detach().numpy(),
                     line_y.cpu().detach().numpy(), fmt)
        else:
            plt.plot(line.cpu().detach().numpy(), fmt)
        out_path = self.output_path[split] / f'line_{step:08d}_{name}.jpg'
        plt.savefig(out_path)
        self.writer.add_figure(f'{split}/{name}', fig, step)

    def dump_histogram(self, tensor: torch.Tensor, epoch: int, desc: str):
        try:
            self.writer.add_histogram(
                desc, tensor.contiguous().view(-1), epoch)
        except:
            print('Error writing histogram')

    def dump_metric(self, value: float, epoch: int, *tags):
        self.writer.add_scalar('/'.join(tags), value, epoch)

    def dump_graph(self, net: torch.nn.Module, tensor: torch.Tensor):
        """
        Dump model graph into tb.
        """
        self.writer.add_graph(net, tensor)

    @staticmethod
    def load_hyperparams(hyperparams_path):
        """
        Load hyperparams from file. Tries to convert them to best type.
        """
        # Process input
        hyperparams_path = Path(hyperparams_path)
        if not hyperparams_path.exists():
            raise OSError('Please provide a valid checkpoints path')
        if hyperparams_path.is_dir():
            hyperparams_path = os.path.join(hyperparams_path, 'hyperparams.txt')
        else:
            hyperparams_path = os.path.join(hyperparams_path.parent.parent, 'hyperparams.txt')
        # Prepare output
        output = {}
        # Read file
        with open(hyperparams_path) as file:
            # Read lines
            for l in file:
                # Remove new line
                l = l.strip()
                # Separate name from value
                toks = l.split(':')
                name = toks[0]
                value = ':'.join(toks[1:]).strip()
                # Parse value
                try:
                    value = ast.literal_eval(value)
                except:
                    pass
                # Add to output
                output[name] = value
        # Return
        return output

    @staticmethod
    def load_checkpoint(model_path: Union[str, Path], verbose: bool = True):
        """
        Load state dict e stats from pre-trained checkpoint. In case a directory is
          given as `model_path`, the best (minor loss) checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restore checkpoint.')

        if model_path.is_dir():
            # Check there are files in that directory
            file_list = sorted(model_path.glob('*.pth'), key=getmtime)
            if len(file_list) == 0:
                # Check there are files in the 'ckpt' subdirectory
                model_path = model_path / 'ckpt'
                file_list = sorted(model_path.glob('*.pth'), key=getmtime)
                if len(file_list) == 0:
                    raise OSError("Couldn't find pth file.")
            # Chose best checkpoint based on minor loss
            if verbose:
                print(f'Search best checkpoint (minor loss)...')
            loss = torch.load(file_list[0])['stats']['mse_loss']
            checkpoint = file_list[0]
            for i in tqdm(range(1, len(file_list))):
                loss_tmp = torch.load(file_list[i])['stats']['mse_loss']
                if loss_tmp < loss:
                    loss = loss_tmp
                    checkpoint = file_list[i]
            if verbose:
                print(f'Best checkpoint found: {checkpoint} (loss: {loss}).')
        elif model_path.is_file():
            if not model_path.as_posix().endswith('.pth'):
                raise OSError('Please provide a valid path for restore checkpoint.')
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        return torch.load(checkpoint)

    def close(self):
        self.writer.close()
