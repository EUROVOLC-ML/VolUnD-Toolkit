from pathlib import Path
import ast
import os
import torch
from torch.utils.data import Subset
from utils.dataset import Dataset
from utils.trainer import Trainer

def parse():
    """
    Load args from file. Tries to convert them to best type.
    """
    try:
        # Process input
        hyperparams_path = Path('./')
        if not hyperparams_path.exists():
            raise OSError('Setup dir not found')
        if hyperparams_path.is_dir():
            hyperparams = os.path.join(hyperparams_path, 'trainingSetup.txt')
        # Prepare output
        output = {}
        # Read file
        with open(hyperparams) as file:
            # Read lines
            for l in file:
                if l.startswith('#'):
                    continue
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

            # Verify setup integrity
            if not all(key in output.keys() for key in ['train_dir',
                                            'val_dir',
                                            'data_len',
                                            'chunk_jump',
                                            'channels_list',
                                            'batch_size',
                                            'data_provider',
                                            'mean',
                                            'std',
                                            'tag',
                                            'log_dir',
                                            'plot_every',
                                            'log_every',
                                            'save_every',
                                            'tensorboard_port',
                                            'layers_base',
                                            'channels_base',
                                            'min_spatial_size',
                                            'start_dilation',
                                            'min_sig_dil_ratio',
                                            'max_channels',
                                            'h_size',
                                            'enable_variational',
                                            'optim',
                                            'reduce_lr_every',
                                            'reduce_lr_factor',
                                            'weight_decay',
                                            'resume',
                                            'epochs',
                                            'lr',
                                            'device']):
                raise AttributeError("Params consistency broken!")
    except (FileNotFoundError, AttributeError, Exception):
        print("Restoring original params value in the setup file... please try to reconfigure setup.")
        f = open(os.path.join(hyperparams_path, 'trainingSetup.txt'), 'w')
        f.write("##############################################################################\n\
# TRAINING SETUP: please don't delete or modify attributes name (before ':') #\n\
##############################################################################\n\
\n\
# Dataset options\n\
train_dir: './dataset/trainingSet'\n\
val_dir: './dataset/validationSet'\n\
data_len: 512\n\
chunk_jump: False\n\
channels_list: None\n\
batch_size: 128\n\
data_provider: 'ram'\n\
mean: None\n\
std: None\n\
\n\
# Experiment options\n\
tag: 'ae'\n\
log_dir: './logs'\n\
plot_every: 1000\n\
log_every: 10\n\
save_every: 10\n\
tensorboard_port: 6006\n\
\n\
# Model options\n\
layers_base: 1\n\
channels_base: 16\n\
min_spatial_size: 2\n\
start_dilation: 3\n\
min_sig_dil_ratio: 50\n\
max_channels: 1024\n\
h_size: 64\n\
enable_variational: False\n\
\n\
# Training options\n\
optim: 'Adam'\n\
reduce_lr_every: None\n\
reduce_lr_factor: 0.1\n\
weight_decay: 0.0005\n\
resume: None\n\
epochs: 32000\n\
lr: 0.00001\n\
device: 'cuda'")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output

if __name__ == '__main__':
    # Get params
    args = parse()

    # Check chunk lenght
    n = args['data_len']
    if not ((n & (n-1) == 0) and n != 0):
        raise AttributeError("Chunk lenght must be a power of 2!")
    
    # Normalization
    normalize_params={"mean":args['mean'], "std":args['std']}

    # Create dataset
    train_dataset = Dataset(args['train_dir'], chunk_len=args['data_len'], chunk_jump=args['chunk_jump'], normalize_params=normalize_params, channels_list=args['channels_list'], provider=args['data_provider'])
    val_dataset = Dataset(args['val_dir'], chunk_len=args['data_len'], chunk_jump=args['chunk_jump'], normalize_params=normalize_params, channels_list=args['channels_list'], provider=args['data_provider'])
    
     # Save number of channels
    example,_,_ = train_dataset[0]
    args['data_channels'] = example.shape[0] # 0=channel, 1=chunk
    if(args['channels_list'] is None):
        args['channels_list'] = torch.arange(args['data_channels'])
    print("Channels: " + str(args['data_channels']))
    
    # Setup dataset dictionary
    args['datasets'] = {'trainingSet': train_dataset, 'validationSet': val_dataset}

    # Define trainer
    trainer = Trainer(args)

    # Run training
    model, metrics = trainer.train()
