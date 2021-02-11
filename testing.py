from pathlib import Path
import ast
import os
from utils.saver import Saver
from utils.dataset import Dataset
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data
from utils.model import Model
from matplotlib import pyplot as plt
from datetime import datetime

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
            hyperparams = os.path.join(hyperparams_path, 'testingSetup.txt')
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
            if not all(key in output.keys() for key in ['checkpoint',
                                            'test_dir',
                                            'data_len',
                                            'chunk_jump',
                                            'channels_list',
                                            'batch_size',
                                            'data_provider',
                                            'mean',
                                            'std',
                                            'device']):
                raise AttributeError("Params consistency broken!")
    except (FileNotFoundError, AttributeError, Exception):
        print("Restoring original params value in the setup file... please try to reconfigure setup.")
        f = open(os.path.join(hyperparams_path, 'testingSetup.txt'), 'w')
        f.write("##############################################################################\n\
# TESTING SETUP: please don't delete or modify attributes name (before ':') #\n\
##############################################################################\n\
\n\
# Checkpoints options\n\
checkpoint: './logs/yyyy-mm-dd_hh-mm-ss_ae/'\n\
\n\
# Dataset options\n\
test_dir: './dataset/testSet'\n\
data_len: 512\n\
chunk_jump: False\n\
channels_list: None\n\
batch_size: 128\n\
data_provider: 'ram'\n\
mean: None\n\
std: None\n\
\n\
# Model options\n\
device: 'cuda'")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output


def plotGraphs(dist, i_channel, outDATETIME, label_activity, label_eruption):
    # Normal y-scale
    if (i_channel != ' ALL'):
        dist_ch = dist[:,i_channel]
    else:
        dist_ch = dist.mean(dim=1)
    _, ax = plt.subplots()
    x = range(dist_ch.shape[0])
    y = dist_ch
    plt.xticks(x, outDATETIME, rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xlabel('Timestamp (yyyy-mm-dd-hh-mm-ss)')
    plt.ylabel('Reconstruction distance')
    max_value = y.max().item()
    plt.fill_between(x, np.array(label_activity, dtype=int)*(max_value/2), color='yellow', label='Activity')
    plt.fill_between(x, np.array(label_eruption, dtype=int)*max_value, color='red', label='Eruption')
    plt.plot(x,y, color='green')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    title="Graph CH" + str(i_channel)
    ax.title.set_text(title)
    plt.tight_layout()
    plt.show()
    # Log y-scale
    if (i_channel != ' ALL'):
        dist_ch = dist[:,i_channel]
    else:
        dist_ch = dist.mean(dim=1)
    _, ax = plt.subplots()
    x = range(dist_ch.shape[0])
    y = dist_ch
    plt.xticks(x, outDATETIME, rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xlabel('Timestamp (yyyy-mm-dd-hh-mm-ss)')
    plt.ylabel('Reconstruction distance (Log scale)')
    max_value = y.max().item()
    plt.fill_between(x, np.array(label_activity, dtype=int)*(max_value/2), color='yellow', label='Activity')
    plt.fill_between(x, np.array(label_eruption, dtype=int)*max_value, color='red', label='Eruption')
    plt.plot(x,y, color='dodgerblue')
    plt.yscale('log')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    title="Graph (LOG y-scale) CH" + str(i_channel)
    ax.title.set_text(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Get params
    args = parse()

    # Retrieve absolute path of checkpoint
    checkpoint = os.path.abspath(args['checkpoint'])

    # Load arguments
    hyperparams = Saver.load_hyperparams(checkpoint)

    # Normalization
    normalize_params={"mean":args['mean'], "std":args['std']}

    # Instantiate dataset
    test_dataset = Dataset(args['test_dir'], chunk_len=args['data_len'], chunk_jump=args['chunk_jump'], normalize_params=normalize_params, channels_list=args['channels_list'], provider=args['data_provider'])
    
    # Instantiate loader
    test_loader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers= 0, drop_last=True)

    # Save number of channels
    example,_,_ = test_dataset[0]
    args['data_channels'] = example.shape[0] # 0=channel, 1=chunk
    if(args['channels_list'] is None):
        args['channels_list'] = torch.arange(args['data_channels'])
    if(args['data_channels'] != hyperparams['data_channels']):
        raise AttributeError("Channels number of checkpoint is not equal to channels number of TestSet!")
    print("Channels: " + str(args['data_channels']))

    # Setup model
    model = Model(hyperparams)
    model.load_state_dict(Saver.load_checkpoint(checkpoint)['state_dict'])
    model.eval()
    model.to(args['device'])

    # Model evalutation
    out = []
    with torch.no_grad():
        for sig,_,_ in tqdm(test_loader, desc='Testing'):
            rec,_,_ = model(sig.to(args['device']))
            out.append(rec.detach().cpu())

    # Group reconstructions
    outLIN = []
    outLABEL = []
    outTIMESTAMP = []
    for i, sig_batch in enumerate(tqdm(out, desc='Elaborating')):
        for j in range(sig_batch.shape[0]):
            outLIN.append(test_dataset[i*args['batch_size']+j][0] - sig_batch[j])
            outLABEL.append(test_dataset[i*args['batch_size']+j][1])
            outTIMESTAMP.append(test_dataset[i*args['batch_size']+j][2])
    outUNIONdiff = torch.stack(outLIN)
    outDATETIME = [datetime.fromtimestamp(t) for t in outTIMESTAMP]
    
    # Compute distance
    print("Compute distances per channel...")
    dist = outUNIONdiff.pow(2).sum(2).sqrt()

    # Compute labels
    label_activity = [(label==1) for label in outLABEL]
    label_eruption = [(label==2) for label in outLABEL]

    # Plot distance per channel
    print("Showing graphs per CH...")
    for i_channel in range(args['data_channels']):
        plotGraphs(dist, i_channel, outDATETIME, label_activity, label_eruption)

    # Plot total distance
    print("Showing total graphs...")
    plotGraphs(dist, " ALL", outDATETIME, label_activity, label_eruption)
