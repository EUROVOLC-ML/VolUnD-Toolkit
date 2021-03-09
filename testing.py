import webbrowser
import sys
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

# Graph visualization on browser
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['webagg.address'] = '0.0.0.0'
matplotlib.rcParams['webagg.open_in_browser'] = False
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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
                                                        'data_location',
                                                        'chunk_len',
                                                        'chunk_only_one',
                                                        'chunk_rate',
                                                        'chunk_random_crop',
                                                        'chunk_linear_subsample',
                                                        'chunk_butterworth_lowpass',
                                                        'chunk_butterworth_highpass',
                                                        'chunk_butterworth_signal_frequency',
                                                        'chunk_butterworth_order',
                                                        'channels_list',
                                                        'batch_size',
                                                        'data_provider',
                                                        'mean',
                                                        'std',
                                                        'label_activity',
                                                        'label_eruption',
                                                        'device',
                                                        'img_quality',
                                                        'web_port']):
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
data_location: './path/to/data/'\n\
chunk_len: 512\n\
chunk_only_one: False\n\
chunk_rate: 1\n\
chunk_random_crop: False\n\
chunk_linear_subsample: 1\n\
chunk_butterworth_lowpass: None\n\
chunk_butterworth_highpass: None\n\
chunk_butterworth_signal_frequency: None\n\
chunk_butterworth_order: 2\n\
channels_list: None\n\
batch_size: 128\n\
data_provider: 'ram'\n\
mean: None\n\
std: None\n\
label_activity: [1]\n\
label_eruption: [2]\n\
\n\
# Model options\n\
device: 'cuda'\n\
\n\
# Image options\n\
img_quality: 600\n\
\n\
# Web options\n\
web_port: 8988")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output


def plotSetup(ax, x, y, channel_name, outDATETIME, label_activity, label_eruption, epoch, y_log=False):
    max_value = y.max().item()
    ax.set_xticks(x)
    ax.set_xticklabels(outDATETIME, rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set(xlabel='Timestamp (yyyy-mm-dd-hh-mm-ss)')
    ax.fill_between(x, np.array(label_activity, dtype=int) *
                    (max_value/2), color='yellow', label='Activity')
    ax.fill_between(x, np.array(label_eruption, dtype=int) *
                    max_value, color='red', label='Eruption')
    print("    In progress 1/2...")
    if not y_log:
        ax.set(ylabel='Reconstruction distance')
        ax.plot(x, y, color='green')
        if epoch == None:
            title = "Graph CH_" + str(channel_name)
        else:
            title = "Graph CH_" + str(channel_name) + f" (epoch {epoch})"
    else:
        ax.set(ylabel='Reconstruction distance (LOG scale)')
        ax.plot(x, y, color='dodgerblue')
        ax.set_yscale('log')
        if epoch == None:
            title = "Graph (LOG y-scale) CH_" + str(channel_name)
        else:
            title = "Graph (LOG y-scale) CH_" + \
                str(channel_name) + f" (epoch {epoch})"
    ax.title.set_text(title)
    print("    In progress 2/2...")


def plotAndSaveGraphs(dist_ch, channel_name, outDATETIME, label_activity, label_eruption, img_location, epoch, dpi=300):
    x = range(dist_ch.shape[0])
    y = dist_ch
    _, ax = plt.subplots(ncols=2, nrows=1, tight_layout=True)
    # Normal y-scale
    print("Elaborating 1/2...")
    plotSetup(ax[0], x, y, channel_name, outDATETIME,
              label_activity, label_eruption, epoch, y_log=False)
    # Log y-scale
    print("Elaborating 2/2...")
    plotSetup(ax[1], x, y, channel_name, outDATETIME,
              label_activity, label_eruption, epoch, y_log=True)
    # Show graphs
    print("Saving...")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    if os.path.isfile(img_location):
        folder = os.path.dirname(os.path.dirname(img_location))
    else:
        folder = img_location
    folder = os.path.join(folder, "testing")
    os.makedirs(folder, exist_ok=True)
    if epoch == None:
        name_file = "CH_" + str(channel_name) + ".png"
    else:
        name_file = "CH_" + str(channel_name) + f"_epoch{epoch:05d}" + ".png"
    plt.savefig(os.path.join(folder, name_file), dpi=dpi)


if __name__ == '__main__':
    # Get params
    args = parse()

    # Set backend port
    matplotlib.rcParams['webagg.port'] = args['web_port']

    # Retrieve absolute path of checkpoint
    checkpoint = os.path.abspath(args['checkpoint'])

    # Load arguments
    hyperparams = Saver.load_hyperparams(checkpoint)
    checkpoint_dict = Saver.load_checkpoint(checkpoint)

    # Normalization
    normalize_params = {"mean": args['mean'], "std": args['std']}

    # Instantiate dataset
    test_dataset = Dataset(args['test_dir'], data_location=args['data_location'], chunk_len=args['chunk_len'], chunk_only_one=args['chunk_only_one'], chunk_rate=args['chunk_rate'], chunk_random_crop=args['chunk_random_crop'], chunk_linear_subsample=args['chunk_linear_subsample'], chunk_butterworth_lowpass=args['chunk_butterworth_lowpass'],
                           chunk_butterworth_highpass=args['chunk_butterworth_highpass'], chunk_butterworth_signal_frequency=args['chunk_butterworth_signal_frequency'], chunk_butterworth_order=args['chunk_butterworth_order'], normalize_params=normalize_params, channels_list=args['channels_list'], provider=args['data_provider'])

    # Instantiate loader
    test_loader = data.DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, drop_last=True)

    # Save number of channels
    example, _, _ = test_dataset[0]
    args['data_channels'] = example.shape[0]  # 0=channel, 1=chunk
    if(args['channels_list'] is None):
        args['channels_list'] = torch.arange(args['data_channels'])
    else:
        args['channels_list'] = torch.tensor(
            args['channels_list'], dtype=torch.int32)
    if(args['data_channels'] != hyperparams['data_channels']):
        raise AttributeError(
            "Channels number of checkpoint is not equal to channels number of TestSet!")
    print("Channels: " + str(args['channels_list'].numpy()))

    # Setup model
    model = Model(data_len=int(hyperparams['chunk_len'] / hyperparams['chunk_linear_subsample']),
                  data_channels=hyperparams['data_channels'],
                  layers_base=hyperparams['layers_base'],
                  channels_base=hyperparams['channels_base'],
                  min_spatial_size=hyperparams['min_spatial_size'],
                  start_dilation=hyperparams['start_dilation'],
                  min_sig_dil_ratio=hyperparams['min_sig_dil_ratio'],
                  max_channels=hyperparams['max_channels'],
                  h_size=hyperparams['h_size'],
                  enable_variational=hyperparams['enable_variational'])
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.eval()
    model.to(args['device'])

    # Model evalutation
    out = []
    with torch.no_grad():
        for sig, _, _ in tqdm(test_loader, desc='Testing'):
            rec, _, _ = model(sig.to(args['device']))
            out.append(rec.detach().cpu())

    # Group reconstructions
    outLIN = []
    outLABEL = []
    outTIMESTAMP = []
    for i, sig_batch in enumerate(tqdm(out, desc='Elaborating')):
        for j in range(sig_batch.shape[0]):
            outLIN.append(
                test_dataset[i*args['batch_size']+j][0] - sig_batch[j])
            outLABEL.append(test_dataset[i*args['batch_size']+j][1])
            outTIMESTAMP.append(test_dataset[i*args['batch_size']+j][2])
    outUNIONdiff = torch.stack(outLIN)
    outDATETIME = [datetime.fromtimestamp(t) for t in outTIMESTAMP]

    # Compute distance
    print("Compute distances per channel...")
    dist = outUNIONdiff.pow(2).sum(2).sqrt()

    # Compute labels
    label_activity = [(label in args['label_activity']) for label in outLABEL]
    label_eruption = [(label in args['label_eruption']) for label in outLABEL]

    # Get checkpoint epoch
    try:
        epoch = checkpoint_dict['epoch']
    except(KeyError):
        print("'Epoch' info not found in checkpoint.")
        epoch = None

    # Plot distance per channel
    print("Showing graphs per CH:")
    for i_channel in range(args['data_channels']):
        print("CHANNEL " + str(i_channel+1) + "/" +
              str(args['data_channels']) + ":")
        # Get single channel reconstruction distance
        dist_ch = dist[:, i_channel]
        # Get channel name
        channel_name = test_dataset.get_channels_name()[i_channel]
        # Plot
        plotAndSaveGraphs(dist_ch, channel_name, outDATETIME, label_activity,
                          label_eruption, checkpoint, epoch, args['img_quality'])

    # Plot total distance
    print("Showing total graphs:")
    # Get mean of all channel reconstruction distance
    dist_ch = dist.mean(dim=1)
    # Plot
    plotAndSaveGraphs(dist_ch, "ALL", outDATETIME, label_activity,
                      label_eruption, checkpoint, epoch, args['img_quality'])

    # Show graph on browser
    print("To view figure, visit http://127.0.0.1:" + str(args['web_port']))
    print("Press Ctrl+C or Ctrl+Break to stop WebAgg server")
    webbrowser.open('http://127.0.0.1:' + str(args['web_port']) + '/', new=1)
    print("### Please, ignore the next 2 printed lines ###")
    plt.show()
