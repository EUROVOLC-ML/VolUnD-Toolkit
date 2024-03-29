import webbrowser
import sys
import os
from utils.saver import Saver
from utils.dataset import Dataset
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data
from utils.model import Model
from utils.parser import testing_parse
from matplotlib import pyplot as plt
from datetime import datetime, timezone
from statsmodels.distributions.empirical_distribution import ECDF

# Graph visualization on browser
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['webagg.address'] = '0.0.0.0'
matplotlib.rcParams['webagg.open_in_browser'] = False
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def plotSetup(ax, x, y, channel_name, outDATETIME, label_activity, label_eruption, epoch, tipology="norm"):
    max_value = y.max().item()
    ax.set_xticks(x)
    ax.set_xticklabels(outDATETIME, rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set(xlabel='Timestamp (yyyy-mm-dd-hh-mm-ss)')
    ax.fill_between(x, np.array(label_activity, dtype=int) * (max_value/2), color='yellow', label='Activity')
    ax.fill_between(x, np.array(label_eruption, dtype=int) * max_value, color='red', label='Eruption')
    print("    In progress 1/2...")
    if tipology == "norm":
        ax.set(ylabel='Reconstruction distance')
        ax.plot(x, y, color='green')
        title = "Graph CH_" + str(channel_name) + f" (epoch {epoch})"
    elif tipology == "log":
        ax.set(ylabel='Reconstruction distance (LOG scale)')
        ax.plot(x, y, color='dodgerblue')
        ax.set_yscale('log')
        title = "Graph (LOG y-scale) CH_" + str(channel_name) + f" (epoch {epoch})"
    ax.title.set_text(title)
    print("    In progress 2/2...")


def plotAndSaveGraphs(dist_ch, channel_name, outDATETIME, label_activity, label_eruption, img_location, epoch, dpi=300, CDF=False):
    x = range(dist_ch.shape[0])
    _, ax = plt.subplots(ncols=2, nrows=1, tight_layout=True)
    # Normal y-scale
    print("Elaborating 1/2...")
    plotSetup(ax[0], x, dist_ch, channel_name, outDATETIME, label_activity, label_eruption, epoch, tipology="norm")
    # Log y-scale
    print("Elaborating 2/2...")
    plotSetup(ax[1], x, dist_ch, channel_name, outDATETIME, label_activity, label_eruption, epoch, tipology="log")
    # Show graphs
    print("Saving...")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    if os.path.isfile(img_location):
        folder = os.path.dirname(os.path.dirname(img_location))
    else:
        folder = img_location
    folder = os.path.join(folder, "testing")
    os.makedirs(folder, exist_ok=True)
    if CDF:
        name_file = "CH_" + str(channel_name) + f"_epoch{epoch:05d}_CDF.png"
    else:
        name_file = "CH_" + str(channel_name) + f"_epoch{epoch:05d}.png"
    plt.savefig(os.path.join(folder, name_file), dpi=dpi)


def getDist(args, normalize_params, train):
    # Instantiate dataset
    dataset = Dataset(args['train_dir'] if train else args['test_dir'],
                           data_location=args['data_location'],
                           chunk_len=args['chunk_len'],
                           chunk_only_one=args['chunk_only_one'],
                           chunk_rate=args['chunk_rate'],
                           chunk_random_crop=args['chunk_random_crop'],
                           data_sampling_frequency=args['data_sampling_frequency'],
                           chunk_linear_subsample=args['chunk_linear_subsample'],
                           chunk_butterworth_lowpass=args['chunk_butterworth_lowpass'],
                           chunk_butterworth_highpass=args['chunk_butterworth_highpass'],
                           chunk_butterworth_order=args['chunk_butterworth_order'],
                           normalize_params=normalize_params,
                           channels_list=args['channels_list'],
                           channels_name=args['channels_name'],
                           provider=args['data_provider'],
                           labels=args['training_labels'] if train else args['test_labels'])

    # Instantiate loader
    loader = data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, drop_last=True)

    # Model evaluation
    out = []
    with torch.no_grad():
        for sig, _, _, _ in tqdm(loader, desc='Training' if train else 'Testing'):
            rec, _, _ = model(sig.to(args['device']))
            out.append(rec.detach().cpu())

    # Group reconstructions
    outLIN = []
    outLABEL = []
    outTIMESTAMP = []
    for i, sig_batch in enumerate(tqdm(out, desc='Elaborating')):
        for j in range(sig_batch.shape[0]):  # batch
            tmp_sig = torch.zeros(sig_batch.shape[1:])
            for k in range(sig_batch.shape[1]):  # channel
                # Insert nan on reconstruction distance if signal is all 0 (station off)
                if dataset[i*args['batch_size']+j][0][k].abs().max() != 0:
                    tmp_sig[k] = dataset[i*args['batch_size']+j][0][k] - sig_batch[j, k]
                else:
                    tmp_sig[k] = np.nan
            outLIN.append(tmp_sig)
            if not train:
                outLABEL.append(dataset[i*args['batch_size']+j][1])
                outTIMESTAMP.append(dataset[i*args['batch_size']+j][2])
    outUNIONdiff = torch.stack(outLIN)
    if not train:
        outDATETIME = [datetime.fromtimestamp(t, timezone.utc) for t in outTIMESTAMP]

    # Compute distance
    print("Compute distances per channel...")
    dist = outUNIONdiff.pow(2).sum(2).sqrt()
    if not train:
        return dist, outLABEL, outDATETIME, dataset.channels_list, dataset.get_channels_name()
    else:
        return dist


if __name__ == '__main__':
    # Get params
    args = testing_parse()

    # Set backend port
    matplotlib.rcParams['webagg.port'] = args['web_port']

    # Retrieve absolute path of checkpoint
    checkpoint = os.path.abspath(args['checkpoint'])

    # Load arguments
    hyperparams = Saver.load_hyperparams(checkpoint)
    checkpoint_dict = Saver.load_checkpoint(checkpoint)

    # Normalization
    normalize_params = {"mean": args['mean'], "std": args['std']}

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
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model.eval()
    model.to(args['device'])

    # Elaborate testSet
    dist, outLABEL, outDATETIME, channels_list, channels_name = getDist(args, normalize_params, train=False)
    
    # Save number of channels
    args['data_channels'] = len(channels_list)

    if args['CDF_mode']:
        # Elaborate trainingSet
        dist_train = getDist(args, normalize_params, train=True)
        
        # Calculate CDF from trainingSet
        print("Calculating CDF from trainingSet...")
        ecdf = []
        for i_channel in tqdm(range(args['data_channels'])):
            ecdf.append(ECDF(dist_train[:,i_channel]))

        # Calculate PDF from testSet distance
        print("Calculating PDF of testSet...")
        for i_channel in tqdm(range(args['data_channels'])):
            dist[:,i_channel] = torch.Tensor(ecdf[i_channel](dist[:,i_channel]))

    # Compute labels
    label_activity = [(label in args['label_activity']) for label in outLABEL]
    label_eruption = [(label in args['label_eruption']) for label in outLABEL]

    # Get checkpoint epoch
    epoch = checkpoint_dict['epoch']

    # Plot distance per channel
    print("Showing graphs per CH:")
    for i_channel in range(args['data_channels']):
        print("CHANNEL " + str(i_channel+1) + "/" + str(args['data_channels']) + ":")
        # Get single channel reconstruction distance
        dist_ch = dist[:, i_channel]
        # Get channel name
        channel_name = channels_name[i_channel]
        # Plot
        plotAndSaveGraphs(dist_ch,
                          channel_name,
                          outDATETIME,
                          label_activity,
                          label_eruption,
                          checkpoint,
                          epoch,
                          args['img_quality'],
                          args['CDF_mode'])

    # Plot total distance
    print("Showing total graphs:")
    # Get mean of all channel reconstruction distance
    dist_ch = dist.mean(dim=1)
    # Plot
    plotAndSaveGraphs(dist_ch,
                      "ALL",
                      outDATETIME,
                      label_activity,
                      label_eruption,
                      checkpoint,
                      epoch,
                      args['img_quality'],
                      args['CDF_mode'])

    # Show graph on browser
    print("To view figure, visit http://127.0.0.1:" + str(args['web_port']))
    print("Press Ctrl+C or Ctrl+Break to stop WebAgg server")
    webbrowser.open('http://127.0.0.1:' + str(args['web_port']) + '/', new=1)
    print("### Please, ignore the next 2 printed lines ###")
    plt.show()
