import torch
from datetime import datetime, timedelta
import scipy.io as sio


def read_file(file_path):
    file_name = file_path.split("\\")[-1]

    # Read file with same library how it is saved
    data_load = sio.loadmat(file_path)

    # Extract signal data from file read
    data = torch.Tensor(data_load['fragment']['data'][0][0]).unsqueeze(1)

    # Extract timestamp of the end of the signals from file read
    timestamp = []
    timestamp.append((datetime.strptime(file_name[0:6], "%y%m%d") + timedelta(minutes=int(file_name[7:10])*10)).timestamp())

    # Check if label data exist on file read and extract it or set to 0 (normal activity)
    try:
        label = []
        label.append(data_load['fragment']['current_label'][0][0][0][0])
    except KeyError:
        print("Label not found: assume all events of class 0 (normal)!")
        label = torch.zeros(len(timestamp)).tolist()

    return data, label, timestamp


def read_file_info(file_path, channels_list):
    # Read file with same library how it is saved
    data_load = sio.loadmat(file_path)

    # Check if channels name exist on file read and extract it or set to value of channels_list
    try:
        channels_name = data_load['CHANNELS_NAME']
    except(KeyError):
        print("Channel Name not found.")
        if channels_list is not None:
            channels_name = [str(channels_list[i].item()) for i in range(channels_list.shape[0])]
        else:
            tmp = torch.arange(data_load['DATA'].shape[0])
            channels_name = [str(tmp[i].item()) for i in range(tmp.shape[0])]

    return channels_name
