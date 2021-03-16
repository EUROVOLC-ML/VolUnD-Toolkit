import torch


def read_file(file_path):
    # Read file with same library how it is saved
    data_load = torch.load(file_path)

    # Extract signal data from file read
    data = data_load['DATA']

    # Extract timestamp data from file read
    timestamp = data_load['TIMESTAMP']

    # Check if label data exist on file read and extract it or set to 0 (normal activity)
    try:
        label = data_load['LABEL']
    except KeyError:
        print("Label not found: assume all events of class 0 (normal)!")
        label = torch.zeros(len(timestamp)).tolist()

    return data, label, timestamp


def read_file_info(file_path, channels_list):
    # Read file with same library how it is saved
    data_load = torch.load(file_path)

    # Check if channels name exist on file read and extract it or set to value of channels_list
    try:
        channels_name = [data_load['CHANNELS_NAME'][i] for i in channels_list] 
    except KeyError:
        print("Channel Name not found.")
        if channels_list is not None:
            channels_name = [str(channels_list[i]) for i in range(len(channels_list))]
        else:
            tmp = torch.arange(data_load['DATA'].shape[0])
            channels_name = [str(tmp[i].item()) for i in range(tmp.shape[0])]

    return channels_name
