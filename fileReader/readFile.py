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
    except(KeyError):
        print("Label not found: assume all events of class 0 (normal)!")
        label = torch.zeros(len(timestamp)).tolist()

    return data,label,timestamp