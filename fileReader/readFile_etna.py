import torch
from datetime import datetime, timedelta
import scipy.io as sio

def read_file(file_path):
    file_name = file_path.split("\\")[-1]
    
    # Read file with same library how it is saved
    tmp = sio.loadmat(file_path)

    # Extract signal data from file read
    data = torch.Tensor(tmp['fragment']['data'][0][0]).unsqueeze(1)

    # Extract timestamp data from file read
    timestamp = []
    timestamp.append((datetime.strptime(file_name[0:6], "%y%m%d") + timedelta(minutes=int(file_name[7:10])*10)).timestamp())

    # Check if label data exist on file read and extract it or set to 0 (normal activity)
    try:
        label = []
        label.append(tmp['fragment']['current_label'][0][0][0][0])
    except(KeyError):
        print("Label not found: assume all events of class 0 (normal)!")
        label = torch.zeros(len(timestamp)).tolist()

    return data,label,timestamp