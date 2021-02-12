import torch
from torch.utils.data import Dataset as TorchDataset
import os
from tqdm import tqdm

class FSProvider(TorchDataset):
    """
    Data provider from file system
    """
    def __init__(self, data_dir, chunk_len, chunk_jump=False, chunk_rate=1, channels_list=None, cache_dir='./Cache'):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - chunk_len (int): length of each item returned by the dataset
        - chunk_jump (boolean): jump between consecutive chunks or get all remain part after cut data in chunk
        - chunk_rate (int): take one chunk every chunk_rate consecutive
        - single_channel (int): select single channel (with given index) from data
        - cache_dir (string): path to directory where dataset information are cached
        """

        # Store args
        self.data_dir = os.path.abspath(data_dir)
        self.chunk_len = chunk_len
        self.chunk_jump = chunk_jump
        self.chunk_rate = chunk_rate
        self.cache_dir = os.path.abspath(cache_dir)
        self.channels_list = channels_list
        
        # Buffer for current file
        self.curr_file_idx = None
        self.curr_file_data = None

        # List files
        self.files = sorted(os.listdir(self.data_dir))

        # Get dataset name for cache
        cache_name = self.data_dir.replace('/', '_').replace('\\', '_')
        cache_name += f'_fs_{chunk_len}_{chunk_jump}'
        cache_name += f'_{"all" if channels_list is None else channels_list}'
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Check cache
        if os.path.isfile(cache_path):
            # Load cached data
            self.data_map = torch.load(cache_path)
        else:
            # Preprocess dataset
            print(f'Preprocessing dataset list: {self.data_dir}')
            # Initialize data map
            self.data_map = []
            # Process each file
            for i,file in enumerate(tqdm(self.files)):
                # Load file
                file_path = os.path.join(self.data_dir, file)
            
                try:
                    # Read file
                    data,label,timestamp=self.read_torch(file_path)

                    # Get length
                    length = data.shape[1]
                    sublenght = data.shape[2]

                    # Get only selected channels
                    if self.channels_list is not None:
                        #channel can be a list
                        data=data[self.channels_list,:,:]

                    # Compute chunk starts
                    chunk_starts = range(0, length, 1)
                    chunk_part_starts = range(0, int(sublenght/self.chunk_len), 1)

                    # Prepare item info
                    if self.chunk_jump:
                        chunk_info = [(i,s,0) for s in chunk_starts]
                    else:
                        chunk_info = [(i,s,s2) for s in chunk_starts for s2 in chunk_part_starts]
                 
                    # Add to data map
                    self.data_map = self.data_map + chunk_info
                   
                except ImportError:
                    print(f'Bad file: {file_path}')
            # Save data map
            print(f'Saving dataset list: {cache_path}')
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(self.data_map, cache_path)
    
    def read_torch(self, file_path):
        data_load = torch.load(file_path)
        data = data_load['DATA']
        timestamp = data_load['TIMESTAMP']
        try:
            label = data_load['LABEL']
        except(KeyError):
            print("Label not found: assume all events of class 0 (normal)!")
            label = torch.zeros(len(timestamp)).tolist()
        return data,label,timestamp
    
    def __len__(self):
        return int(len(self.data_map)/self.chunk_rate)
    
    def __getitem__(self, idx):
        # Index data map
        file_idx,chunk_start,chunk_part_start = self.data_map[idx*self.chunk_rate]
        # Check buffer
        if self.curr_file_idx is not None and self.curr_file_idx == file_idx:
            # Read from buffer
            data,label,timestamp = self.curr_file_data
        else:
            # Load file
            file_name = self.files[file_idx]
            file_path = os.path.join(self.data_dir, file_name)
            data,label,timestamp = self.read_torch(file_path)
            # Save to buffer
            self.curr_file_idx = file_idx
            self.curr_file_data = (data,label,timestamp)

        # Select channel
        if self.channels_list is not None:
            data = data[self.channels_list,:,:]

        # Get chunk
        chunk = data[:,chunk_start,chunk_part_start*self.chunk_len:(chunk_part_start+1)*self.chunk_len]
        label_chunk = label[chunk_start]
        time_chunk = timestamp[chunk_start]
        
        # Free whole data storage
        chunk = chunk.clone()
     
        # Return
        return chunk,label_chunk,time_chunk

class RAMProvider(TorchDataset):
    """
    Data provider from RAM
    """
    def __init__(self, data_dir, chunk_len, chunk_jump=False, chunk_rate=1, channels_list=None, cache_dir='./Cache'):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - chunk_len (int): length of each item returned by the dataset
        - chunk_jump (boolean):  jump between consecutive chunks or get all remain part after cut data in chunk
        - chunk_rate (int): take one chunk every chunk_rate consecutive
        - single_channel (int): select single channel (with given index) from data
        - cache_dir (string): path to directory where dataset information are cached
        """

        # Store args
        self.cache_dir = os.path.abspath(cache_dir)

        # Get dataset name for cache
        cache_name = os.path.abspath(data_dir).replace('/', '_').replace('\\', '_')
        cache_name += f'_ram_{chunk_len}_{chunk_jump}'
        cache_name += f'_{"all" if  channels_list is None else channels_list}'
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Check cache
        if os.path.isfile(cache_path):
            # Load cached data
            print(f'RAMProvider: loading cache {cache_path}')
            self.data = torch.load(cache_path)
        else:
            print('RAMProvider: reading all files')
            # Initialize data
            self.data = []
            # Create FS provider
            fs_provider = FSProvider(data_dir, chunk_len, chunk_jump, chunk_rate, channels_list, cache_dir)
            # Read all files
            for i in tqdm(range(len(fs_provider))):
                # Get data
                data,label,timestamp = fs_provider[i]
                # Add to data
                self.data.append((data,label,timestamp))
            
            # Save data
            print(f'Saving data: {cache_path}')
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(self.data, cache_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return
        return self.data[idx]

class Dataset(TorchDataset):
    
    def __init__(self, data_dir, chunk_len, chunk_jump=False, chunk_rate=1, normalize_params=None, channels_list=None, cache_dir='./Cache', provider='ram'):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - normalize (dict): contains tensors with mean and std (if None, don't normalize)
        - chunk_len (int): length of each item returned by the dataset
        - chunk_jump (boolean):  jump between consecutive chunks or get all remain part after cut data in chunk
        - chunk_rate (int): take one chunk every chunk_rate consecutive
        - single_channel (int): select single channel (with given index) from data
        - cache_dir (string): path to directory where dataset information are cached
        - provider ('ram'|'fs'): pre-load data on RAM or load from file system
        """

        # Initialize provider
        self.provider = provider
        assert self.provider in ['ram', 'fs'], "Dataset provider must be either 'ram' or 'fs'!"
        if self.provider == 'ram':
            self.provider = RAMProvider(data_dir, chunk_len, chunk_jump, chunk_rate, channels_list, cache_dir)
        elif self.provider == 'fs':
            self.provider = FSProvider(data_dir, chunk_len, chunk_jump, chunk_rate, channels_list, cache_dir)

        # Store normalization params
        self.normalize_params = normalize_params
        if (self.normalize_params['mean'] is not None) and (self.normalize_params['std'] is not None):
            self.norm_mean = torch.FloatTensor(normalize_params['mean'])
            self.norm_std =torch.FloatTensor(normalize_params['std'])
            # Check lenght list
            if self.provider[0][0].shape[0] != len(self.norm_mean) and self.provider[0][0].shape[0] != len(self.norm_std):
                raise AttributeError("MEAN and STD list must have same lenght of channels!")
            print("Normalization params: MEAN=" + str(self.norm_mean) + " & STD="+ str(self.norm_std))
                
    def __len__(self):
        return len(self.provider)
    
    def __getitem__(self, idx):
        # Read data
        data,label,timestamp = self.provider[idx]
        
        # Normalize
        data_tmp = torch.Tensor(data.shape[0],data.shape[1])
        if (self.normalize_params['mean'] is not None) and (self.normalize_params['std'] is not None):
            for i in range(data.shape[0]):
                data_tmp[i] = (data[i] - self.norm_mean[i])/self.norm_std[i]
        else:
            for i in range(data.shape[0]):
                data_tmp[i] = data[i]
        
        # Return
        return data_tmp,label,timestamp
