import torch
from torch.utils.data import Dataset as TorchDataset
import os
from tqdm import tqdm
from random import randint
from fileReader.readFile import read_file

if os.name=='nt':
    import win32api, win32con
    print("Recognise Windows Mode")
else:
    print("Recognise Unix Mode")
def is_not_hidden(f):
    if os.name=='nt':
        attribute = win32api.GetFileAttributes(f)
        return not(attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM))
    else: #linux-osx
        return not(f.startswith('.'))

class FSProvider(TorchDataset):
    """
    Data provider from file system
    """
    def __init__(self, data_dir, chunk_len, chunk_only_one=False, chunk_rate=1, chunk_random_crop=False, chunk_linear_subsample=1, channels_list=None, cache_dir='./cache', training_labels=None):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - chunk_len (int): length of each item returned by the dataset
        - chunk_only_one (boolean): take one or all chunk of single signal
        - chunk_rate (int): if chunk_only_one=False, take one chunk every chunk_rate
        - chunk_random_crop (boolean): if chunk_only_one=True, take one chunk randomly in single signal
        - chunk_linear_subsample (int): apply linear subsample to sigle signal, MUST BE A POWER OF 2 (1,2,4,8,16,32,64,128...)
        - channel_list (list of int): if not None, select channel (with given index) from data
        - cache_dir (string): path to directory where dataset information are cached
        - training_labels (list of int): if not None, use only data (with given integer) of normal activity
        """

        # Store args
        self.data_dir = os.path.abspath(data_dir)
        self.chunk_len = chunk_len
        self.chunk_only_one = chunk_only_one
        self.chunk_rate = chunk_rate
        self.chunk_random_crop = chunk_random_crop
        self.chunk_linear_subsample = chunk_linear_subsample
        self.cache_dir = os.path.abspath(cache_dir)
        self.channels_list = channels_list
        self.training_labels = training_labels

        #Check linear subsample setup value
        if (self.chunk_linear_subsample & (self.chunk_linear_subsample-1)) != 0:
            raise AttributeError("chunk_linear_subsample must be a power of 2!")
        if self.chunk_linear_subsample >= self.chunk_len:
            raise AttributeError("chunk_linear_subsample must be lower than chunk_len!")
        
        # Buffer for current file
        self.curr_file_idx = None
        self.curr_file_data = None

        # List files
        self.files = sorted([f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)) and is_not_hidden(os.path.join(self.data_dir, f)) and os.path.basename(f)[0] != '.'])

        # Check dir
        if len(self.files) == 0:
            raise FileNotFoundError(self.data_dir + " is empty!")

        # Get dataset name for cache
        cache_name = self.data_dir.replace('/', '_').replace('\\', '_')
        cache_name += f'_fs_{chunk_len}_{chunk_only_one}_{chunk_rate}_{chunk_random_crop}_{chunk_linear_subsample}'
        cache_name += f'_label_{"all" if  training_labels is None else "".join(str(l) for l in training_labels).replace(" ", "_")}'
        cache_name += f'_ch_{"all" if  channels_list is None else "".join(str(c) for c in channel_list).replace(" ", "_")}'
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Create setup map
        setup_map = {'files':self.files,
                    'data_dir':self.data_dir,
                    'training_labels':self.training_labels,
                    'channels_list':self.channels_list,
                    'chunk_len':self.chunk_len,
                    'chunk_only_one':self.chunk_only_one}
        # Check cache
        reload_cache = True
        if os.path.isfile(cache_path):
            # Load cached data
            print(f'FSProvider: loading cache {cache_path}')
            self.data_map, setup_map_loaded = torch.load(cache_path)
            # Check if cache is up to date
            if setup_map == setup_map_loaded:
                reload_cache = False
                print("FSProvider: cache is up to date.")
            else:
                reload_cache = True
                print("FSProvider: cache is out of date. Reloading...")
        if reload_cache:
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
                    data,label,timestamp=read_file(file_path)

                    #Check training mode
                    if self.training_labels is not None:
                        if label not in self.training_labels:
                            continue

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
                    if self.chunk_only_one:
                        chunk_info = [(i,s,0) for s in chunk_starts]
                    else:
                        chunk_info = [(i,s,s2) for s in chunk_starts for s2 in chunk_part_starts]
                 
                    # Add to data map
                    self.data_map = self.data_map + chunk_info
                   
                except ImportError:
                    print(f'Bad file: {file_path}. File must be exported by torch lib.')
            # Check if data_map is empty
            if len(self.data_map) == 0:
                raise FileExistsError(f"There isn't any data to use in {self.data_dir} (if training_labels is setted, please check data labels).")
            # Save data map
            print(f'Saving dataset list: {cache_path}')
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save((self.data_map,setup_map), cache_path)
    
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
            data,label,timestamp = read_file(file_path)
            # Save to buffer
            self.curr_file_idx = file_idx
            self.curr_file_data = (data,label,timestamp)

        # Select channel
        if self.channels_list is not None:
            data = data[self.channels_list,:,:]

        # Calculate chunk
        if self.chunk_only_one and self.chunk_random_crop:
            delta = randint(0,data.shape[2]-self.chunk_len)
        else:
            delta = 0
        m1 = (chunk_part_start*self.chunk_len) + delta
        m2 = ((chunk_part_start+1)*self.chunk_len) + delta
        point_list = range(m1,m2,self.chunk_linear_subsample)
        
        # Get chunk
        chunk = data[:,chunk_start,point_list]
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
    def __init__(self, data_dir, chunk_len, chunk_only_one=False, chunk_rate=1, chunk_random_crop=False, chunk_linear_subsample=1, channels_list=None, cache_dir='./cache', training_labels=None):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - chunk_len (int): length of each item returned by the dataset
        - chunk_only_one (boolean): take one or all chunk of single signal
        - chunk_rate (int): if chunk_only_one=False, take one chunk every chunk_rate
        - chunk_random_crop (boolean): if chunk_only_one=True, take one chunk randomly in single signal
        - chunk_linear_subsample (int): apply linear subsample to sigle signal, MUST BE POWER OF 2 (1,2,4,8,16,32,64,128...)
        - channel_list (list of int): if not None, select channel (with given index) from data
        - cache_dir (string): path to directory where dataset information are cached
        - training_labels (list of int): if not None, use only data (with given integer) of normal activity
        """

        # Store args
        self.cache_dir = os.path.abspath(cache_dir)

        # Get dataset name for cache
        cache_name = os.path.abspath(data_dir).replace('/', '_').replace('\\', '_')
        cache_name += f'_ram_{chunk_len}_{chunk_only_one}_{chunk_rate}_{chunk_random_crop}_{chunk_linear_subsample}'
        cache_name += f'_label_{"all" if  training_labels is None else "".join(str(l) for l in training_labels).replace(" ", "_")}'
        cache_name += f'_ch_{"all" if  channels_list is None else "".join(str(c) for c in channel_list).replace(" ", "_")}'
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Create setup map
        setup_map = {'files':sorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and is_not_hidden(os.path.join(data_dir, f)) and os.path.basename(f)[0] != '.']),
                    'data_dir':data_dir,
                    'training_labels':training_labels,
                    'channels_list':channels_list,
                    'chunk_len':chunk_len,
                    'chunk_only_one':chunk_only_one}
        # Check cache
        reload_cache = True
        if os.path.isfile(cache_path):
            # Load cached data
            print(f'RAMProvider: loading cache {cache_path}')
            self.data, setup_map_loaded = torch.load(cache_path)
            # Check if cache is up to date
            if setup_map == setup_map_loaded:
                reload_cache = False
                print("RAMProvider: cache is up to date.")
            else:
                reload_cache = True
                print("RAMProvider: cache is out of date. Reloading...")
        if reload_cache:
            print('RAMProvider: reading all files')
            # Initialize data
            self.data = []
            # Create FS provider
            fs_provider = FSProvider(data_dir, chunk_len=chunk_len, chunk_only_one=chunk_only_one, chunk_rate=chunk_rate, chunk_random_crop=chunk_random_crop, chunk_linear_subsample=chunk_linear_subsample, channels_list=channels_list, cache_dir=cache_dir, training_labels=training_labels)
            # Read all files
            for i in tqdm(range(len(fs_provider))):
                # Get data
                data,label,timestamp = fs_provider[i]
                # Add to data
                self.data.append((data,label,timestamp))
            
            # Save data
            print(f'Saving data: {cache_path}')
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save((self.data,setup_map), cache_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return
        return self.data[idx]

class Dataset(TorchDataset):
    
    def __init__(self, data_dir, chunk_len, chunk_only_one=False, chunk_rate=1, chunk_random_crop=False, chunk_linear_subsample=1, normalize_params=None, channels_list=None, cache_dir='./cache', training_labels=None, provider='ram'):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - normalize (dict): contains tensors with mean and std (if None, don't normalize)
        - chunk_len (int): length of each item returned by the dataset
        - chunk_only_one (boolean): take one or all chunk of single signal
        - chunk_rate (int): if chunk_only_one=False, take one chunk every chunk_rate
        - chunk_random_crop (boolean): if chunk_only_one=True, take one chunk randomly in single signal
        - chunk_linear_subsample (int): apply linear subsample to sigle signal, MUST BE POWER OF 2 (1,2,4,8,16,32,64,128...)
        - channel_list (list of int): if not None, select channel (with given index) from data
        - cache_dir (string): path to directory where dataset information are cached
        - training_labels (list of int): if not None, use only data (with given integer) of normal activity
        - provider ('ram'|'fs'): pre-load data on RAM or load from file system
        """

        # Initialize provider
        self.provider = provider
        assert self.provider in ['ram', 'fs'], "Dataset provider must be either 'ram' or 'fs'!"
        if self.provider == 'ram':
            self.provider = RAMProvider(data_dir, chunk_len=chunk_len, chunk_only_one=chunk_only_one, chunk_rate=chunk_rate, chunk_random_crop=chunk_random_crop, chunk_linear_subsample=chunk_linear_subsample, channels_list=channels_list, cache_dir=cache_dir, training_labels=training_labels)
        elif self.provider == 'fs':
            self.provider = FSProvider(data_dir, chunk_len=chunk_len, chunk_only_one=chunk_only_one, chunk_rate=chunk_rate, chunk_random_crop=chunk_random_crop, chunk_linear_subsample=chunk_linear_subsample, channels_list=channels_list, cache_dir=cache_dir, training_labels=training_labels)

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
