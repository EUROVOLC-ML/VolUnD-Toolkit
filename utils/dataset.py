import torch
from torch.utils.data import Dataset as TorchDataset
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # fix ctrl+c scipy bug
from tqdm import tqdm
from random import randint
from scipy import signal
from fileReader.readFile import read_file, read_file_info
import json

if os.name == 'nt':
    import win32api
    import win32con
    print("Recognise Windows Mode")
else:
    print("Recognise Unix Mode")


def is_not_hidden(f):
    if os.name == 'nt':
        attribute = win32api.GetFileAttributes(f)
        return not(attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM))
    else:  # linux-osx
        return not(f.startswith('.'))


class FSProvider(TorchDataset):
    """
    Data provider from file system
    """

    def __init__(self, data_dir, data_location, chunk_len, chunk_only_one=False, chunk_rate=1, chunk_random_crop=False, data_sampling_frequency=None, chunk_linear_subsample=1, chunk_butterworth_lowpass=None, chunk_butterworth_highpass=None, chunk_butterworth_order=2, channels_list=None, channels_name=None, cache_dir='./cache', labels=None):
        """
        Args:
        - data_dir (string): path to directory containing files
        - data_location (string): path to directory containing files (if data_dir is a file list)
        - chunk_len (int): length of each item returned by the dataset
        - chunk_only_one (boolean): take one or all chunk of single signal
        - chunk_rate (int): if chunk_only_one=False, take one chunk every chunk_rate
        - chunk_random_crop (boolean): if chunk_only_one=True, take one chunk randomly in single signal
        - data_sampling_frequency (int): set frequency (Hz) of input signals 
        - chunk_linear_subsample (int): apply linear subsample to sigle signal, MUST BE POWER OF 2 (1,2,4,8,16,32,64,128...)
        - chunk_butterworth_lowpass (int): if not None, apply butterworth low pass filter at chunk_butterworth_lowpass Hz
        - chunk_butterworth_highpass (int): if not None, apply butterworth high pass filter at chunk_butterworth_highpass Hz
        - chunk_butterworth_order (int): set order of butterworth filter
        - channels_list (list of int): if not None, select channel (with given index) from data
        - channels_name (list of string): if not None, set name of channels (stations)
        - cache_dir (string): path to directory where dataset information are cached
        - labels (list of int): if not None, use only data (with given integer) of normal activity
        """

        # Store args
        self.data_location = os.path.abspath(data_location)
        self.data_dir = os.path.abspath(data_dir)
        self.chunk_len = chunk_len
        self.chunk_only_one = chunk_only_one
        self.chunk_rate = chunk_rate
        self.chunk_random_crop = chunk_random_crop
        self.data_sampling_frequency = data_sampling_frequency
        self.chunk_linear_subsample = chunk_linear_subsample
        self.cache_dir = os.path.abspath(cache_dir)
        self.channels_list = channels_list
        self.channels_name = channels_name
        self.labels = labels

        # Check data_sampling_frequency
        if (self.data_sampling_frequency is None) or (self.data_sampling_frequency <= 0):
            raise AttributeError("data_sampling_frequency must be greater than 0!")

        # Check linear subsample setup value
        if (self.chunk_linear_subsample & (self.chunk_linear_subsample-1)) != 0:
            raise AttributeError("chunk_linear_subsample must be a power of 2!")
        if self.chunk_linear_subsample >= self.chunk_len:
            raise AttributeError("chunk_linear_subsample must be lower than chunk_len!")

        # Buffer for current file
        self.curr_file_idx = None
        self.curr_file_data = None

        # Check if data_dir is a file list
        if os.path.isfile(self.data_dir):
            file_list = []
            if self.data_dir.lower().endswith('.pt'):
                file_list = torch.load(self.data_dir)[os.path.basename(self.data_dir)[:-3]]
            elif self.data_dir.lower().endswith('.txt'):
                with open(self.data_dir) as infile:
                    for line in infile:
                        file_list.append(line)
            elif self.data_dir.lower().endswith('.json'):
                with open(self.data_dir, "r") as infile:
                    file_list = json.load(infile)[os.path.basename(self.data_dir)[:-5]]
            else:
                raise AttributeError("Broken file list")

            if os.path.isabs(file_list[0]):
                self.files = file_list
            else:
                self.files = [os.path.join(self.data_location, f) for f in file_list if os.path.isfile(os.path.join(
                    self.data_location, f)) and is_not_hidden(os.path.join(self.data_location, f)) and os.path.basename(f)[0] != '.']
        else:
            self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(
                self.data_dir, f)) and is_not_hidden(os.path.join(self.data_dir, f)) and os.path.basename(f)[0] != '.']

        # Check dir
        if len(self.files) == 0:
            raise FileNotFoundError(
                os.path.basename(self.data_dir) + " is empty!")

        # Calculate Butterworth filter setup
        if (chunk_butterworth_highpass is not None) and (chunk_butterworth_lowpass is not None):
            self.butterworth_sos = signal.butter(chunk_butterworth_order, [chunk_butterworth_highpass/(
                0.5*self.data_sampling_frequency), chunk_butterworth_lowpass/(0.5*self.data_sampling_frequency)], analog=False, btype='bandpass', output='sos')
        elif chunk_butterworth_highpass is not None:
            self.butterworth_sos = signal.butter(chunk_butterworth_order, chunk_butterworth_highpass/(
                0.5*self.data_sampling_frequency), analog=False, btype='highpass', output='sos')
        elif chunk_butterworth_lowpass is not None:
            self.butterworth_sos = signal.butter(chunk_butterworth_order, chunk_butterworth_lowpass/(
                0.5*self.data_sampling_frequency), analog=False, btype='lowpass', output='sos')
        else:
            self.butterworth_sos = None

        # Get channels_list
        if self.channels_list is None:
            data_example, _, _ = read_file(self.files[0])
            self.channels_list = torch.arange(data_example.shape[0])
        else:
            self.channels_list = torch.tensor(self.channels_list)
        print("Channels: " + str(self.channels_list.numpy()))
        
        # Get channels name
        if self.channels_name is None:
            self.channels_name = read_file_info(os.path.join(self.data_dir, self.files[0]), self.channels_list)
        elif len(self.channels_name) is not len(read_file_info(os.path.join(self.data_dir, self.files[0]), self.channels_list)):
            raise AttributeError("channels_name must have same length of signal channels!")

        # Get dataset name for cache
        cache_name = os.path.basename(self.data_dir).replace(
            '/', '_').replace('\\', '_').replace('.', '_')
        cache_name += f'_fs_{chunk_len}_{chunk_only_one}_{chunk_rate}_{chunk_random_crop}_{chunk_linear_subsample}'
        cache_name += f'_label_{"all" if  labels is None else "".join(str(l) for l in labels).replace(" ", "_")}'
        cache_name += f'_ch_{"all" if  channels_list is None else "".join(str(c) for c in channels_list).replace(" ", "_")}'
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Create setup map
        setup_map = {'files': self.files,
                     'data_dir': os.path.basename(self.data_dir),
                     'labels': self.labels,
                     'channels_list': self.channels_list,
                     'chunk_len': self.chunk_len,
                     'chunk_only_one': self.chunk_only_one,
                     'channels_name': self.channels_name}
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
            print(
                f'Preprocessing dataset list: {os.path.basename(self.data_dir)}')
            # Initialize data map
            self.data_map = []
            # Process each file
            for i, file in enumerate(tqdm(self.files)):
                # Load file

                try:
                    # Read file
                    data, label, timestamp = read_file(file)

                    # Get only selected channels
                    if self.channels_list is not None:
                        # channel can be a list
                        data = data[self.channels_list, :, :]

                    # Check training mode
                    if self.labels is not None:
                        label_list = []
                        for i in range(len(label)):
                            if label[i] in self.labels:
                                label_list.append(i)
                        data = data[:, label_list, :]

                    # Get length
                    length = data.shape[1]
                    sublength = data.shape[2]

                    # Compute chunk starts
                    chunk_starts = range(0, length, 1)
                    chunk_part_starts = range(0, int(sublength/self.chunk_len), 1)

                    # Prepare item info
                    if self.chunk_only_one:
                        chunk_info = [(i, s, 0) for s in chunk_starts]
                    else:
                        chunk_info = [
                            (i, s, s2) for s in chunk_starts for s2 in chunk_part_starts]

                    # Add to data map
                    self.data_map = self.data_map + chunk_info

                except ImportError:
                    print(
                        f'Bad file: {file}. File must be exported by torch lib.')
            # Check if data_map is empty
            if len(self.data_map) == 0:
                raise FileExistsError(
                    f"There isn't any data to use in {self.data_dir} (if labels is setted, please check data labels).")
            # Save data map
            print(f'Saving dataset list: {cache_path}')
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save((self.data_map, setup_map), cache_path)

    def __len__(self):
        return int(len(self.data_map)/self.chunk_rate)

    def __getitem__(self, idx):
        # Index data map
        file_idx, chunk_start, chunk_part_start = self.data_map[idx*self.chunk_rate]
        # Check buffer
        if self.curr_file_idx is not None and self.curr_file_idx == file_idx:
            # Read from buffer
            data, label, timestamp = self.curr_file_data
        else:
            # Load file
            file_name = self.files[file_idx]
            file_path = os.path.join(self.data_dir, file_name)
            data, label, timestamp = read_file(file_path)
            # Save to buffer
            self.curr_file_idx = file_idx
            self.curr_file_data = (data, label, timestamp)

        # Select channel
        data = data[self.channels_list, :, :]

        # Calculate chunk
        if self.chunk_only_one and self.chunk_random_crop:
            delta = randint(0, data.shape[2]-self.chunk_len)
        else:
            delta = 0
        m1 = (chunk_part_start*self.chunk_len) + delta
        m2 = ((chunk_part_start+1)*self.chunk_len) + delta
        point_list = range(m1, m2, self.chunk_linear_subsample)

        # Get chunk
        chunk = data[:, chunk_start, point_list]
        label_chunk = label[chunk_start]
        time_chunk = timestamp[chunk_start] - int(data.shape[2]/self.data_sampling_frequency - m2/self.data_sampling_frequency)

        # Free whole data storage
        chunk = chunk.clone()

        # Apply Butterworth filter
        if self.butterworth_sos is not None:
            chunk = torch.tensor(signal.sosfiltfilt(self.butterworth_sos, chunk).copy())

        # Return
        return chunk, label_chunk, time_chunk

    def get_channels_name(self):
        return self.channels_name

    def get_channels_list(self):
        return self.channels_list


class RAMProvider(TorchDataset):
    """
    Data provider from RAM
    """

    def __init__(self, data_dir, data_location, chunk_len, chunk_only_one=False, chunk_rate=1, chunk_random_crop=False, data_sampling_frequency=None, chunk_linear_subsample=1, chunk_butterworth_lowpass=None, chunk_butterworth_highpass=None, chunk_butterworth_order=2, channels_list=None, channels_name=None, cache_dir='./cache', labels=None):
        """
        Args:
        - data_dir (string): path to directory containing files
        - data_location (string): path to directory containing files (if data_dir is a file list)
        - chunk_len (int): length of each item returned by the dataset
        - chunk_only_one (boolean): take one or all chunk of single signal
        - chunk_rate (int): if chunk_only_one=False, take one chunk every chunk_rate
        - chunk_random_crop (boolean): if chunk_only_one=True, take one chunk randomly in single signal
        - data_sampling_frequency (int): set frequency (Hz) of input signals 
        - chunk_linear_subsample (int): apply linear subsample to sigle signal, MUST BE POWER OF 2 (1,2,4,8,16,32,64,128...)
        - chunk_butterworth_lowpass (int): if not None, apply butterworth low pass filter at chunk_butterworth_lowpass Hz
        - chunk_butterworth_highpass (int): if not None, apply butterworth high pass filter at chunk_butterworth_highpass Hz
        - chunk_butterworth_order (int): set order of butterworth filter
        - channels_list (list of int): if not None, select channel (with given index) from data
        - channels_name (list of string): if not None, set name of channels (stations)
        - cache_dir (string): path to directory where dataset information are cached
        - labels (list of int): if not None, use only data (with given integer) of normal activity
        """

        # Store args
        self.cache_dir = os.path.abspath(cache_dir)

        # Check if data_dir is a file list
        if os.path.isfile(data_dir):
            file_list = []
            if data_dir.lower().endswith('.pt'):
                file_list = torch.load(data_dir)[os.path.basename(data_dir)[:-3]]
            elif data_dir.lower().endswith('.txt'):
                with open(data_dir) as infile:
                    for line in infile:
                        file_list.append(line)
            elif data_dir.lower().endswith('.json'):
                with open(data_dir, "r") as infile:
                    file_list = json.load(infile)[os.path.basename(data_dir)[:-5]]
            else:
                raise AttributeError("Broken file list")

            if os.path.isabs(file_list[0]):
                self.files = file_list
            else:
                self.files = [os.path.join(data_location, f) for f in file_list if os.path.isfile(os.path.join(
                    data_location, f)) and is_not_hidden(os.path.join(data_location, f)) and os.path.basename(f)[0] != '.']
        else:
            self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(
                data_dir, f)) and is_not_hidden(os.path.join(data_dir, f)) and os.path.basename(f)[0] != '.']

        # Get dataset name for cache
        cache_name = os.path.basename(data_dir).replace('/', '_').replace('\\', '_').replace('.', '_')
        cache_name += f'_ram_{chunk_len}_{chunk_only_one}_{chunk_rate}_{chunk_random_crop}_{chunk_linear_subsample}'
        cache_name += f'_label_{"all" if  labels is None else "".join(str(l) for l in labels).replace(" ", "_")}'
        cache_name += f'_ch_{"all" if  channels_list is None else "".join(str(c) for c in channels_list).replace(" ", "_")}'
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Create setup map
        setup_map = {'files': self.files,
                     'data_dir': os.path.basename(data_dir),
                     'labels': labels,
                     'channels_list_setup': channels_list,
                     'channels_name_setup': channels_name,
                     'chunk_len': chunk_len,
                     'chunk_only_one': chunk_only_one,
                     'chunk_rate': chunk_rate,
                     'chunk_random_crop': chunk_random_crop,
                     'data_sampling_frequency': data_sampling_frequency,
                     'chunk_linear_subsample': chunk_linear_subsample,
                     'chunk_butterworth_lowpass': chunk_butterworth_lowpass,
                     'chunk_butterworth_highpass': chunk_butterworth_highpass,
                     'chunk_butterworth_order': chunk_butterworth_order}
        # Check cache
        reload_cache = True
        if os.path.isfile(cache_path):
            # Load cached data
            print(f'RAMProvider: loading cache {cache_path}')
            self.data, setup_map_loaded = torch.load(cache_path)
            # Save channels info and remove it for comparison
            self.channels_list = setup_map_loaded['channels_list']
            self.channels_name = setup_map_loaded['channels_name']
            del setup_map_loaded['channels_list']
            del setup_map_loaded['channels_name']
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
            fs_provider = FSProvider(data_dir,
                                     data_location=data_location,
                                     chunk_len=chunk_len,
                                     chunk_only_one=chunk_only_one,
                                     chunk_rate=chunk_rate,
                                     chunk_random_crop=chunk_random_crop,
                                     data_sampling_frequency=data_sampling_frequency,
                                     chunk_linear_subsample=chunk_linear_subsample,
                                     chunk_butterworth_lowpass=chunk_butterworth_lowpass,
                                     chunk_butterworth_highpass=chunk_butterworth_highpass,
                                     chunk_butterworth_order=chunk_butterworth_order,
                                     channels_list=channels_list,
                                     channels_name=channels_name,
                                     cache_dir=cache_dir,
                                     labels=labels)
            # Read all files
            for i in tqdm(range(len(fs_provider))):
                # Get data
                data, label, timestamp = fs_provider[i]
                # Add to data
                self.data.append((data, label, timestamp))
            # Read channels info
            self.channels_name = fs_provider.get_channels_name()
            self.channels_list = fs_provider.get_channels_list()
            setup_map['channels_name'] = self.channels_name
            setup_map['channels_list'] = self.channels_list
            # Save data
            print(f'Saving data: {cache_path}')
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save((self.data, setup_map), cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_channels_name(self):
        return self.channels_name

    def get_channels_list(self):
        return self.channels_list


class Dataset(TorchDataset):

    def __init__(self, data_dir, data_location, chunk_len, chunk_only_one=False, chunk_rate=1, chunk_random_crop=False, data_sampling_frequency=None, chunk_linear_subsample=1, chunk_butterworth_lowpass=None, chunk_butterworth_highpass=None, chunk_butterworth_order=2, normalize_params=None, channels_list=None, channels_name=None, cache_dir='./cache', labels=None, provider='ram'):
        """
        Args:
        - data_dir (string): path to directory containing files.
        - data_location (string): path to directory containing files (if data_dir is a file list).
        - chunk_len (int): length of each item returned by the dataset
        - chunk_only_one (boolean): take one or all chunk of single signal
        - chunk_rate (int): if chunk_only_one=False, take one chunk every chunk_rate
        - chunk_random_crop (boolean): if chunk_only_one=True, take one chunk randomly in single signal
        - data_sampling_frequency (int): set frequency (Hz) of input signals 
        - chunk_linear_subsample (int): apply linear subsample to sigle signal, MUST BE POWER OF 2 (1,2,4,8,16,32,64,128...)
        - chunk_butterworth_lowpass (int): if not None, apply butterworth low pass filter at chunk_butterworth_lowpass Hz
        - chunk_butterworth_highpass (int): if not None, apply butterworth high pass filter at chunk_butterworth_highpass Hz
        - chunk_butterworth_order (int): set order of butterworth filter
        - normalize_params (dict): contains tensors with mean and std (if None, don't normalize)
        - channels_list (list of int): if not None, select channel (with given index) from data
        - channels_name (list of string): if not None, set name of channels (stations)
        - cache_dir (string): path to directory where dataset information are cached
        - labels (list of int): if not None, use only data (with given integer) of normal activity
        - provider ('ram'|'fs'): pre-load data on RAM or load from file system
        """

        # Initialize provider
        self.provider = provider
        if self.provider not in ['ram', 'fs']:
            raise AttributeError("Dataset provider must be either 'ram' or 'fs'!")
        if self.provider == 'ram':
            self.provider = RAMProvider(data_dir,
                                        data_location=data_location,
                                        chunk_len=chunk_len,
                                        chunk_only_one=chunk_only_one,
                                        chunk_rate=chunk_rate,
                                        chunk_random_crop=chunk_random_crop,
                                        data_sampling_frequency=data_sampling_frequency,
                                        chunk_linear_subsample=chunk_linear_subsample,
                                        chunk_butterworth_lowpass=chunk_butterworth_lowpass,
                                        chunk_butterworth_highpass=chunk_butterworth_highpass,
                                        chunk_butterworth_order=chunk_butterworth_order,
                                        channels_list=channels_list,
                                        channels_name=channels_name,
                                        cache_dir=cache_dir,
                                        labels=labels)
        elif self.provider == 'fs':
            self.provider = FSProvider(data_dir,
                                       data_location=data_location,
                                       chunk_len=chunk_len,
                                       chunk_only_one=chunk_only_one,
                                       chunk_rate=chunk_rate,
                                       chunk_random_crop=chunk_random_crop,
                                       data_sampling_frequency=data_sampling_frequency,
                                       chunk_linear_subsample=chunk_linear_subsample,
                                       chunk_butterworth_lowpass=chunk_butterworth_lowpass,
                                       chunk_butterworth_highpass=chunk_butterworth_highpass,
                                       chunk_butterworth_order=chunk_butterworth_order,
                                       channels_list=channels_list,
                                       channels_name=channels_name,
                                       cache_dir=cache_dir,
                                       labels=labels)

        # Read channels info
        self.channels_name = self.provider.get_channels_name()
        self.channels_list = self.provider.get_channels_list()
        
        # Store normalization params
        self.normalize_params = normalize_params
        if (self.normalize_params['mean'] is not None) and (self.normalize_params['std'] is not None):
            if isinstance(self.normalize_params['mean'], list) and isinstance(self.normalize_params['std'], list):
                self.norm_mean = torch.Tensor(normalize_params['mean'])
                self.norm_std = torch.Tensor(normalize_params['std'])
            elif os.path.isfile(self.normalize_params['mean']) and os.path.isfile(self.normalize_params['std']):
                if self.normalize_params['mean'].lower().endswith('.pt') and self.normalize_params['std'].lower().endswith('.pt'):
                    self.norm_mean = torch.Tensor(torch.load(normalize_params['mean'])['mean'])
                    self.norm_std = torch.Tensor(torch.load(normalize_params['std'])['std'])
                elif self.normalize_params['mean'].lower().endswith('.json') and self.normalize_params['std'].lower().endswith('.json'):
                    with open(normalize_params['mean'], "r") as infile:
                        self.norm_mean = torch.Tensor(json.load(infile)['mean'])
                    with open(normalize_params['std'], "r") as infile:
                        self.norm_std = torch.Tensor(json.load(infile)['std'])
            # Check length list
            if self.provider[0][0].shape[0] != len(self.norm_mean) and self.provider[0][0].shape[0] != len(self.norm_std):
                raise AttributeError("MEAN and STD list must have same length of channels!")
            print("Normalization params: \n\tMEAN=" + str(self.norm_mean.numpy()) + "\n\tSTD=" + str(self.norm_std.numpy()))

    def __len__(self):
        return len(self.provider)

    def __getitem__(self, idx):
        # Read data
        data, label, timestamp = self.provider[idx]

        # Normalize
        data_tmp = torch.Tensor(data.shape[0], data.shape[1])
        if (self.normalize_params['mean'] is not None) and (self.normalize_params['std'] is not None):
            for i in range(data.shape[0]):
                data_tmp[i] = (data[i] - self.norm_mean[i])/self.norm_std[i]
        else:
            for i in range(data.shape[0]):
                data_tmp[i] = data[i]

        # Return
        return data_tmp, label, timestamp

    def get_channels_name(self):
        return self.channels_name

    def get_channels_list(self):
        return self.channels_list
