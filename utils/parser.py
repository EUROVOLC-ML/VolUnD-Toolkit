import ast
from pathlib import Path
import os


def training_parse():
    """
    Load args from file. Tries to convert them to best type.
    """
    try:
        # Process input
        hyperparams_path = Path('./')
        if not hyperparams_path.exists():
            raise OSError('Setup dir not found')
        if hyperparams_path.is_dir():
            hyperparams = os.path.join(hyperparams_path, 'trainingSetup.txt')
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
            if not all(key in output.keys() for key in ['train_dir',
                                                        'val_dir',
                                                        'data_location',
                                                        'chunk_len',
                                                        'chunk_only_one',
                                                        'chunk_rate',
                                                        'chunk_random_crop',
                                                        'data_sampling_frequency',
                                                        'chunk_linear_subsample',
                                                        'chunk_butterworth_lowpass',
                                                        'chunk_butterworth_highpass',
                                                        'chunk_butterworth_order',
                                                        'channels_list',
                                                        'channels_name',
                                                        'batch_size',
                                                        'data_provider',
                                                        'mean',
                                                        'std',
                                                        'training_labels',
                                                        'validation_labels',
                                                        'tag',
                                                        'log_dir',
                                                        'plot_every',
                                                        'log_every',
                                                        'save_every',
                                                        'tensorboard_enable',
                                                        'tensorboard_port',
                                                        'layers_base',
                                                        'channels_base',
                                                        'min_spatial_size',
                                                        'start_dilation',
                                                        'min_sig_dil_ratio',
                                                        'max_channels',
                                                        'h_size',
                                                        'enable_variational',
                                                        'optim',
                                                        'reduce_lr_every',
                                                        'reduce_lr_factor',
                                                        'weight_decay',
                                                        'epochs',
                                                        'lr',
                                                        'device',
                                                        'checkpoint']):
                raise AttributeError("Params consistency broken!")
    except (FileNotFoundError, AttributeError, Exception):
        print("Restoring original params value in the setup file... please try to reconfigure setup.")
        f = open(os.path.join(hyperparams_path, 'trainingSetup.txt'), 'w')
        f.write("##############################################################################\n\
# TRAINING SETUP: please don't delete or modify attributes name (before ':') #\n\
##############################################################################\n\
\n\
# Dataset options\n\
train_dir: './dataset/trainingSet'\n\
val_dir: './dataset/validationSet'\n\
data_location: './path/to/data/'\n\
chunk_len: 512\n\
chunk_only_one: False\n\
chunk_rate: 1\n\
chunk_random_crop: False\n\
data_sampling_frequency: None\n\
chunk_linear_subsample: 1\n\
chunk_butterworth_lowpass: None\n\
chunk_butterworth_highpass: None\n\
chunk_butterworth_order: 2\n\
channels_list: None\n\
channels_name: None\n\
batch_size: 128\n\
data_provider: 'ram'\n\
mean: None\n\
std: None\n\
training_labels: [0]\n\
validation_labels: None\n\
\n\
# Experiment options\n\
tag: 'ae'\n\
log_dir: './logs'\n\
plot_every: 1000\n\
log_every: 10\n\
save_every: 10\n\
tensorboard_enable: True\n\
tensorboard_port: 6006\n\
\n\
# Model options\n\
layers_base: 1\n\
channels_base: 16\n\
min_spatial_size: 2\n\
start_dilation: 3\n\
min_sig_dil_ratio: 50\n\
max_channels: 1024\n\
h_size: 64\n\
enable_variational: False\n\
\n\
# Training options\n\
optim: 'Adam'\n\
reduce_lr_every: None\n\
reduce_lr_factor: 0.1\n\
weight_decay: 0.0005\n\
epochs: 32000\n\
lr: 0.00001\n\
device: 'cuda'\n\
\n\
# Checkpoints restore options\n\
checkpoint: None")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output

def detection_parse():
    """
    Load args from file. Tries to convert them to best type.
    """
    try:
        # Process input
        hyperparams_path = Path('./')
        if not hyperparams_path.exists():
            raise OSError('Setup dir not found')
        if hyperparams_path.is_dir():
            hyperparams = os.path.join(hyperparams_path, 'detectionSetup.txt')
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
                                                        'detection_dir',
                                                        'data_location',
                                                        'chunk_len',
                                                        'chunk_only_one',
                                                        'chunk_rate',
                                                        'chunk_random_crop',
                                                        'data_sampling_frequency',
                                                        'chunk_linear_subsample',
                                                        'chunk_butterworth_lowpass',
                                                        'chunk_butterworth_highpass',
                                                        'chunk_butterworth_order',
                                                        'channels_list',
                                                        'channels_name',
                                                        'batch_size',
                                                        'data_provider',
                                                        'mean',
                                                        'std',
                                                        'original_labels',
                                                        'detection_labels',
                                                        'threshold_percentiles',
                                                        'consecutive_outliers',
                                                        'hysteresis',
                                                        'voting',
                                                        'detection_channels_voting',
                                                        'threshold_percentile_voting',
                                                        'consecutive_outlier_voting',
                                                        'hysteresis_voting',
                                                        'device']):
                raise AttributeError("Params consistency broken!")
    except (FileNotFoundError, AttributeError, Exception):
        print("Restoring original params value in the setup file... please try to reconfigure setup.")
        f = open(os.path.join(hyperparams_path, 'detectionSetup.txt'), 'w')
        f.write("##############################################################################\n\
# DETECTION SETUP: please don't delete or modify attributes name (before ':') #\n\
##############################################################################\n\
\n\
# Checkpoints options\n\
checkpoint: './logs/yyyy-mm-dd_hh-mm-ss_ae/'\n\
\n\
# Dataset options\n\
detection_dir: './dataset/detectionSet'\n\
data_location: './path/to/data/'\n\
chunk_len: 512\n\
chunk_only_one: False\n\
chunk_rate: 1\n\
chunk_random_crop: False\n\
data_sampling_frequency: None\n\
chunk_linear_subsample: 1\n\
chunk_butterworth_lowpass: None\n\
chunk_butterworth_highpass: None\n\
chunk_butterworth_order: 2\n\
channels_list: None\n\
channels_name: None\n\
batch_size: 128\n\
data_provider: 'ram'\n\
mean: None\n\
std: None\n\
\n\
# Detection options\n\
original_labels: './path/to/labels'\n\
detection_labels: [2]\n\
threshold_percentiles: None\n\
consecutive_outliers: None\n\
hysteresis: None\n\
voting: False\n\
detection_channels_voting: None\n\
threshold_percentile_voting: None\n\
consecutive_outlier_voting: None\n\
hysteresis_voting: None\n\
\n\
# Model options\n\
device: 'cuda'")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output


def testing_parse():
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
                                                        'train_dir',
                                                        'test_dir',
                                                        'data_location',
                                                        'CDF_mode',
                                                        'chunk_len',
                                                        'chunk_only_one',
                                                        'chunk_rate',
                                                        'chunk_random_crop',
                                                        'data_sampling_frequency',
                                                        'chunk_linear_subsample',
                                                        'chunk_butterworth_lowpass',
                                                        'chunk_butterworth_highpass',
                                                        'chunk_butterworth_order',
                                                        'channels_list',
                                                        'channels_name',
                                                        'batch_size',
                                                        'data_provider',
                                                        'mean',
                                                        'std',
                                                        'training_labels',
                                                        'test_labels',
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
train_dir: './dataset/trainingSet'\n\
test_dir: './dataset/testSet'\n\
data_location: './path/to/data/'\n\
CDF_mode: False\n\
chunk_len: 512\n\
chunk_only_one: False\n\
chunk_rate: 1\n\
chunk_random_crop: False\n\
data_sampling_frequency: None\n\
chunk_linear_subsample: 1\n\
chunk_butterworth_lowpass: None\n\
chunk_butterworth_highpass: None\n\
chunk_butterworth_order: 2\n\
channels_list: None\n\
channels_name: None\n\
batch_size: 128\n\
data_provider: 'ram'\n\
mean: None\n\
std: None\n\
training_labels: [0]\n\
test_labels: None\n\
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


def visualization_parse():
    """
    Load args from file. Tries to convert them to best type.
    """
    try:
        # Process input
        hyperparams_path = Path('./')
        if not hyperparams_path.exists():
            raise OSError('Setup dir not found')
        if hyperparams_path.is_dir():
            hyperparams = os.path.join(hyperparams_path, 'visualizationSetup.txt')
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
            if not all(key in output.keys() for key in ['logs_dir',
                                                        'tensorboard_port']):
                raise AttributeError("Params consistency broken!")
    except (FileNotFoundError, AttributeError, Exception):
        print("Restoring original params value in the setup file... please try to reconfigure setup.")
        f = open(os.path.join(hyperparams_path, 'visualizationSetup.txt'), 'w')
        f.write("###################################################################################\n\
# VISUALIZATION SETUP: please don't delete or modify attributes name (before ':') #\n\
###################################################################################\n\
\n\
# Logs options\n\
logs_dir: './logs'\n\
tensorboard_port: 6006")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output