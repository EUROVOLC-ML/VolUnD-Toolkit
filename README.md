# VOLcano UNrest Detection (VolUnD) Toolkit

[![VOLUND_LOGO](docs/images/VolUnD-logo.png)](https://github.com/EUROVOLC-ML/VolUnD-Toolkit)

VolUnD is a VOLcano UNrest Detection Toolkit written in Python 3 using the PyTorch library. It performs anomaly detection of
volcanic historical data in an unsupervised way, leveraging the hypothesis that normal activity dominates the
event distribution and can therefore be treated as the main source of information for training a model able to
identify deviations as anomalies. These anomalies could be attributed to changes in the volcano state.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5537253.svg)](https://doi.org/10.5281/zenodo.5537253)

## Table of contents
* [Project Structure](#project-structure)
* [Setup](#setup)
* [Usage](#usage)
* [Notes](#notes)
* [Licensing](#licensing)
* [Meta](#Meta)

## Project Structure

The toolkit is developed in Python 3 using the PyTorch library, and is structured as follows.

The root directory contains the following folders:
- **[cache](cache)** (internal use): storage directory for internally-processed dataset.
- **[dataset](dataset)**: directory containing default locations for train/validation/test files. Each directory can contain an arbitrary number of files, each of which must be saved in PyTorch format (using torch.save) in dictionary format, containing the following keys:
  - CHANNELS_NAME (optional): list of name for each channel,
  - TIME_DESC (optional): natural-language description of the temporal interval represented in the file,
  - DATA: float tensor of size "stations × number of signals × chunk length",
  - LABEL (optional): 0 for no activity or normal activity, 1 for e.g. mild volcanic activity, 2 for e.g. energetic eruptive activity ; if not provided, non-normal events will not be emphasized during visualization,
  - TIMESTAMP: list of Unix timestamps of size “number of signals”, corresponding to the end of the signals in DATA.
- **[docs](docs)**: directory containing the IPython Notebook to export the dataset to be used as input and logos.
- **[fileReader](fileReader)** (for Advanced user): directory containing the function to read dataset files. Modify it if you want to use your own dataset files (without any adaptation to our format).
- **[logs](logs)**: directory where training sessions are saved.
- **[utils](utils)** (internal use): directory containing the main source code.
	
The main files in the toolkit are:
- **[training.py](training.py)**: starts the training phase; a web dashboard will also be launched where it is possible to monitor training progress through various plots.
- **[visualization.py](visualization.py)**: starts an instance of the backend to view past training sessions on the web dashboard sessions.
- **[testing.py](testing.py)**: shows reconstruction distances on test data.
- **[detection.py](detection.py)**: starts inference and detection of the alarms.
- **[detection_plotter.py](detection_plotter.py)**: shows an interactive plot of a specific key (channel, consecutive outliers, hysteresis). Parameters: `--det_dict_path ./path/to/detection_dict.pt [--save_plots]`
- **[trainingSetup.txt](trainingSetup.txt)**: configures the training options. Each option is specified in a single line, using the following syntax: 
		key: value, where “key” is an option name, and “value” is the corresponding value. String values should be quoted; numeric values should not be quoted; unspecified values can be provided as “None” (unquoted); list values can be grouped between brackets.
		Possible options are:
  - train_dir: folder (or file list in PyTorch or JSON format) where the dataset files for the training phase are located
  - val_dir: folder (or file list in PyTorch or JSON format) where the dataset files for the validation phase are located
  - data_location: folder where the dataset files are located, used only if train_dir or val_dir are file lists
  - chunk_len: chunk length (i.e., temporal length of a single input to the model); default 512
  - chunk_only_one: take one or all chunk of single signal; default False
  - chunk_rate: if chunk_only_one=False, take one chunk every chunk_rate; default 1
  - chunk_random_crop: if chunk_only_one=True, take one chunk randomly in single signal; default False
  - data_sampling_frequency: set frequency (Hz) of input signals
  - chunk_linear_subsample: apply linear subsample to single signal, MUST BE A POWER OF 2 (1,2,4,8,16,32,64,128...); default 1 (not apply linear subsample)
  - chunk_butterworth_lowpass: apply butterworth low pass filter at this frequency (Hz); default None (not apply low pass filter)
  - chunk_butterworth_highpass: apply butterworth high pass filter at this frequency (Hz); default None (not apply high pass filter)
  - chunk_butterworth_order: set order of butterworth filter; default 2
  - channels_list: if present, list of channels (i.e., input stations) to use; default “None”
  - channels_name: if present, list of channels name (i.e., name of input stations); default “None”
  - batch_size: mini-batch size for gradient descent; default 128
  - data_provider: specifies whether data should be stored on RAM (faster; value “ram”) or should be read from the filesystem (slower; value “fs”); default “ram”
  - mean: if not None, list (or file in PyTorch or JSON format) of per-channel means for standardization; default None
  - std: if not None, list (or file in PyTorch or JSON format) of per-channel standard deviations for standardization; default None
  - training_labels: list of normal activity labels to train model, if None select all available labels; default [0]
  - validation_labels: list of activity labels to validate model, if None select all available labels; default None
  - tag: name to assign to the training session in the web dashboard
  - log_dir: folder where to save the training data
  - plot_every: defines how often (number of iterations) dashboard figures (inputs, reconstructions) should be updated; default 1000
  - log_every: defines how often (number of iterations) dashboard plots (loss, accuracy) should be updated; default 10
  - save_every: defines how often (number of epochs) the model should be saved; default 10
  - tensorboard_enable: start TensorBoard daemon to visualize data on browser; default True
  - tensorboard_port: set tensorboard port to view telemetry on browser; default 6006
  - layers_base: in the model, number of convolution layers to be applied before down-sampling or up-sampling; default 1
  - channels_base: in the model, initial number of channels computed from the input signal; default 16
  - min_spatial_size: in the model, minimum temporal size (in spite of the name), under which down-sampling should not be performed; default 2
  - start_dilation: in the model, initial dilation values in the encoder’s convolutional layers; default 3
  - min_sig_dil_ratio: in the model, minimum ratio between temporal length of the signal at each layer and the corresponding dilation value; which the ratio is smaller, dilation is reduced; default 50
  - max_channels: in the model, channels are doubled at each down-sampling or up-sampling layer, until the maximum number of channels is reached; default 1024
  - h_size: in the model, size of the representation at the bottleneck; default 64
  - enable_variational: choose whether to use AE (False) or VAE (True); default False
  - optim: optimizer to use; default Adam
  - reduce_lr_every: defines how often (number of epochs) learning rate should be reduced; default None
  - reduce_lr_factor: defines the factor by which the learning rate should be reduced; default 0.1
  - weight_decay: weight for L2 regularization; default 0.0005
  - epochs: number of total training epochs; default 32000
  - lr: starting learning rate; default 0.00001
  - device: processor to use for training (cpu or cuda); default “cuda”
  - checkpoint: checkpoint folder to continue previous training (it can be a specific checkpoint file or the folder containing all checkpoints; in this case, the best checkpoint based on training loss will be selected)
- **[visualizationSetup.txt](visualizationSetup.txt)**: to configure visualization parameters. Parameters:
  - logs_dir: folder where to find the previously saved training sessions
  - tensorboard_port: set tensorboard port to view telemetry on browser; default 6006
- **[testingSetup.txt](testingSetup.txt)**: to configure the testing parameters. Parameters:
  - checkpoint: model to be validated (it can be a specific checkpoint file or the folder containing all checkpoints; in this case, the best checkpoint based on training loss will be selected)
  - train_dir: as above
  - test_dir: folder where the dataset files for the testing phase are located
  - data_location: as above
  - CDF_mode: show reconstruction distances as percentage using Probability Density Function derived from Cumulative Distribution Function of trainingSet
  - chunk_len: as above
  - chunk_only_one: as above
  - chunk_rate: as above
  - chunk_random_crop: as above
  - data_sampling_frequency: as above
  - chunk_linear_subsample: as above
  - chunk_butterworth_lowpass: as above
  - chunk_butterworth_highpass: as above
  - chunk_butterworth_order: as above
  - channels_list: as above
  - channels_name: as above
  - batch_size: as above
  - data_provider: as above
  - mean: as above
  - std: as above
  - training_labels: as above
  - test_labels: list of activity labels to test model, if None select all available labels; default None
  - label_activity: list of pre-eruption activity labels; default [1]
  - label_eruption: list of eruption activity labels; default [2]
  - device: as above
  - img_quality: dpi of graph's image saved on logs/yyyy-mm-dd_hh-mm-ss_.../testing
  - web_port: set Tornado port to view result on browser; default 8988 
- **[detectionSetup.txt](detectionSetup.txt)**: to configure detection parameters. Parameters:
  - checkpoint: as above
  - detection_dir: folder where the dataset files for the detection phase are located
  - data_location: as above
  - chunk_len: as above
  - chunk_only_one: as above
  - chunk_rate: as above
  - chunk_random_crop: as above
  - data_sampling_frequency: as above
  - chunk_linear_subsample: as above
  - chunk_butterworth_lowpass: as above
  - chunk_butterworth_highpass: as above
  - chunk_butterworth_order: as above
  - channels_list: as above
  - channels_name: as above
  - batch_size: as above
  - data_provider: as above
  - mean: as above
  - std: as above
  - original_labels: dict with key "LABELS_LIST", list of original labels and key "DATETIME_LIST", list of original corresponding datetime. More information on creating this dictionary can be found in "[EtnaDataset_Creation.ipynb](docs/EtnaDataset_Creation.ipynb)".
  - detection_labels: list of activity labels to be detected, if None select [2]; default None
  - threshold_percentiles: list of percentiles for which to calculate the thresholds, for each channel, that must be exceeded by reconstruction distance
  - consecutive_outliers: list of consecutive outliers to consider an alarm
  - hysteresis: list of hysteresis to extend an alarm in hours
  - voting: choose whether channels must vote to consider an alarm (True) or if alarms must be considered separated for each channel (False); default False
  - threshold_percentile_voting: list of percentiles, one for each channel, for which to calculate the thresholds that must be exceeded by reconstruction distance
  - consecutive_outliers_voting: list of consecutive outliers to consider an alarm
  - hysteresis_voting: list of hysteresis to extend an alarm in hours
- **[requirements.txt](requirements.txt)**: can be used to install the modules necessary for the correct functioning of the toolkit by running the following command: “pip install -r requirements.txt”

## Setup
This software requires python3. Refer to [requirements.txt](requirements.txt) to install the necessary packages:
- tqdm
- numpy
- pytorch
- torchvision
- tensorboard
- matplotlib
- scipy
- statsmodels
- pandas

Is possible to use the preferred method such as pip, conda, etc.

To install the required modules using **pip**, you can type the following:
>```pip3 install -r /path/to/requirements.txt```

## Usage
Configure the corresponding setup.txt file and then run the desired script. Example: `python training.py`
- [trainingSetup.txt](trainingSetup.txt) refers to [training.py](training.py)
- [testingSetup.txt](testingSetup.txt) refers to [testing.py](testing.py)
- [visualizationSetup.txt](visualizationSetup.txt) refers to [visualization.py](visualization.py)
- [detectionSetup.txt](detectionSetup.txt) refers to [detection.py](detection.py)

For `detection_plotter.py`, the command is: `python detection_plotter.py --det_dict_path ./path/to/detection_dict.pt [--save_plots]`

## Notes
If you make change to dataset, remember to empty cache folder.

## Licensing
Copyright (c) 2021, all persons listed in [CONTRIBUTORS.md](CONTRIBUTORS.md) in alphabetical order.

This project is licensed under the [EUPL, v1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12). See [LICENSE](LICENSE) for more information.

## Meta

[![INGV_LOGO](docs/images/INGV_OE-logo.png)](https://www.ct.ingv.it/)

[![PERCEIVE_LOGO](docs/images/PERCEIVE-logo.png)](http://www.perceivelab.com/index)

This software was developed and made available as part of the **EUROVOLC** project. 

For more information see [EuroVolk Wiki - Open software](https://eurovolc.cp.dias.ie/index.php/Open_software)

[![EUROVOLC_LOGO](docs/images/EUROVOLC-logo.jpg)](https://eurovolc.eu)

## Cite this code
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5537253.svg)](https://doi.org/10.5281/zenodo.5537253)

Raffaele Mineo, Paolo Spadaro, Flavio Cannavò, Simone Palazzo, Andrea Cannata, & Concetto Spampinato. (2021). EUROVOLC-ML/VolUnD-Toolkit. Zenodo. https://doi.org/10.5281/zenodo.5537253
