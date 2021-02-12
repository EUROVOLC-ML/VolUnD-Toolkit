![alt text](https://github.com/EUROVOLC-ML/VolUnD-Toolkit/blob/main/docs/VolUnD-logo.png?raw=true)



# VolUnD Toolkit

VOLcano UNrest Detection



The toolkit is developed in Python 3 using the PyTorch library, and is structured as follows.



The root directory contains the following folders:

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;cache (internal use): storage directory for internally-processed dataset.

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;dataset: directory containing default locations for train/validation/test files. Each directory can contain an arbitrary number of files, each of which must be saved in PyTorch format (using torch.save) in dictionary format, containing the following keys

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;CHANNEL_NAMES: list of names for each channels,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;TIME_DESC: natural-language description of the temporal interval of represented in the file

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;DATA: float tensor of size “stations × number of signals × chunk length

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;LABEL (optional): 0 for no activity or normal activity, 1 for e.g. mild volcanic activity, 2 for e.g. energetic eruptive activity ; if not provided, non-normal events will not be emphasized during visualization 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;TIMESTAMP: list of Unix timestamps of size “number of signals”, corresponding to the signals in DATA

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;logs: directory where training sessions are saved.

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;utils (internal use): directory containing the main source code.

&nbsp;&nbsp;&nbsp;&nbsp;

The main files in the toolkit are:

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;training.py: starts the training phase; a web dashboard will also be launched where it is possible to monitor training progress through various plots.

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;visualization.py: starts an instance of the backend to view past training sessions on the web dashboard sessions.

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;testing.py: shows reconstruction distances on test data.

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;trainingSetup.txt: configures the training options. Each options is specified in a single line, using the following syntax: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;key: value

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where “key” is an option name, and “value” is the corresponding value. String values should be quoted; numeric values should not be quoted; unspecified values can be provided as “None” (unquoted); list values can be grouped between brackets.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Possible options are:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;train_dir: folder where the dataset files for the training phase are located

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;val_dir: folder where the dataset files for the validation phase are located

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;data_len: chunk length (i.e., temporal length of the a single input to the model); default 512

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;chunk_jump: choose if a signal is truncated to data_len or get all part of signal in chunk of data_len; default False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;channels_list: if present, list of channels (i.e., input stations) to use; default “None”

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;batch_size: mini-batch size for gradient descent; default 128

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;data_provider: specifies whether data should be stored on RAM (faster; value “ram”) or should be read from the filesystem (slower; value “fs”); default “ram”

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;mean: if not None, list of per-channel means for standardization; default None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;std: if not None, list of per-channel standard deviations for standardization; default None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;tag: name to assign to the training session in the web dashboard

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;log_dir: folder where to save the training data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;plot_every: defines how often (number of iterations) dashboard figures (inputs, reconstructions) should be updated; default 1000

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;log_every: defines how often (number of iterations) dashboard plots (loss, accuracy) should be updated; default 10

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;save_every: defines how often (number of epochs) the model should be saved; default 10

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;layers_base: in the model, number of convolution layers to be applied before down-sampling or up-sampling; default 1

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;channels_base: in the model, initial number of channels computed from the input signal; default 16

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;min_spatial_size: in the model, minimum temporal size (in spite of the name), under which down-sampling should not be performed; default 2

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;start_dilation: in the model, initial dilation values in the encoder’s convolutional layers; default 3

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;min_sig_dil_ratio: in the model, minimum ratio between temporal length of the signal at each layer and the corresponding dilation value; which the ratio is smaller, dilation is reduced; default 50

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;max_channels: in the model, channels are doubled at each down-sampling or up-sampling layer, until the maximum number of channels is reached; default 1024

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;h_size: in the model, size of the representation at the bottleneck; default 64

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;enable_variational: choose whether to use AE (False) or VAE (True); default False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;optim: optimizer to use; default Adam

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;reduce_lr_every: defines how often (number of epochs) learning rate should be reduced; default None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;reduce_lr_factor: defines the factor by which the learning rate should be reduced; default 0.1

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;weight_decay: weight for L2 regularization; default 0.0005

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;resume: checkpoint folder to continue previous training

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;epochs: number of total training epochs; default 32000

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;lr: starting learning rate; default 0.00001

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;device: processor to use for training (cpu or cuda); default “cuda”

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;visualizationSetup.txt: to configure visualization parameters. Parameters:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;logs_dir: folder where to find the previously saved training sessions 

&nbsp;&nbsp;&nbsp;&nbsp;●&nbsp;&nbsp;&nbsp;&nbsp;testingSetup.txt: to configure the testing parameters. Parameters:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;checkpoint: model to be validated (it can be a specific checkpoint file or the folder containing all checkpoints; in this case, the best checkpoint based on training loss will be selected)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;test_dir: folder where the dataset files for the testing phase are located

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;data_len: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;chunk_jump: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;channels_list: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;batch_size: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;data_provider: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;device: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;mean: as above

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○&nbsp;&nbsp;&nbsp;&nbsp;std: as above

requirements.txt: can be used to install the modules necessary for the correct functioning of the toolkit by running the following command: “pip install -r requirements.txt”

