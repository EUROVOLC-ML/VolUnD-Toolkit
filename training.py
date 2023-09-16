from utils.dataset import Dataset
from utils.trainer import Trainer
from utils.parser import training_parse


if __name__ == '__main__':
    # Get params
    args = training_parse()

    # Normalization
    normalize_params = {"mean": args['mean'], "std": args['std']}

    # Create dataset
    train_dataset = Dataset(args['train_dir'],
                            data_location=args['data_location'],
                            data_key=args['data_key'],
                            chunk_len=args['chunk_len'],
                            chunk_only_one=args['chunk_only_one'],
                            chunk_rate=args['chunk_rate'],
                            chunk_random_crop=args['chunk_random_crop'],
                            data_sampling_frequency=args['data_sampling_frequency'],
                            chunk_linear_subsample=args['chunk_linear_subsample'],
                            chunk_butterworth_lowpass=args['chunk_butterworth_lowpass'],
                            chunk_butterworth_highpass=args['chunk_butterworth_highpass'],
                            chunk_butterworth_order=args['chunk_butterworth_order'],
                            normalize_params=normalize_params,
                            channels_list=args['channels_list'],
                            channels_name=args['channels_name'],
                            provider=args['data_provider'],
                            labels=args['training_labels'])
    val_dataset = Dataset(args['val_dir'],
                          data_location=args['data_location'],
                          data_key=args['data_key'],
                          chunk_len=args['chunk_len'],
                          chunk_only_one=args['chunk_only_one'],
                          chunk_rate=args['chunk_rate'],
                          chunk_random_crop=args['chunk_random_crop'],
                          data_sampling_frequency=args['data_sampling_frequency'],
                          chunk_linear_subsample=args['chunk_linear_subsample'],
                          chunk_butterworth_lowpass=args['chunk_butterworth_lowpass'],
                          chunk_butterworth_highpass=args['chunk_butterworth_highpass'],
                          chunk_butterworth_order=args['chunk_butterworth_order'],
                          normalize_params=normalize_params,
                          channels_list=args['channels_list'],
                          channels_name=args['channels_name'],
                          provider=args['data_provider'],
                          labels=args['validation_labels'])

    # Save number of channels
    args['data_channels'] = len(train_dataset.channels_list)
    args['channels_list'] = train_dataset.channels_list

    # Setup dataset dictionary
    args['datasets'] = {'trainingSet': train_dataset,
                        'validationSet': val_dataset}

    # Define trainer
    trainer = Trainer(args)

    # Run training
    model, metrics = trainer.train()
