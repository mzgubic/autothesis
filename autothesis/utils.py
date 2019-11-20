import os
from pathlib import Path

data_path = Path(os.getenv('DATA'))
src_path = Path(os.getenv('SRC'))

def model_dir_name(settings, use_epochs):

    relevant_keys = ['cell', 'hidden_size', 'learning_rate', 'batch_size', 'max_len']
    relevant_keys += ['n_epochs' if use_epochs else 'n_steps']
    folder_name = '__'.join(['{}{}'.format(k, settings[k]) for k in relevant_keys])

    return data_path / 'run' / settings['token'] / folder_name
