import os
from pathlib import Path

data_path = Path(os.getenv('DATA'))
src_path = Path(os.getenv('SRC'))

def model_dir_name(cls_name, settings):

    relevant_keys = ['hidden_size', 'learning_rate', 'n_steps', 'batch_size', 'max_len']
    folder_name = cls_name + '__' + '__'.join(['{}{}'.format(k, settings[k]) for k in relevant_keys])

    return data_path / 'run' / settings['token'] / folder_name
