import os
from pathlib import Path

if os.getenv('USER') == 'zgubic' and 'pposx' in os.uname().nodename: # local
    SRC="/Users/zgubic/Projects/autothesis"
    DATA="/Users/zgubic/Projects/autothesis"
else: # pplxint and condor
    SRC="/home/zgubic/thesis/autothesis"
    DATA="/data/atlassmallfiles/users/zgubic/thesis"

data_path = Path(DATA)
src_path = Path(SRC)

def model_dir_name(settings, use_epochs):

    relevant_keys = ['cell', 'hidden_size', 'learning_rate', 'batch_size', 'max_len', 'n_cores']
    relevant_keys += ['n_epochs' if use_epochs else 'n_steps']
    folder_name = '__'.join(['{}{}'.format(k, settings[k]) for k in relevant_keys])

    return data_path / 'run' / settings['token'] / folder_name
