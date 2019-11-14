import os
from pathlib import Path

data_path = Path(os.getenv('DATA'))
src_path = Path(os.getenv('SRC'))

def model_dir_name(cls_name, token, max_len, n_steps):

    return data_path / 'run' / token / '{}_{}_{}steps'.format(cls_name, max_len, n_steps)
