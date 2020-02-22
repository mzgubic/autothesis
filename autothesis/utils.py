import time
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

def model_dir_name(settings):

    relevant_keys = ['debug', 'cell', 'hidden_size', 'learning_rate', 'batch_size', 'max_len', 'n_cores', 'n_epochs']
    folder_name = '__'.join(['{}{}'.format(k, settings[k]) for k in relevant_keys])

    return data_path / 'run' / settings['token'] / folder_name

def report(txt, fpath):

    print(txt)
    with open(fpath, 'a') as handle:
        handle.write(txt + '\n')

    
def timeit(f):
    def decorated(*args, **kwargs):
        t0 = time.time()
        r = f(*args, **kwargs)
        print(' --> {} took {:2.2f}s'.format(f.__name__, time.time() - t0))
        return r
    return decorated
