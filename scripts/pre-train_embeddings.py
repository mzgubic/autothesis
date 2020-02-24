import h5py
import os
import pickle
import argparse
import word2vec
import numpy as np
import utils
import generate

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', default='word2vec', choices=['word2vec', 'GloVe'])
parser.add_argument('--hidden-size', type=int, default=64)
parser.add_argument('--negative', type=int, default=5)
parser.add_argument('--skip-gram', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--force', action='store_true')
args = parser.parse_args()


def generate_input_file(settings):

    # word2vec input file location
    loc = utils.data_path / 'embeddings' / settings['algorithm']
    fpath = loc / 'input_file.txt'
    if not os.path.exists(loc):
        os.makedirs(loc)

    # only compute of not there or force recompute
    if not os.path.exists(fpath) or args.force:
        
        # load the token2idx and idx_arrays
        vocab = generate.get_vocab('word', small=args.debug)
        array_path = utils.data_path / 'tokens' / 'word' / '{}_tokens.h5'.format('small' if args.debug else 'full')
        with h5py.File(array_path, 'r') as handle:
            seq = np.array(handle['train'], dtype=int)

        # convert to string array
        s_arr = vocab.int2str(seq)
        s = ' '.join(s_arr)

        # and save
        with open(fpath, 'w') as handle:
            handle.write(s)

    return fpath

if __name__ == '__main__':

    # settings
    settings = {a:getattr(args, a) for a in dir(args) if a[0] != '_'}
    print(settings)

    # generate the input file for the model
    ifile = generate_input_file(settings)
    ofile = utils.data_path / 'embeddings' / settings['algorithm'] / 'w2v_model.bin'

    # train the word2vec model
    word2vec.word2vec(ifile, ofile,
                      size=args.hidden_size,
                      negative=args.negative,
                      cbow=not args.skip_gram,
                      min_count=1)

    # examine the model
    model = word2vec.load(str(ofile))




