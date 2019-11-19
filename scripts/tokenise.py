import argparse
import json
import os
import h5py
import numpy as np
import pickle
import utils
from generate import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--tokens', default='character', choices=['character', 'word'])
parser.add_argument('--input_txt', default='ZZZ_combined_theses.txt')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--small', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':

    # build the vocabulary
    token_to_idx = {'<unk>':0}
    total_size = 0
    with open(utils.data_path / 'txts' / args.input_txt, 'r') as handle:
        for i, line in enumerate(handle):
            total_size += len(line)
            for char in line:
                if char not in token_to_idx:
                    token_to_idx[char] = len(token_to_idx)
            if i>=5 and args.small:
                break

    # create and fill the output arrays
    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size

    dtype = int if len(token_to_idx) > 255 else np.uint8
    val = np.zeros(val_size, dtype)
    test = np.zeros(test_size, dtype)
    train = np.zeros(train_size, dtype)
    splits = [train, val, test]

    split_idx, current_idx = 0, 0
    with open(utils.data_path / 'txts' / args.input_txt, 'r') as handle:
        for i, line in enumerate(handle):
            for char in line:
                splits[split_idx][current_idx] = token_to_idx[char]
                current_idx+=1
                if current_idx == len(splits[split_idx]):
                    split_idx+=1
                    current_idx=0
            if i>=5 and args.small:
                break

    # save the vocab and output arrays
    out_path = utils.data_path/'tokens'/args.tokens
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # numpy arrays
    with h5py.File(out_path/'{}_tokens.h5'.format('small' if args.small else 'full'), 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('valid', data=val)
        f.create_dataset('test', data=test)

    # easy to read vocab
    json_data = token_to_idx
    with open(out_path/'{}_vocab.json'.format('small' if args.small else 'full'), 'w') as f:
        json.dump(json_data, f)

    # pickle vocab
    vocab = Vocab(token_to_idx)
    with open(out_path/'{}_vocab.pkl'.format('small' if args.small else 'full'), 'wb') as f:
        pickle.dump(vocab, f)

    print(token_to_idx)
    print(total_size)

    

