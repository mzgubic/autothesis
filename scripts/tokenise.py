import re
import argparse
import json
import os
import h5py
import numpy as np
import pickle
import utils
import nltk
from nltk.corpus import words
from generate import Vocab


def sweep_lines(fpath):
    with open(fpath, 'r') as handle:
        for line in handle:
            yield line[:-1]


def sweep_words(lines):
    """
    Yield the word tokens
    """

    # separate into tokens based on whitespace
    tokens = (t for line in lines for t in line.split())

    # and separate tokens into alpha, numeric, and other characters
    tokens = (subtoken.lower() for t in tokens for subtoken in re.split('(\W)', t) if subtoken != '')

    for token in tokens:
        yield token


def sweep_chars(lines):
    """
    Yield the character tokens
    """
    for line in lines:
        for char in line:
            yield char


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default='character', choices=['character', 'word'])
    parser.add_argument('--input_txt', default='ZZZ_combined_theses.txt')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()

    # some helpers
    fpath = utils.data_path / 'txts' / args.input_txt
    sweep_tokens = sweep_words if args.token == 'word' else sweep_chars

    # first sweep to build the vocabulary
    print('First sweep to build the vocabulary')
    lines = sweep_lines(fpath)
    tokens = sweep_tokens(lines)

    token_to_idx = {'<unk>':0}
    total_size = 0
    for i, token in enumerate(tokens):

        # add to vocab if not there already
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

        total_size += 1
        if i>=1000 and args.small:
            break

    # second sweep to fill the output arrays
    print('Second sweep to fill the output arrays')
    lines = sweep_lines(fpath)
    tokens = sweep_tokens(lines)

    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size

    dtype = int if len(token_to_idx) > 255 else np.uint8
    val = np.zeros(val_size, dtype)
    test = np.zeros(test_size, dtype)
    train = np.zeros(train_size, dtype)
    splits = [train, val, test]

    split_idx, current_idx = 0, 0
    for i, token in enumerate(tokens):

        # add the token
        splits[split_idx][current_idx] = token_to_idx[token]
        current_idx+=1

        # go to next split
        if current_idx == len(splits[split_idx]):
            split_idx+=1
            current_idx=0

        if i>=1000 and args.small:
            break


    # save the vocab and output arrays
    out_path = utils.data_path/'tokens'/args.token
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

    print(train)
    print(token_to_idx)
    print(total_size)



