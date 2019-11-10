import argparse
import json
import os
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--tokens', default='character', choices=['character', 'word'])
parser.add_argument('--input_txt', default='ZZZ_combined_theses.txt')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
args = parser.parse_args()

if __name__ == '__main__':

    # build the vocabulary
    with open(utils.data_path / 'txts' / args.input_txt, 'r') as handle:
        corpus = handle.read()

    print(corpus[:100])

    # create and fill the output arrays

    # save the vacab and output arrays

