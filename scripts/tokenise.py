import argparse
import json
import os
import h5py
import numpy as np
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--tokens', default='character', choices=['character', 'word'])
parser.add_argument('--input_txt', default='ZZZ_combined_theses.txt')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
args = parser.parse_args()

if __name__ == '__main__':

    # build the vocabulary
    token_to_idx = {}
    total_size = 0
    with open(utils.data_path / 'txts' / args.input_txt, 'r') as handle:
        for i, line in enumerate(handle):
            print(line)
            total_size += len(line)
            for char in line:
                if char not in token_to_idx:
                    token_to_idx[char] = len(token_to_idx)+1
            if i > 10:
                break

    print(token_to_idx)
    print(total_size)

    # create and fill the output arrays
    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size

    val = np.zeros(val_size, int)
    test = np.zeros(test_size, int)
    train = np.zeros(train_size, int)
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
            if i > 10:
                break

    print(train)
    print(val)
    print(test)

    # save the vocab and output arrays
    out_path = utils.data_path/'tokens'/args.tokens
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with h5py.File(out_path/'tokens.h5', 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('val', data=val)
        f.create_dataset('test', data=test)

    idx_to_token = {token_to_idx[k]:k for k in token_to_idx}
    print(idx_to_token)
    json_data = {'token_to_idx':token_to_idx, 'idx_to_token':idx_to_token}

    with open(out_path/'dicts.json', 'w') as f:
        json.dump(json_data, f)
    

