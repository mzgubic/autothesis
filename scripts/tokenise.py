import re
import time
import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd
import pickle
from generate import Vocab
import utils


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


@utils.timeit
def index(tokens):
    """
    Creates a mapping of tokens to indices, counts the tokens,
    and creates a list of indices representing the tokens.
    """
    token2idx = {'<unk>':0}
    token_counts = {t:0 for t in token2idx}
    idx_list = []
    for i, token in enumerate(tokens):

        # add to vocab if not there already
        if token in token2idx:
            token_counts[token] += 1
        else:
            token2idx[token] = len(token2idx)
            token_counts[token] = 1

        # add to index array
        idx_list.append(token2idx[token])

        if i>=100000 and args.small:
            break

    return token2idx, token_counts, np.array(idx_list)


@utils.timeit
def reduce_tokens(token2idx, token_counts, idx_array, args):

    # identify keeper tokens
    common_tokens = ['<unk>'] + [token for token in token_counts if token_counts[token] > args.threshold]

    # prepare the outputs
    t2i = {'<unk>':0}
    tc = {t:0 for t in common_tokens}
    array = np.zeros(len(idx_array))

    # create the mapping from old to new indices
    mapping = {}
    for token in common_tokens:
        if token == '<unk>':
            continue
        t2i[token] = len(t2i)
        mapping[token2idx[token]] = t2i[token]

    # loop over idx_array and apply the map
    i2t = {token2idx[t]:t for t in token2idx}
    for i, idx in enumerate(idx_array):

        # change token to <unk> if not common
        token = i2t[idx]
        token = token if token in common_tokens else '<unk>'

        # update the array
        array[i] = t2i[token] 

        # update the counts
        tc[token] += 1
    
    return t2i, tc, array


@utils.timeit
def phrasify(token2idx, token_counts, idx_array):
    """
    Build phrases based on the score:
            score = (count(ti, tj) - delta) / (count(ti) * count(tj))

    Expand the vocabulary by 1% using the phrases with the highest score.
    """

    # count phrase occurence
    bigram_counts = {t:{} for t in token2idx}
    i2t = {token2idx[t]:t for t in token2idx}
    n_tokens = len(idx_array)
    for i in range(n_tokens-1):
        this_token = i2t[idx_array[i]]
        next_token = i2t[idx_array[i+1]]
        try:
            bigram_counts[this_token][next_token] += 1
        except KeyError:
            bigram_counts[this_token][next_token] = 1

    # compute the scores
    counts = [bigram_counts[first][second] for first in bigram_counts for second in bigram_counts[first]]
    delta = np.max(counts) / 100.
    bigram_scores = {ft:{st:0 for st in bigram_counts[ft]} for ft in bigram_counts}
    for first in bigram_counts:
        for second in bigram_counts[first]:
            score = (bigram_counts[first][second] - delta) / (token_counts[first] * token_counts[second])**0.5
            bigram_scores[first][second] = score

    # keep good phrases (above threshold ones)
    # (increase vocabulary by 1%)
    scores = [bigram_scores[first][second] for first in bigram_scores for second in bigram_scores[first]]
    threshold = np.min(sorted(scores, reverse=True)[:int(1+0.01*n_tokens)])
    good_phrases = {(first, second) for first in bigram_scores for second in bigram_scores[first] if bigram_scores[first][second] > threshold}

    # add good phrases to token2idx
    for first, second in good_phrases:
        phrase = '{} {}'.format(first, second)
        token2idx[phrase] = len(token2idx)

    # update idx_array and token_counts
    itr = iter(range(n_tokens - 1))
    new_idx_list = []
    token_counts = {token:0 for token in token2idx}
    for i in itr:

        # is the token combination a phrase?
        this_token = i2t[idx_array[i]]
        next_token = i2t[idx_array[i+1]]
        phrase = '{} {}'.format(this_token, next_token)

        # if yes, record the phrase and skip the second token of the phrase
        if phrase in token2idx:
            token = phrase
            # skip the second token of the phrase
            try:
                _ = next(itr)
            except StopIteration:
                pass
        # if not, just use the token
        else:
            token = this_token
        
        # update idx array
        new_idx_list.append(token2idx[token])
        token_counts[token] += 1

    return token2idx, token_counts, np.array(new_idx_list)


def split_token_array(idx_array, args):

    # determine the number of elements
    n_tokens = len(idx_array)
    val_size = int(args.val_frac * n_tokens)
    test_size = int(args.test_frac * n_tokens)

    test = idx_array[:test_size]
    val = idx_array[test_size:test_size+val_size]
    train = idx_array[test_size+val_size:]

    return train, val, test

def save_tokens(train, val, test, token2idx, args):

    # save the vocab and output arrays
    out_path = utils.data_path / 'tokens' / args.token
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # numpy arrays
    with h5py.File(out_path / '{}_tokens.h5'.format('small' if args.small else 'full'), 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('valid', data=val)
        f.create_dataset('test', data=test)

    # easy to read vocab
    json_data = token2idx
    with open(out_path/'{}_vocab.json'.format('small' if args.small else 'full'), 'w') as f:
        json.dump(json_data, f)

    # pickle vocab
    vocab = Vocab(token2idx)
    with open(out_path/'{}_vocab.pkl'.format('small' if args.small else 'full'), 'wb') as f:
        pickle.dump(vocab, f)

def show_first(n, token2idx, idx_array):
    i2t = {token2idx[k]:k for k in token2idx}
    i2t[0] = '<unk>'
    test_text = ' '.join([i2t[i] for i in idx_array[:n]])
    print(test_text)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default='character', choices=['character', 'word'])
    parser.add_argument('--input-txt', default='ZZZ_combined_theses.txt')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--threshold', type=int, default=300, help='Minimum frequency of tokens to be kept (otherwise: <unk>)')
    parser.add_argument('--phrase-runs', type=int, default=2, help='Max phrase length')
    args = parser.parse_args()

    # some helpers
    fpath = utils.data_path / 'txts' / args.input_txt
    sweep_tokens = sweep_words if args.token == 'word' else sweep_chars

    # first sweep to build the vocabulary
    print('First sweep to build the vocabulary and count entries')
    lines = sweep_lines(fpath)
    tokens = sweep_tokens(lines)
    token2idx, token_counts, idx_array = index(tokens)

    # for words: add phrases
    if args.token == 'word':

        # identify phrases and merge the individual word tokens into phrases
        print('Identifying phrases')
        for _ in range(args.phrase_runs):
            token2idx, token_counts, idx_array = phrasify(token2idx, token_counts, idx_array)
        
        # remove rare words
        show_first(40, token2idx, idx_array)
        token2idx, token_counts, idx_array = reduce_tokens(token2idx, token_counts, idx_array, args)
        show_first(40, token2idx, idx_array)

    # split arrays
    print('Split the tokens into training, validation, and test sets')
    train, val, test = split_token_array(idx_array, args)

    # save the array
    save_tokens(train, val, test, token2idx, args)

    for length in range(args.phrase_runs):
        print()
        phrases = [p for p in token2idx if p.count(' ') > length]
        for p in phrases:
            print(p)


