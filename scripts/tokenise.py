import re
import time
import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.corpus import words
from generate import Vocab
import utils
from julia import Main
Main.include("../autothesis/fast.jl")


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

        if i>=1000000 and args.small:
            break

    return token2idx, token_counts, np.array(idx_list)


@utils.timeit
def phrasify(token2idx, token_counts, idx_array):
    """
    Build phrases based on the score:
            score = (count(ti, tj) - delta) / (count(ti) * count(tj))

    Expand the vocabulary by 1%, for the phrases with the highest score.
    """
    # build the coocurrency matrix
    n_tokens = len(token_counts)
    counts_bi = np.zeros(shape=(n_tokens, n_tokens))
    counts_first = np.zeros(shape=(n_tokens, n_tokens))
    counts_second = np.zeros(shape=(n_tokens, n_tokens))
    
    # loop over the indices to count bigrams
    for k in range(len(idx_array)-1):
        idx_i, idx_j = idx_array[k], idx_array[k+1]
        counts_bi[idx_i, idx_j] += 1
        counts_first[idx_i,:] += 1
        counts_second[:, idx_j] += 1
    # and take care of first and last token
    counts_first[idx_array[-1], :] += 1
    counts_second[:, idx_array[0]] += 1

    # compute the score
    delta = np.max(counts_bi) / 100.
    scores = (counts_bi - delta) / (counts_first * counts_second)
    scores = np.nan_to_num(scores, copy=False, nan=0, posinf=0, neginf=0)

    # increase vocabulary by 1%
    i2t = {token2idx[k]:k for k in token2idx}
    threshold = np.min(sorted(scores.flatten(), reverse=True)[:int(1+0.01*n_tokens)])
    idx_pairs = np.argwhere(scores >= threshold)
    
    # add the phrases to token2idx
    for idx_i, idx_j in idx_pairs:
        phrase = '{} {}'.format(i2t[idx_i], i2t[idx_j])
        token2idx[phrase] = len(token2idx)

    # update the idx_array and the token counts
    new_idx_array = []
    token_counts = {t:0 for t in token2idx}
    itr = iter(range(len(idx_array) - 1))
    for i in itr:

        # current and next index
        this_idx = idx_array[i]
        next_idx = idx_array[i+1]
        phrase = '{} {}'.format(i2t[this_idx], i2t[next_idx])

        # if they are not a phrase
        if phrase not in token2idx:
            token = i2t[this_idx]
            new_idx_array.append(this_idx) # add index to the new list
            token_counts[token] += 1 # and count

        # if they are a phrase
        else:
            phrase = '{} {}'.format(i2t[this_idx], i2t[next_idx])
            new_idx_array.append(token2idx[phrase]) # add to new list
            token_counts[phrase] += 1 # and add a count
            try:
                skip = next(itr) # skip the second part of the phrase
            except StopIteration:
                continue

    return token2idx, token_counts, np.array(new_idx_array)


@utils.timeit
def reduce_tokens(token2idx, token_counts, idx_array, args):

    # TODO: delete these two lines
    token2idx = token2idx.copy()
    token_counts = token_counts.copy()

    show_first(20, token2idx, idx_array)

    new_t2i = {'<unk>':0}
    new_idx_array = np.copy(idx_array)
    for token in sorted(token2idx.keys()):

        # this is the new index
        old_idx = token2idx[token]
        count = token_counts[token]

        # new index: merged with unknown if below threshold
        above_threshold = count >= args.threshold
        new_idx = len(set(new_t2i.values())) if above_threshold else 0

        # change the t2i and idx_array
        new_t2i[token] = new_idx
        there = np.where(idx_array == old_idx)
        new_idx_array[there] = new_idx

        # remove counts
        if not above_threshold:
            if token != '<unk>':
                token_counts[token] = 0
            token_counts['<unk>'] += count

    show_first(20, new_t2i, new_idx_array)

    return new_t2i, token_counts, new_idx_array


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
    test_text = ' '.join([i2t[i] for i in idx_array[:20]])
    print(test_text)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default='character', choices=['character', 'word'])
    parser.add_argument('--input-txt', default='ZZZ_combined_theses.txt')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--threshold', type=int, default=300, help='Minimum frequency of tokens to be kept (otherwise: <unk>)')
    parser.add_argument('--phrase-length', type=int, default=3, help='Max phrase length')
    args = parser.parse_args()

    # some helpers
    fpath = utils.data_path / 'txts' / args.input_txt
    sweep_tokens = sweep_words if args.token == 'word' else sweep_chars

    # first sweep to build the vocabulary
    print('First sweep to build the vocabulary and count entries')
    lines = sweep_lines(fpath)
    tokens = sweep_tokens(lines)
    token2idx, token_counts, idx_array = index(tokens)

    show_first(20, token2idx, idx_array)

    # for words: add phrases and reduce number of tokens 
    if args.token == 'word':

        # throw away tokens which dont appear often enough
        print('Throwing away rare tokens')
        #token2idx, token_counts, idx_array = reduce_tokens(token2idx, token_counts, idx_array, args)
        t0 = time.time()
        p_t2i, p_tc, p_ia = reduce_tokens(token2idx, token_counts, idx_array, args)
        print('python took {}'.format(time.time() - t0))

        t0 = time.time()
        j_t2i, j_tc, j_ia = Main.reduce_tokens(token2idx, token_counts, idx_array, args.threshold)
        print('julia took {}'.format(time.time() - t0))

        for i, token in enumerate(token2idx):
            print()
            print(token)
            print('o', token2idx[token], token_counts[token])
            print('p', p_t2i[token], p_tc[token])
            print('j', j_t2i[token], j_tc[token])
            if i == 5:
                break

        ## identify phrases and merge the individual word tokens into phrases
        #print('length of t2i is {}, idx_array {}'.format(len(token2idx), type(idx_array)))
        #print('Identifying phrases')
        #for _ in range(args.phrase_length - 1):
        #    token2idx, token_counts, idx_array = phrasify(token2idx, token_counts, idx_array)
        #    print('length of t2i is {}, idx_array {}'.format(len(token2idx), type(idx_array)))

    # second sweep to fill the output arrays
    print('Split the tokens into training, validation, and test sets')
    train, val, test = split_token_array(idx_array, args)

    # save the array
    save_tokens(train, val, test, token2idx, args)

    for length in range(args.phrase_length - 1):
        print()
        phrases = [p for p in token2idx if p.count(' ') > length]
        for p in phrases:
            print(p)

    print(test[:20])
    print(val[:20])
    print(train[:20])


