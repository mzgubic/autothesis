import numpy as np
import json
import h5py
import utils

def generate(split, token, batch_size=8, max_len=4, small=False):
    """
    Generate samples from the text corpus.

    Arguments:
        split (string):   train, valid, test
        batch_size (int): batch size
        max_len (int):    how long the sequences should be
        token (str):      what to use as tokens (characters or words)
        small (bool):     whether to use the small text corpus (for debug only)

    Yields:
        batch, labels
        batch (np.array)  (batch_size, max_len) shaped array
        labels (np.array) (batch_size, ) shaped array
    """

    # load the numpy array
    data_path = utils.data_path/'tokens'/token

    with h5py.File(data_path/'{}tokens.h5'.format('small_' if small else ''), 'r') as handle:
        seq = np.array(handle[split])

    # sample the subparts
    while True:

        # create arrays
        batch = np.zeros((batch_size, max_len), dtype=int)
        labels = np.zeros(batch_size, dtype=int)
        max_ind = len(seq) - max_len - 1

        starts = np.random.randint(max_ind, size=batch_size)
        for i, ind in enumerate(starts):
            batch[i, :] = seq[ind:ind+max_len]
            labels[i] = seq[ind+max_len]

        yield batch, labels


def get_vocab(token, small=False):
    """
    Get the vocabulary

    Arguments:
        token (string): character or word
        small (bool): whether to use the small vocab or the large one

    Returns:
        token_to_idx (dict): token to index mapping
        idx_to_token (dict): inverse mapping
    """

    # get the file
    data_path = utils.data_path/'tokens'/token
    with open(data_path/'{}dicts.json'.format('small_' if small else ''), 'r') as f:
        json_data = json.load(f)
    
    return json_data['token_to_idx'], json_data['idx_to_token']


def int2str(array, vocab):
    """
    Transform the array of indices in an array of tokens.

    Arguments:
        array (np.array (int)): array of indices
        token (string): character or word

    Returns:
        text (np.array (str)): array of tokens corresponding to indices
    """

    # get the dicts
    _, i2t = vocab

    # convert to string array
    array = np.array(array)
    text = array.astype(str)

    decode = np.vectorize(lambda x: i2t[x])
    text = decode(text)

    return text


def one_hot_encode(array, vocab):
    
    print(vocab[0])
    print(len(vocab[0]))

    # flatten the original array
    in_shape = array.shape
    new_dim = len(vocab[0])
    flat = array.reshape(-1)
    
    # create the encoded flat array
    encoded = np.zeros(shape=(*array.shape, new_dim))
    encoded = encoded.reshape((-1, new_dim))
    encoded[np.arange(flat.size), flat] = 1

    # reshape encoded array to original shape
    encoded = encoded.reshape((*in_shape, new_dim))

    return encoded

def one_hot_decode(array):

    decoded = np.zeros(array.shape[:-1], dtype=int)
    indices = np.nonzero(array)
    decoded[indices[:-1]] = indices[-1]

    return decoded
    

def main():

    max_len = 5
    token = 'character'
    small = True
    vocab = get_vocab(token, small)

    for batch, labels in generate('test', token=token, max_len=5, small=small):
        #print(batch, labels)
        #print(int2str(batch, vocab))
        #print(int2str(labels, vocab))
        #one_hot_encode(labels, vocab)

        one_hot_batch = one_hot_encode(batch, vocab)
        new_batch = one_hot_decode(one_hot_batch)

        one_hot_labels = one_hot_encode(labels, vocab)
        new_labels = one_hot_decode(one_hot_labels)
        
        print()
        break

if __name__ == '__main__':
    main()

