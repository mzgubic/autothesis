import numpy as np
import pickle
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

    with h5py.File(data_path/'{}_tokens.h5'.format('small' if small else 'full'), 'r') as handle:
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


class Vocab():
    
    def __init__(self, token_to_idx):

        self.t2i = token_to_idx
        self.i2t = {token_to_idx[k]:k for k in token_to_idx}
        self.size = len(token_to_idx)

    def __getitem__(self, key):
        
        # token to index
        if type(key) == str:
            try:
                return self.t2i[key]
            except KeyError:
                return self.t2i['<unk>']

        # or index to token
        elif type(key) == int:
            return self.i2t[key]

        # but nothing else
        else:
            raise TypeError('Vocabulary can be indexed either by tokens or indices, not by {}'.format(key))

    def __repr__(self):
        
        return 'Vocab({})'.format(self.t2i)



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
    with open(data_path/'{}_vocab.pkl'.format('small' if small else 'full'), 'rb') as f:
        vocab = pickle.load(f)
    
    return vocab


def int2str(array, vocab):
    """
    Transform the array of indices in an array of tokens.

    Arguments:
        array (np.array (int)): array of indices
        token (string): character or word

    Returns:
        text (np.array (str)): array of tokens corresponding to indices
    """

    # convert to string array
    array = np.array(array)
    text = array.astype(str)

    decode = np.vectorize(lambda x: vocab[int(x)])
    text = decode(text)

    return text


def one_hot_encode(array, vocab):

    # flatten the original array
    in_shape = array.shape
    new_dim = vocab.size
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

    max_len = 10
    token = 'character'
    small = True
    vocab = get_vocab(token, small)

    for batch, labels in generate('train', token=token, max_len=max_len, small=small):
        print(batch)
        print(labels)
        print(int2str(batch, vocab))
        print(int2str(labels, vocab))
        one_hot_encode(labels, vocab)

        one_hot_batch = one_hot_encode(batch, vocab)
        new_batch = one_hot_decode(one_hot_batch)

        one_hot_labels = one_hot_encode(labels, vocab)
        new_labels = one_hot_decode(one_hot_labels)
        print(new_batch)
        print(new_labels)
        
        print()
        break

if __name__ == '__main__':
    main()

