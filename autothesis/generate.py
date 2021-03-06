import re
import numpy as np
import pickle
import h5py
import itertools
import word2vec
import torch
import torch.nn.functional as F
import utils


def sweep_lines(fpath):
    with open(fpath, 'r') as handle:
        for line in handle:
            yield line[:-1]


def yield_words(lines):
    """
    Yield the word tokens
    """

    # separate into tokens based on whitespace
    tokens = (t for line in lines for t in line.split())

    # and separate tokens into alpha, numeric, and other characters
    tokens = (subtoken.lower() for t in tokens for subtoken in re.split('(\W)', t) if subtoken != '')

    for token in tokens:
        yield token


def yield_chars(lines):
    """
    Yield the character tokens
    """
    for line in lines:
        for char in line:
            yield char


def get_n_batches_in_epoch(split, token, batch_size, max_len, small):

    # load the numpy array
    data_path = utils.data_path/'tokens'/token

    with h5py.File(data_path/'{}_tokens.h5'.format('small' if small else 'full'), 'r') as handle:
        seq = np.array(handle[split])

    # exhaust iterator after one epoch: when the number of yielded characters is the total number of characters
    per_batch = batch_size * max_len
    total = len(seq)
    n_batches = int(total/per_batch)

    return n_batches


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

    # exhaust iterator after one epoch: when the number of yielded characters is the total number of characters
    per_batch = batch_size * max_len
    total = len(seq)
    n_batches = int(total/per_batch)

    if split == 'train':
        print('Generator will yield {} batches before exhausting'.format(n_batches))

    # sample the subparts
    for _ in range(n_batches):

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
        self.i2t[0] = '<unk>'
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

    def int2str(self, array):
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
    
        decode = np.vectorize(lambda x: self[int(x)])
        text = decode(text)
    
        return text


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


def str2batch(intxt, vocab, emb):
    """
    Turn a string of characters to a batch ready to be processed.

    Arguments:
        intxt (string): string to be translated

    Returns:
        batch (torch.Tensor): (1, len(intxt), vocab_size)
    """

    # token == character
    if emb == None:
        inds = np.array([vocab[char] for char in intxt])
        inds = one_hot_encode(inds, vocab)
        batch = torch.unsqueeze(torch.Tensor(inds), dim=0)
    # token == word
    else:
        tokens = [t for t in yield_words([intxt])]
        batch = np.zeros((1, len(tokens), emb.vectors.shape[1]))
        for i in range(len(tokens)):
            batch[0, i, :] = emb.get_vector(tokens[i])
        batch = torch.Tensor(batch)

    return batch

def compose(model, vocab, emb, txt, temperature, how_many):
    """
    Continue the paragraph given starting text.

    Arguments:
        txt (string):      string to be continued by the model
        temperature (float): "temperature" which adds uncertainty to sampling
        how_many (int):      how many characters to add

    Returns:
        txt (string):        continued string
    """
        
    # predict new characters
    for i in range(how_many):

        # output of the network
        batch = str2batch(txt, vocab, emb)
        output = model(batch)

        # construct the distribution
        distribution = F.softmax(output/temperature, dim=1).detach().numpy().flatten()

        # and sample from it
        # token == 'character'
        if emb == None:
            sample = np.random.choice(np.arange(vocab.size), p=distribution)
            new = vocab[int(sample)]
            txt = txt+new
        # token == 'word'
        else:
            sample = np.random.choice(np.arange(emb.vectors.shape[0]), p=distribution)
            new = vocab[int(sample)]

            # try to resample to get rid of <unk> predictions, otherwise use "the"
            n_attempts = 0
            while new == '<unk>' and n_attempts < 5:
                sample = np.random.choice(np.arange(emb.vectors.shape[0]), p=distribution)
                new = vocab[int(sample)]
                n_attempts += 1
            if new == '<unk>':
                new = 'the'
                
            txt = txt+' '+new
    
    return txt

def get_embedding(algorithm):

    fpath = utils.data_path / 'embeddings' / algorithm / 'w2v_model.bin'
    model = word2vec.load(str(fpath))
    return model


def w2v_encode(array, emb, vocab):

    # create out array
    batch_size = array.shape[0]
    seq_len = array.shape[1]
    emb_size = emb.vectors.shape[1]
    out_array = np.zeros([batch_size, seq_len, emb_size])

    # text array as input
    txt_array = vocab.int2str(array)

    # loop over the array and get vector for each token
    for i, j in itertools.product(range(batch_size), range(seq_len)):
        token = txt_array[i,j]
        out_array[i,j,:] = emb.get_vector(token)

    return out_array


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

    max_len = 20
    token = 'word'
    small = False
    vocab = get_vocab(token, small)

    for batch, labels in generate('train', token=token, max_len=max_len, small=small):
        print(batch)
        print(batch.shape)
        #print(labels)
        print(vocab.int2str(batch))
        #one_hot_batch = one_hot_encode(batch, vocab)
        #new_batch = one_hot_decode(one_hot_batch)
        #one_hot_labels = one_hot_encode(labels, vocab)
        #new_labels = one_hot_decode(one_hot_labels)
        #print(new_batch)
        #print(new_labels)

        emb = get_embedding('word2vec')
        print()
        print('encoding')
        batch = w2v_encode(batch, emb, vocab)
        print(batch.shape)
        
        print('new batch')
        break

if __name__ == '__main__':
    main()

