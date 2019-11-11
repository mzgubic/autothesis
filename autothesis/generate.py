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


def int2str(array, token):
    """
    Transform the array of indices in an array of tokens.

    Arguments:
        array (np.array (int)): array of indices
        token (string): character or word

    Returns:
        text (np.array (str)): array of tokens corresponding to indices
    """

    # get the vocabulary
    _, i2t = get_vocab(token)

    # convert to string array
    array = np.array(array)
    text = array.astype(str)

    decode = np.vectorize(lambda x: i2t[x])
    text = decode(text)

    return text
    

def main():

    token = 'character'

    for batch, labels in generate('test', token=token, max_len=10, small=True):
        print(batch, labels)
        #print(int2str(batch, token))
        #print(int2str(labels, token))
        #print()
        break

if __name__ == '__main__':
    main()

