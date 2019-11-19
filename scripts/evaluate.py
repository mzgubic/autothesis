import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pickle
import generate
from model import CharacterRNN
from pathlib import Path

def plot_losses(loc):

    # load data
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))

    # settings 
    every_n = settings['every_n']
    token = settings['token']
    small = settings['small']
    max_len = settings['max_len']
    criterion = nn.CrossEntropyLoss()

    # load the models
    models = []
    for fname in os.listdir(model_dir/'checkpoints'):
        vocab = generate.get_vocab(token, settings['small'])
        model = CharacterRNN(64, vocab)
        model.load_state_dict(torch.load(model_dir/'checkpoints'/fname))
        model.eval()
        models.append(model)

    # prepare training and validation sets
    N = 10000
    splits = ['train', 'valid']
    gens = {split:generate.generate(split, token=token, max_len=max_len, small=small, batch_size=N) for split in splits}
    batch, labels = {}, {}
    for split in splits:
        for b, l in gens[split]:
            b = generate.one_hot_encode(b, vocab)
            batch[split], labels[split] = torch.Tensor(b), torch.Tensor(l).long()
            break

    # evaluate the models
    loss = {split:[] for split in splits}
    acc = {split:[] for split in splits}
    for i, model in enumerate(models):
        print(i)
        for split in splits:
            # loss
            outputs = model(batch[split])
            l = criterion(outputs, labels[split])
            loss[split].append(l)
            # accuracy
            _, preds = torch.max(outputs, 1)
            acc[split].append(sum(preds==labels[split]) / float(N))

    # plot both quantities
    for quantity, description in zip([loss, acc], ['Loss', 'Accuracy']):
        fig, ax = plt.subplots()
        for split in splits:
            ax.plot((1+np.arange(len(quantity[split])))*every_n, quantity[split], label=split)
        ax.set_xlabel('Training step')
        ax.set_ylabel(description)
        upper = ax.get_ylim()[1] if description == 'Loss' else 1
        ax.set_ylim(0, upper)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.legend()
        ax.grid(alpha=0.5, which='both')
        plt.savefig(model_dir/'{}.pdf'.format(description))

def freestyle(loc):

    # load data
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))

    # settings 
    token = settings['token']
    n_steps = settings['n_steps']
    every_n = settings['every_n']
    how_many = 100
    temperature = 0.5

    # load the models
    vocab = generate.get_vocab(token, settings['small'])
    for i, fname in enumerate(os.listdir(model_dir/'checkpoints')):

        # load the model
        model = CharacterRNN(64, vocab)
        model.load_state_dict(torch.load(model_dir/'checkpoints'/fname))
        model.eval()

        # monitor progress
        monitor = ['\n{}/{} '.format((i+1)*every_n, n_steps)]
        monitor.append(model.compose('The Standard Mo', temperature, how_many))
        monitor.append(model.compose('[23] ATLAS Co', temperature, how_many))
        monitor.append(model.compose('[15] S. Wein', temperature, how_many))
        monitor.append(model.compose('s = ', temperature, how_many))
        for m in monitor:
            print(m)
            #with open(model_dir/'out_stream.txt', 'a') as handle:
            #    handle.write(m+'\n')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='/data/atlassmallfiles/users/zgubic/thesis/run/character/CharacterRNN_20_100steps')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        freestyle(args.input_dir)
    plot_losses(args.input_dir)
