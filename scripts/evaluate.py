import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pickle
import generate
from model import CharacterModel
from pathlib import Path

def plot_losses(loc):

    # load data
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))

    # settings 
    cell = settings['cell']
    hidden_size = settings['hidden_size']
    every_n = settings['every_n']
    token = settings['token']
    small = settings['small']
    max_len = settings['max_len']
    n_epochs = settings['n_epochs']
    n_steps = settings['n_steps']
    criterion = nn.CrossEntropyLoss()

    # load the models
    models = []
    vocab = generate.get_vocab(token, small)
    for fname in os.listdir(model_dir/'checkpoints'):
        model = CharacterModel(cell, hidden_size, vocab)
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
        t0 = time.time()
        print(i)
        for split in splits:
            # loss
            outputs = model(batch[split])
            l = criterion(outputs, labels[split])
            loss[split].append(float(l))
            # accuracy
            _, preds = torch.max(outputs, 1)
            a = sum(preds==labels[split]) / float(N)
            acc[split].append(float(a))
        print('{:2.2f}s'.format(time.time()-t0))

    for split in splits:
        with open(model_dir/'best_{}_acc.txt'.format(split), 'w') as handle:
            best = max(acc[split])
            handle.write('{}\n'.format(best))

    # plot both quantities
    for quantity, description in zip([loss, acc], ['Loss', 'Accuracy']):
        fig, ax = plt.subplots()
        for split in splits:
            xs = (1+np.arange(len(quantity[split])))*every_n
            if n_epochs > 1:
                xs = xs / n_steps
            ax.plot(xs, quantity[split], label=split)
        ax.set_xlabel('Training step')
        if n_epochs > 1:
            ax.set_xlabel('Epoch')
        ax.set_ylabel(description)
        upper = ax.get_ylim()[1] if description == 'Loss' else 1
        ax.set_ylim(0, upper)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_title(model_dir.name, fontsize=8)
        ax.legend()
        ax.grid(alpha=0.5, which='both')
        plt.savefig(model_dir/'{}.pdf'.format(description))

def freestyle(loc):

    # load data
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))

    # settings 
    cell = settings['cell']
    hidden_size = settings['hidden_size']
    token = settings['token']
    small = settings['small']
    n_steps = settings['n_steps']
    every_n = settings['every_n']
    how_many = 100
    temperature = 0.5

    # load the models
    vocab = generate.get_vocab(token, small)
    for i, fname in enumerate(os.listdir(model_dir/'checkpoints')):

        # load the model
        model = CharacterModel(cell, hidden_size, vocab)
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
            with open(model_dir/'evaluate_stream.txt', 'a') as handle:
                handle.write(m+'\n')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    default = '/data/atlassmallfiles/users/zgubic/thesis/run/character/cellRNN__hidden_size64__learning_rate0.001__batch_size64__max_len20__n_epochs2'
    parser.add_argument('--input-dir', default=default)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        freestyle(args.input_dir)
    plot_losses(args.input_dir)
