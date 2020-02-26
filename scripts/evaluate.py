import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pickle
import generate
from model import LanguageModel
import utils
from pathlib import Path


def distributions(model, base_batch, repl_batch, keep_depth, vocab):

    # keep the last keep_depth characters from batch, replace the previous ones
    new_batch = np.array(base_batch)
    new_batch[:, :-keep_depth] = repl_batch[:, :-keep_depth]

    if keep_depth == 0:
        new_batch = repl_batch

    # predictions
    t_base = torch.Tensor(generate.one_hot_encode(base_batch, vocab))
    base_distr = F.softmax(model(t_base), dim=1).detach()
    t_new = torch.Tensor(generate.one_hot_encode(new_batch, vocab))
    new_distr = F.softmax(model(t_new), dim=1).detach()
    
    return base_distr, new_distr


def compute_switch_prob(model, base_batch, repl_batch, keep_depth, vocab):

    base_distr, new_distr = distributions(model, base_batch, repl_batch, keep_depth, vocab)
    base_argmax = np.argmax(base_distr.numpy(), axis=1)
    new_argmax = np.argmax(new_distr.numpy(), axis=1)

    n_total = len(base_argmax)
    n_same = sum(base_argmax == new_argmax)
    switch_prob = (n_total - n_same) / n_total

    return switch_prob


def plot_switch_prob(loc):

    # load settings
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))
    cell = settings['cell']
    hidden_size = settings['hidden_size']
    token = settings['token']
    small = settings['small']
    max_len = settings['max_len']

    # load the final model
    vocab = generate.get_vocab(token, small)
    fnames = os.listdir(model_dir/'checkpoints')
    fname = fnames[-1]

    # load the model
    model = CharacterModel(cell, hidden_size, vocab)
    model.load_state_dict(torch.load(model_dir/'checkpoints'/fname))
    model.eval()

    # prepare the base and replacement batch
    N = 100
    gen = generate.generate('valid', token=token, max_len=max_len, small=small, batch_size=N)
    base_batch, _ = next(gen)
    repl_batch, _ = next(gen)
   
    # compute the average KL divs over the batch
    depths = [i for i in range(max_len)]
    switch_probs = [compute_switch_prob(model, base_batch, repl_batch, keep_depth, vocab) for keep_depth in depths]

    # make the plot
    fig, ax = plt.subplots()
    ax.plot(depths, switch_probs, 'tomato')
    ax.plot(depths, [0.01]*len(depths), 'k')
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1)
    ax.set_xlim(0, max_len)
    ax.set_title('Probability of switching predicted character\n{}'.format(model_dir.name), fontsize=7)
    ax.set_xlabel('sequence keep-depth')
    ax.set_ylabel('Probabillity')
    ax.grid()
    plt.savefig(model_dir/'SwitchProbability.pdf')
    

def plot_losses(loc):

    # load data
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))

    # settings 
    cell = settings['cell']
    hidden_size = settings['hidden_size']
    token = settings['token']
    small = settings['small']
    max_len = settings['max_len']
    n_epochs = settings['n_epochs']
    n_saves = settings['n_saves']
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
            xs = (1+np.arange(len(quantity[split]))) / n_saves
            ax.plot(xs, quantity[split], label=split)
        ax.set_xlabel('Training epoch')
        if n_epochs > 1:
            ax.set_xlabel('Epoch')
        ax.set_ylabel(description)
        upper = ax.get_ylim()[1] if description == 'Loss' else 1
        ax.set_ylim(0, upper)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_title(model_dir.name, fontsize=7)
        ax.legend()
        ax.grid(alpha=0.5, which='both')
        plt.savefig(model_dir/'{}.pdf'.format(description))

def freestyle(loc): # TODO

    # load data
    model_dir = Path(loc)
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))
    print(settings)

    # settings 
    cell = settings['cell']
    hidden_size = settings['hidden_size']
    token = settings['token']
    small = settings['small']
    how_many = 100

    # load the models
    vocab = generate.get_vocab(token, small)
    fnames = os.listdir(model_dir/'checkpoints')
    fname = fnames[-1]

    # load the model
    model = LanguageModel(cell, hidden_size, vocab)
    model.load_state_dict(torch.load(model_dir/'checkpoints'/fname))
    model.eval()

    # monitor 
    sents = ['The Standard Mo', 'non-abelia', 'silicon pixel det', 'estimate the t', '[23] ATLAS Co']
    temperatures = [0.01 + 0.1*i for i in range(11)]
    eval_stream =  model_dir/'evaluate_stream.txt'

    for temperature in temperatures:
        txt = '\nTemperature = {}'.format(temperature)
        utils.report(txt, eval_stream)
        for sent in sents:
            txt = model.compose(sent, temperature, how_many)
            utils.report(txt, eval_stream)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    default = '/data/atlassmallfiles/users/zgubic/thesis/run/character/debugTrue__cellRNN__hidden_size64__learning_rate0.001__batch_size64__max_len20__n_cores1__n_epochs1'
    parser.add_argument('--input-dir', default=default)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    print(args)

    plot_switch_prob(args.input_dir)
    if args.verbose:
        freestyle(args.input_dir)
    plot_losses(args.input_dir)
