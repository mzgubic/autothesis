import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import pickle
import argparse
import numpy as np
import generate
import evaluate
import utils
from model import CharacterRNN

parser = argparse.ArgumentParser()
parser.add_argument('--token', default='character', choices=['character', 'word'])
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--small', action='store_true')
parser.add_argument('--n-steps', type=int, default=1000)
parser.add_argument('--force', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':

    # training and sampling
    temperature = 0.5
    how_many = 50
    vocab = generate.get_vocab(args.token, small=args.small)

    # build the model
    model = CharacterRNN(args.token, vocab)

    # directory housekeeping
    model_dir = utils.model_dir_name(type(model).__name__, args.token, args.max_len, args.n_steps)
    if os.path.exists(model_dir) and args.force:
        os.system('rm -r {}/*'.format(model_dir))
    else:
        os.makedirs(model_dir)

    # create criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # create the validation set
    n_valid = 10000
    valid_gen = generate.generate('val', token=args.token, max_len=args.max_len, small=args.small, batch_size=n_valid)
    for valid_batch, valid_labels in valid_gen:
        valid_batch = generate.one_hot_encode(valid_batch, vocab)
        valid_batch, valid_labels = torch.Tensor(valid_batch), torch.Tensor(valid_labels).long()
        break

    # training loop
    every_n = int(args.n_steps/100)
    running_loss = 0
    training_losses = []
    valid_losses = []
    t0 = time.time()
    for i, (batch, labels) in enumerate(generate.generate('train', token=args.token, max_len=args.max_len, small=args.small)):

        # one hot encode
        batch = generate.one_hot_encode(batch, vocab)

        # turn into torch tensors
        batch = torch.Tensor(batch)
        labels = torch.Tensor(labels).long()

        # zero the gradients
        optimizer.zero_grad()

        # forward and backward pass and optimisation step
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # monitor the losses
        running_loss += loss
        if i % every_n == (every_n-1):

            # append the training losses
            training_losses.append(float(running_loss/every_n))
            running_loss = 0

            # compute the valid loss
            valid_outputs = model(valid_batch)
            valid_losses.append(float(criterion(valid_outputs, valid_labels)))

            # monitor progress
            monitor = ['\n{}/{} done'.format(i+1, args.n_steps)]
            monitor.append(model.compose('The Standard Model of pa', temperature, how_many))
            monitor.append(model.compose('[23] Aad G', temperature, how_many))
            monitor.append(model.compose('arxiv.', temperature, how_many))
            for m in monitor:
                print(m)
                with open(model_dir/'out_stream.txt', 'a') as handle:
                    handle.write(m+'\n')

        if i >= args.n_steps:
            break
    
    # save the losses
    time_per_step = (time.time() - t0) / args.n_steps
    print('tiem per step: {}'.format(time_per_step))
    loss_dict = {'train':training_losses, 'valid':valid_losses, 'time_per_step':time_per_step}
    pickle.dump(loss_dict, open(model_dir/ 'losses.pkl', 'wb'))

    # save the settings
    settings = {'token':args.token, 'max_len':args.max_len, 'small':args.small,
                'n_steps':args.n_steps, 'every_n':every_n}
    pickle.dump(settings, open(model_dir/ 'settings.pkl', 'wb'))

    # evaluate
    evaluate.plot_losses(model_dir)

