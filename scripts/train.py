import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import pickle
import argparse
import htcondor
import numpy as np
import generate
import evaluate
import utils
from model import CharacterModel

parser = argparse.ArgumentParser()
parser.add_argument('--token', default='character', choices=['character', 'word'])
parser.add_argument('--cell', default='RNN', choices=['RNN', 'GRU', 'LSTM'])
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--hidden-size', type=int, default=64)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--small', action='store_true')
parser.add_argument('--force', action='store_true')
parser.add_argument('--condor', action='store_true')
parser.add_argument('--n-cores', type=int, default=1)
parser.add_argument('--n-saves', type=int, default=20)
parser.add_argument('--n-epochs', type=int, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


def train(settings, model_dir):

    # training and sampling
    temperature = 0.5
    how_many = 70
    vocab = generate.get_vocab(args.token, small=args.small)

    # build the model
    model = CharacterModel(args.cell, args.hidden_size, vocab)

    # create criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # create the validation set
    n_valid = 10000
    valid_gen = generate.generate('valid', token=args.token, max_len=args.max_len, small=args.small, batch_size=n_valid)
    for valid_batch, valid_labels in valid_gen:
        valid_batch = generate.one_hot_encode(valid_batch, vocab)
        valid_batch, valid_labels = torch.Tensor(valid_batch), torch.Tensor(valid_labels).long()
        break

    # how many epochs do we need?
    batches_per_epoch = generate.get_n_batches_in_epoch('train', args.token, args.batch_size, args.max_len, args.small)

    # training settings
    every_n = int(batches_per_epoch/args.n_saves) if not args.debug else 50
    running_loss = 0
    training_losses = []
    valid_losses = []
    t0 = time.time()
 
    # dump the settings
    pickle.dump(settings, open(model_dir/ 'settings.pkl', 'wb'))
    out_stream = model_dir / 'out_stream.txt'

    # run the training loop
    for epoch in range(1, args.n_epochs+1):

        opening = ['', '#'*20, '# Epoch {} (t={:2.2f}h)'.format(epoch, (time.time() - t0)/3600.), '#'*20, '']
        for txt in opening:
            utils.report(txt, out_stream)

        # create the generator for each epoch
        train_gen = generate.generate('train', token=args.token, max_len=args.max_len,
                                      small=args.small, batch_size=args.batch_size)
        for i, (batch, labels) in enumerate(train_gen):

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
                monitor = ['\n{}/{} done'.format(i+1, batches_per_epoch)]
                monitor.append(model.compose('The Standard Model of', temperature, how_many))
                for m in monitor:
                    utils.report(m, out_stream)
                
                # save the model
                torch.save(model.state_dict(), model_dir/'checkpoints'/'epoch{}_step_{}.pt'.format(epoch, round(i/every_n)))

            if i >= 1000 and args.debug:
                break
    
    # save information
    dt = (time.time() - t0)
    time_txt = '\ntime taken: {:2.2f}h\n'.format(dt/3600.)
    utils.report(time_txt, out_stream)
    utils.report(str(dt/3600.), model_dir/'time.txt')
        
    loss_dict = {'train':training_losses, 'valid':valid_losses, 'time_taken':dt}
    pickle.dump(loss_dict, open(model_dir/ 'losses.pkl', 'wb'))

    # evaluate
    evaluate.plot_losses(model_dir)


def write_job(model_dir):

    options = sys.argv
    options.remove('--condor')
    options.remove('--force')

    commands = [
                '#!/bin/sh',
                'cd {}'.format(utils.SRC),
                'source {}/setup.sh'.format(utils.SRC),
                'cd scripts',
                'python {}'.format(' '.join(options))
                ]
    print(commands[-1])

    script = model_dir / '{}.sh'.format(model_dir.name)
    with open(script, 'w') as handle:
        for c in commands:
            handle.write(c+'\n')

def send_job(model_dir):

    # create the submit object
    d = {'executable':model_dir / '{}.sh'.format(model_dir.name),
         'arguments':'$(ClusterID)',
         'output':'{}/$(ClusterId).out'.format(model_dir),
         'error':'{}/$(ClusterId).err'.format(model_dir),
         'log':'{}/$(ClusterId).log'.format(model_dir),
         'request_cpus':args.n_cores,
         'request_memory':'40 GB',
         'getenv':True,
         'stream_output':True,
         'stream_error':True,
         }

    sub = htcondor.Submit(d)

    # create the scheduler object
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        sub.queue(txn)
        print('submitted')


if __name__ == '__main__':

    # settings
    settings = {a:getattr(args, a) for a in dir(args) if a[0] != '_'}

    # model dir
    model_dir = utils.model_dir_name(settings)
    if args.force:
        os.system('rm -r {}'.format(model_dir))
    if not model_dir.exists():
        os.makedirs(model_dir/'checkpoints')

    # send to codor
    if args.condor:
        write_job(model_dir)
        send_job(model_dir)

    # or train interactively
    else:
        train(settings, model_dir)


