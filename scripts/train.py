import argparse
import numpy as np
import matplotlib.pyplot as plt

import generate
from model import CharacterRNN

parser = argparse.ArgumentParser()
parser.add_argument('--token', default='character', choices=['character', 'word'])
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--small', action='store_true')
parser.add_argument('--n-steps', type=int, default=1000)
args = parser.parse_args()

if __name__ == '__main__':

    # training and sampling
    temperature = 0.5
    how_many = 50

    vocab = generate.get_vocab(args.token, small=small)

    # build the model
    model = CharacterRNN(args.token, vocab)

    # create criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    every_n = int(total_n/100)
    running_loss = 0
    losses = []
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
            print('{}/{} done'.format(i+1, total_n))
            losses.append(running_loss/every_n)
            running_loss = 0
            print(model.compose('The Standard Model of pa', temperature, how_many))

        if i >= args.n_steps:
            break
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(losses))*every_n, losses, label='training loss')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.savefig('Losses.pdf')


