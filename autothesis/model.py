import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import utils
import generate

class CharacterRNN(nn.Module):

    def __init__(self, token, vocab):

        super(CharacterRNN, self).__init__()
        self.token = token
        self.vocab = vocab

        # embedding parameters
        self.input_size = self.vocab.size
        self.hidden_size = 64

        # layers
        self.rnn = torch.nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        self.dense = torch.nn.Linear(self.hidden_size, self.vocab.size)

    def forward(self, x):

        rnn_output, h = self.rnn(x)
        output = torch.squeeze(self.dense(h), dim=0)

        return output


if __name__ == '__main__':

    # settings
    token = 'character'
    max_len = 5
    small = False
    total_n = 1000
    vocab = generate.get_vocab(token, small=small)

    # build the model
    model = CharacterRNN(token, vocab)

    # create criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    every_n = int(total_n/50)
    running_loss = 0
    losses = []
    for i, (batch, labels) in enumerate(generate.generate('train', token=token, max_len=max_len, small=small)):

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

        if i >= total_n:
            break
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(losses))*every_n, losses, label='training loss')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.savefig('losses.pdf')


