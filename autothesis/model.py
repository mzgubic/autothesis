import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import utils
import generate


class LanguageModel(nn.Module):

    def __init__(self, cell, input_size, hidden_size, output_size):

        super(LanguageModel, self).__init__()

        # parameters
        self.cell = cell
        self.input_size = input_size
        self.hidden_size = hidden_size

        # layers
        tclass = getattr(torch.nn, self.cell)
        self.rnn = tclass(self.input_size, self.hidden_size, batch_first=True)
        self.dense = torch.nn.Linear(self.hidden_size, output_size)

    def forward(self, x):

        # apply rnn
        rnn_output, state = self.rnn(x)

        # unpack
        if self.cell in ['RNN', 'GRU']:
            h = state
        elif self.cell == 'LSTM':
            h, c = state

        # create output
        output = torch.squeeze(self.dense(h), dim=0)

        return output


if __name__ == '__main__':

    # settings
    #token = 'word'
    token = 'character'
    max_len = 20
    hidden_size = 16
    small = False

    # training and sampling
    total_n = 10000
    temperature = 0.5
    how_many = 50

    # create the vocab, model, (and embedding)
    vocab = generate.get_vocab(token, small=small)
    if token == 'word':
        emb = generate.get_embedding('word2vec')
        input_size = emb.vectors.shape[1]
        output_size = emb.vectors.shape[0]
    elif token == 'character':
        emb = None
        input_size = vocab.size
        output_size = vocab.size

    model = LanguageModel('RNN', input_size, hidden_size, output_size)

    # create criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    every_n = int(total_n/100)
    running_loss = 0
    losses = []
    for i, (batch, labels) in enumerate(generate.generate('train', token=token, max_len=max_len, small=small)):

        # one hot encode
        if token == 'character':
            batch = generate.one_hot_encode(batch, vocab)
        # or embed
        elif token == 'word':
            batch = generate.w2v_encode(batch, emb, vocab)

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
            print(generate.compose(model, vocab, emb, 'The Standard Model of ', temperature, how_many))

        if i >= total_n:
            break
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(losses))*every_n, losses, label='training loss')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.savefig('losses.pdf')


