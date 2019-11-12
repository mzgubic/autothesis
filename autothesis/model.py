import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    small = True
    vocab = generate.get_vocab(token, small=True)

    # build the model
    model = CharacterRNN(token, vocab)

    # create criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for i, (batch, labels) in enumerate(generate.generate('train', token=token, max_len=max_len, small=small)):

        #print()
        #print(batch, labels)

        # one hot encode
        batch = generate.one_hot_encode(batch, vocab)
        #labels = generate.one_hot_encode(labels, vocab)

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




        if i >= 5:
            break

