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
        self.vocab_size = len(vocab[0])
        self.input_size = self.vocab_size
        self.hidden_size = 64

        # layers
        self.rnn = torch.nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        self.dense = torch.nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):

        x = self.rnn(x)
        x = F.relu(self.dense(x))

        return x


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

        print()
        print(batch, labels)

        # one hot encode
        batch = generate.one_hot_encode(batch, vocab)
        labels = generate.one_hot_encode(labels, vocab)

        # turn into torch tensors
        batch = torch.Tensor(batch)
        print(batch)

        # zero the gradients
        optimizer.zero_grad()

        # forward and backward pass and optimisation step
        outputs = model(batch)
        print(outputs)



        if i >= 5:
            break

