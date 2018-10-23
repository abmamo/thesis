import pickle 
# get puzzles from pickle file
data = pickle.load( open( "vectorized_puzzles.pkl", "rb" ) )

import torch

n_letters = 10
n_hidden = 10
n_categories = 5

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

test_input = torch.cat((data[0]['similar'][0], data[0]['similar'][1], data[0]['similar'][2], data[0]['similar'][3], data[0]['odd'][0]), 0)
print(test_input)
#output, next_hidden = rnn(test_input, hidden)
#print(output)
