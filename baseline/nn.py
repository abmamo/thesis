import torch.nn as nn

class RNN(nn.Module):
      # inherit from the Module class
      def __init__(self, input_size, hidden_size, output_size):
          # class initialisation
          super(RNN, self).__init__()
      
          # set hidden layer size
          self.hidden_size = hidden_size
          

          # what are these lines?
          self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
          self.i2o = nn.Linear(input_size + hidden_size, output_size)
          self.softmax = nn.LogSoftmax(dim=1)
     # forward pass
     def forward(self, input, hidden):
         # what do these lines do
         combined = torch.cat((input, hidden), 1)
         hidden = self.i2h(combined)
         output = self.i2o(combined)
         output = self.softmax(output)
         return output, hidden

     def init_hidden(self):
         # return zero matrix
         return torch.zeros(1, self.hidden_size)


criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()
