#where does the softmax get added
# how to input the elements
import torch
import numpy as np

import pickle
# get puzzles from pickle file
data = pickle.load( open( "vectorized_puzzles.pkl", "rb" ) )

N, D_in, H, D_out = 90, 5, 100, 5

# convert to pytorch tensors
inputs = torch.from_numpy(np.array(data["inputs"]))
output = torch.from_numpy(np.array(data["output"]))

print(data['inputs'][0])
print(data['output'][0])
x = inputs
y = output

print(x.size())
print(y.size())

# Use the nn package to define our model and loss function.
# create the layers of the neural network
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    # add softmax here
)

model = model.double()
# assuming this is a multi class classification problem
loss_fn = torch.nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # pass it through a softmax here or add a layer

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
