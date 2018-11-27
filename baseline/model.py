# -*- coding: utf-8 -*-

import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generator import Generator
from vectorize import buildVocab, makePuzzleVector, makePuzzleTarget

if torch.cuda.is_available():
    print("using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    def cudaify(model):
        model.cuda()
else:
    print("using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        pass

class TwoLayerClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, input_size, hidden_size):
        super(TwoLayerClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.linear1(input_vec).clamp(min=0)
        nextout = self.linear2(nextout)
        return F.log_softmax(nextout, dim=1)

class Trainer:

    def __init__(self, train_data, test_data, epochs = 20, dimension = 300):
        self.num_training_epochs = epochs
        self.hidden_layer_size = dimension
        self.num_choices = len(train_data[0][0])
        self.train_data = train_data
        self.test_data = test_data
        self.vocab = buildVocab(self.train_data + self.test_data)

    def train(self):
        model = TwoLayerClassifier(self.num_choices,
                                   self.num_choices * len(self.vocab),
                                   self.hidden_layer_size)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(self.num_training_epochs):
            print('epoch {}'.format(epoch))
            for instance, label in self.train_data:
                # Step 1. Remember that PyTorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Make our input vector and also we must wrap the target in a
                # Tensor as an integer.
                input_vec = makePuzzleVector((instance, label), self.vocab)
                target = makePuzzleTarget(label)

                # Step 3. Run our forward pass.
                log_probs = model(input_vec)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
        return model


    def evaluate(self, model, test_d):
        """Evaluates the trained network on test data."""
        word_to_ix = self.vocab
        with torch.no_grad():
            correct = 0
            for instance, label in test_d:
                input_vec = makePuzzleVector((instance, label), word_to_ix)
                log_probs = model(input_vec)
                probs = [math.exp(log_prob) for log_prob in log_probs.tolist()[0]]
                ranked_probs = list(zip(probs, range(len(probs))))
                response = max(ranked_probs)[1]
                if response == label:
                    correct += 1
        return correct/len(test_d)


