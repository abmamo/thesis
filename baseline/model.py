# -*- coding: utf-8 -*-

import math
import random
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generator import Generator
from vectorize import buildVocab, makePuzzleVector, makePuzzleTarget, makePuzzleTargets, makePuzzleMatrix

if torch.cuda.is_available():
    print("trainer using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    def cudaify(model):
        model.cuda()
else:
    print("trainer using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        pass

class MultiLayerClassifier(nn.Module):
    def __init__(self, num_labels, input_size, hidden_size, num_hidden):
        # initialise classifier
        super(MultiLayerClassifier, self).__init__()
        # define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # define hidden layers
        if num_hidden > 2:
           self.hidden_layers = [nn.Linear(hidden_size, hidden_size) for i in range(num_hidden -2)]
        else:
           self.hidden_layers = []
        # define output layer
        self.output_layer = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        # pass it to the inptu layer
        nextout = self.input_layer(input_vec).clamp(min=0)
        if len(self.hidden_layers) > 2:
            # pass it throught the hidden layers
            for hidden_layer in self.hidden_layers:
                nextout = hidden_layer(nextout)
        # pass it through the output layer
        nextout = self.output_layer(nextout)
        # return the softmax
        return F.log_softmax(nextout, dim=1)

class Trainer:

    def __init__(self, train_data, test_data, epochs = 20, dimension = 300, num_hidden = 2):
        self.num_training_epochs = epochs
        self.hidden_layer_size = dimension
        self.num_hidden = num_hidden
        self.num_choices = len(train_data[0][0])
        self.train_data = train_data
        self.test_data = test_data
        self.vocab = buildVocab(self.train_data + self.test_data)
        # list to store training data accuracy at each step
        self.train_acc = []
        # list to store the epoch values at each step
        self.epoch_step = []
        # list to store testing data accruacy at each epoch
        self.test_acc = []

    def train(self):
        model = MultiLayerClassifier(self.num_choices, self.num_choices * len(self.vocab), self.hidden_layer_size, self.num_hidden)
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

    def batch_train(self, batch_size):
        model = MultiLayerClassifier(self.num_choices, self.num_choices * len(self.vocab), self.hidden_layer_size, self.num_hidden)
        cudaify(model)
        print(model)
        loss_function = nn.NLLLoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = optim.Adam(model.parameters())
        for epoch in range(self.num_training_epochs):
            model.zero_grad()
            batch = random.sample(self.train_data, batch_size)
            input_matrix = makePuzzleMatrix(batch, self.vocab)
            target = makePuzzleTargets([label for (_, label) in batch])
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch {}'.format(epoch))
                train_acc = self.evaluate(model, self.train_data[:200])
                test_acc = self.evaluate(model, self.test_data)
                print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
                self.train_acc.append(train_acc)
                self.test_acc.append(test_acc)
                self.epoch_step.append(epoch)
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


