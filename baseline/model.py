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
        # list to store training data accuracy at each step
        self.train_acc = []
        # list to store the epoch values at each step
        self.epoch_step = []
        # list to store testing data accruacy at each epoch
        self.test_acc = []

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

    def batch_train(self, batch_size):
        model = TwoLayerClassifier(self.num_choices,
                                   self.num_choices * len(self.vocab),
                                   self.hidden_layer_size)
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

    def plot(self):
        '''
            Function to save the accuracy vs epoch graph.
        '''
        plt.plot(self.epoch_step, self.train_acc, label="training accuracy")
        plt.plot(self.epoch_step, self.test_acc, label="testing_accuracy")
        plt.title("Accuracy vs Epochs Graph with Training Data Size: " + str(len(self.train_data)))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


