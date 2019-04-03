# -*- coding: utf-8 -*-

import math
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.nn import MultiLayerClassifier
from model.generator import Generator
from model.vectorize import buildVocab, makePuzzleVector, makePuzzleTarget, makePuzzleTargets, makePuzzleMatrix

if torch.cuda.is_available():
    print("trainer using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        model.cuda()
else:
    print("trainer using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        pass


class Trainer:

    def __init__(self, train_data, test_data, epochs = 20, dimension = 300, old_model=None, num_hidden = 2, batch_size = 1000):
        self.num_training_epochs = epochs
        self.hidden_layer_size = dimension
        self.num_hidden = num_hidden
        self.num_choices = len(train_data[0][0])
        self.train_data = train_data
        self.test_data = test_data
        self.vocab = buildVocab(self.train_data + self.test_data)
        self.batch_size = batch_size
        if old_model == None:
            self.model = MultiLayerClassifier(self.vocab, self.num_choices, self.hidden_layer_size, self.num_hidden)
        else:
            new_model = MultiLayerClassifier(self.vocab, self.num_choices, self.hidden_layer_size, self.num_hidden)
            self.model = new_model.initialize_from_model_and_vocab(old_model, new_model.vocab)

    def train(self):
        print(self.model)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(self.num_training_epochs):
            print('epoch {}'.format(epoch))
            for instance, label in self.train_data:
                self.model.zero_grad()
                input_vec = makePuzzleVector((instance, label), self.vocab)
                target = makePuzzleTarget(label)
                log_probs = self.model(input_vec)
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
            train_acc = self.evaluate(self.model, self.train_data[:200])
            test_acc = self.evaluate(self.model, self.test_data)
            print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
        return self.model

    def batch_train(self):
        print(self.model)
        loss_function = nn.NLLLoss()
        batch_size = self.batch_size
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(self.num_training_epochs):
            self.model.zero_grad()
            batch = random.sample(self.train_data, batch_size)
            input_matrix = makePuzzleMatrix(batch, self.model.vocab)
            target = makePuzzleTargets([label for (_, label) in batch])
            log_probs = self.model(input_matrix)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch {}'.format(epoch))
                train_acc = self.evaluate(self.model, self.train_data[:200])
                test_acc = self.evaluate(self.model, self.test_data)
                print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
        return self.model


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

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


base = 10
length = 2
choice = 5
epochs = 2000
dimension = 100
training_size = 50000
testing_size = 100
alphabet = ALPHABET[:base]

g = Generator(alphabet, length, choice)
train_data = g.generate_data(training_size)
test_data = g.generate_data(testing_size)
trainer = Trainer(train_data, test_data, epochs, dimension)
model = trainer.batch_train()
train_acc = trainer.evaluate(model, train_data[:200])
test_acc = trainer.evaluate(model, test_data)

# Doesn't work when you increase the lengths of words
# because how initialize from old model works maybe change that to account
# for increase in the length of words in our puzzle


# Doesn't work when you increase the number of choices 
# either becaue there is a size mismatch in in batch train

base = 10
length = 3
choice = 5
epochs = 2000
dimension = 100
training_size = 50000
testing_size = 100
alphabet = ALPHABET[:base]

g = Generator(alphabet, length, choice)
train_data = g.generate_data(training_size)
test_data = g.generate_data(testing_size)
trainer = Trainer(train_data, test_data, epochs, dimension, model)
new_model = trainer.batch_train()
train_acc = trainer.evaluate(new_model, train_data[:1000])
test_acc = trainer.evaluate(new_model, test_data)

base = 10
length = 3
choice = 5
epochs = 2000
dimension = 100
training_size = 50000
testing_size = 100
alphabet = ALPHABET[:base]

g = Generator(alphabet, length, choice)
train_data = g.generate_data(training_size)
test_data = g.generate_data(testing_size)
trainer = Trainer(train_data, test_data, epochs, dimension)
model = trainer.batch_train()
train_acc = trainer.evaluate(model, train_data[:200])
test_acc = trainer.evaluate(model, test_data)


