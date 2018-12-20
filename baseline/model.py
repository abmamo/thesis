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
    def __init__(self, vocab, num_labels, hidden_size, num_hidden):
        super(MultiLayerClassifier, self).__init__()
        self.vocab = vocab
        self.num_labels = num_labels
        # Calculate hidden size using num labels and the total num of
        # puzzles
        input_size = num_labels * len(vocab)
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.input_layer = nn.Linear(input_size, hidden_size)
        if num_hidden > 2:
            self.hidden_layers = [nn.Linear(hidden_size, hidden_size) for i in range(num_hidden - 2)]
        else:
            self.hidden_layers = []
        self.output_layer = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.input_layer(input_vec).clamp(min=0)
        if self.num_hidden > 2:
            for hidden_layer in self.hidden_layers:
                nexout = hidden_layer(nextout)
        nextout = self.output_layer(nextout)
        return F.log_softmax(nextout, dim=1)

    def dump(self):
        weights = {}
        for word in self.vocab:
            word_index = self.vocab[word]
            for label in range(self.num_labels):
                weights[(word, label)] = list(self.input_layer.weight[:,label * len(self.vocab) + word_index].data.numpy())
        return weights

    @staticmethod
    def initialize_from_model_and_vocab(model, vocab):
        result = MultiLayerClassifier(vocab, model.num_labels, model.hidden_size, model.num_hidden)
        input_layer_weights = [[0.0] * model.hidden_size for i in range(model.num_labels * len(vocab))]
        for ((word, choice_index), weight_vector) in model.dump().items():
            input_layer_weights[len(vocab) * choice_index + vocab[word]] = weight_vector
        input_layer_weights = torch.t(torch.FloatTensor(input_layer_weights))
        input_layer_weights.requires_grad = True
        result.input_layer.weight = torch.nn.Parameter(input_layer_weights)
        return result



class Trainer:

    def __init__(self, train_data, test_data, epochs = 20, dimension = 300, num_hidden = 2, batch_size = 1000):
        self.num_training_epochs = epochs
        self.hidden_layer_size = dimension
        self.num_hidden = num_hidden
        self.num_choices = len(train_data[0][0])
        self.train_data = train_data
        self.test_data = test_data
        self.vocab = buildVocab(self.train_data + self.test_data)
        self.batch_size = 1000

    def train(self):
        model = MultiLayerClassifier(self.vocab, self.num_choices, self.hidden_layer_size, self.num_hidden)
        print(model)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())
        for epoch in range(self.num_training_epochs):
            print('epoch {}'.format(epoch))
            for instance, label in self.train_data:
                model.zero_grad()
                input_vec = makePuzzleVector((instance, label), self.vocab)
                target = makePuzzleTarget(label)
                log_probs = model(input_vec)
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
            train_acc = self.evaluate(model, self.train_data[:200])
            test_acc = self.evaluate(model, self.test_data)
            print('train: {:.2f}; test: {:.2f}'.format(train_acc, test_acc))
        return model

    def batch_train(self, model):
        print(model)
        loss_function = nn.NLLLoss()
        batch_size = self.batch_size
        optimizer = optim.Adam(model.parameters())
        for epoch in range(self.num_training_epochs):
            model.zero_grad()
            batch = random.sample(self.train_data, batch_size)
            input_matrix = makePuzzleMatrix(batch, model.vocab)
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

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

base = 10
length = 2
choice = 5
epochs = 2000
dimension = 100
training_size = 5000
testing_size = 100
alphabet = ALPHABET[:base]

g = Generator(alphabet, length, choice)
train_data = g.generate_data(training_size)
test_data = g.generate_data(testing_size)
trainer = Trainer(train_data, test_data, epochs, dimension)
model = trainer.train()
#model = trainer.batch_train(1000)
train_acc = trainer.evaluate(model, train_data[:200])
test_acc = trainer.evaluate(model, test_data)

