# -*- coding: utf-8 -*-
# import all dependencies
import math
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# use cuda if available
if torch.cuda.is_available():
    print("neural network using gpu")
    print("------------------------")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    def cudaify(model):
        model.cuda()
else:
    print("neural network using cpu")
    print("------------------------")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    def cudaify(model):
        pass

# A multi layer feed forwardd neural network


class MultiLayerClassifier(nn.Module):
    def __init__(self, vocab, num_labels, hidden_size, num_hidden):
        # Initialise class
        super(MultiLayerClassifier, self).__init__()
        # Set vocabulary
        self.vocab = vocab
        # Set the number of labels / number of choices
        self.num_labels = num_labels
        # Calculate hidden size using num labels and the total num of
        # puzzles
        input_size = num_labels * len(vocab)
        # Set the number of neurons in our hidden layers
        self.hidden_size = hidden_size
        # Set the number of hidden layers
        self.num_hidden = num_hidden
        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # Define hidden layers accordingly
        if num_hidden > 2:
            self.hidden_layers = [
                nn.Linear(hidden_size, hidden_size) for i in range(num_hidden - 2)]
        else:
            self.hidden_layers = []
        # Define output layer
        self.output_layer = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        # input the vector into the input layer
        nextout = self.input_layer(input_vec).clamp(min=0)
        # pass the result of the previous layer to a hidden layer
        if self.num_hidden > 2:
            for hidden_layer in self.hidden_layers:
                nextout = hidden_layer(nextout)
        # output layer
        nextout = self.output_layer(nextout)
        # pass output layer outcome through a softmax
        return F.log_softmax(nextout, dim=1)

    def dump(self):
        # defien empty dictionary to store weights in
        weights = {}
        # iterate through each word in the vocabulary
        for word in self.vocab:
            # get the index of each word
            word_index = self.vocab[word]
            for label in range(self.num_labels):
                # get the value of the weights in the outer layer
                weights[(word, label)] = list(
                    self.input_layer.weight[:, label * len(self.vocab) + word_index].data)
        # return weights
        return weights

    @staticmethod
    def initialize_from_model_and_vocab(model, vocab):
        # Initialise new network with new vocabulary and the values from the old model
        result = MultiLayerClassifier(
            vocab, model.num_labels, model.hidden_size, model.num_hidden)
        # Set the weights to be zero initially
        input_layer_weights = [
            [0.0] * model.hidden_size for i in range(model.num_labels * len(vocab))]
        # Update the weights using the weights from the old model
        for ((word, choice_index), weight_vector) in model.dump().items():
            # update the weights with the weight vector from the previous model
            input_layer_weights[len(vocab) * choice_index +
                                vocab[word]] = weight_vector
        # transpose the weights
        input_layer_weights = torch.t(FloatTensor(input_layer_weights))
        # make computing gradients easier
        input_layer_weights.requires_grad = True
        # update the layer of the model
        result.input_layer.weight = torch.nn.Parameter(input_layer_weights)
        # return the resulting model
        return result
