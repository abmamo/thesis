# transfrom bassline puzzles into tensors. Each letter is 
# described by a one hot vector. Strings are then a 
# combination of these one hot vectors stacked up. 

# read puzzles from pickle file
import torch
import pickle
import itertools
# figure out a way to use pytorch tensors
import numpy as np


all_letters = '0123456789'
# generateall two letter combinations of the digits above
all_digits = list(itertools.product(all_letters, repeat = 2))
all_digits = [x[0] + x[1] for x in all_digits]
data = pickle.load( open( "puzzles.pkl", "rb" ) )

def line_to_index(line):
    '''
       Function that takes a letter and returns its index in our alphabet
    '''
    # return the index of the number in our alphabet
    return all_digits.index(line)


def line_to_tensor(line):
    '''
       Function to convert an entire string into an n dimensional array where
       each row is a one hot vector representation fo the ith character
    '''
    # create a 1x10 zero tensor
    tensor = np.zeros((1, len(all_digits)))
    # Set the ith entry to be 1
    tensor[0][line_to_index(line)] = 1
    return tensor

def vectorize_data(data):
    '''
       Vectorize generated data
    '''
    vectorized_data = []
    # for each puzzle generated
    for d in data:
        # convert everything to vectors
        v_d = {"input" : np.array([line_to_tensor(x) for x in d["input"]]), "output" : np.array(d['output'])}
        # append to new puzzle list
        vectorized_data.append(v_d)
    return np.array(vectorized_data)

def split_puzzles(puzzles):
    '''
       Function to turn our list of dictionary puzzles into two numpy arrays
       for input and output
    '''
    # create empty lists to store inputs and outputs separately
    inputs, output = [], []
    for d in puzzles:
        # iterate and append to our lists
        inputs.append(d["input"])
        output.append(d["output"])
    # return a dict of two numpy arrays
    return {"inputs" : np.array(inputs), "output": np.array(output)}
      

def save_puzzles(puzzles):
    '''
       Dumps our dictionary to a pickle file
    '''
    # open file
    f = open("vectorized_puzzles.pkl", "wb")
    # save file
    pickle.dump(puzzles,f)
    f.close()

   

def run():
    v = vectorize_data(data)
    dataset = split_puzzles(v)
    print(dataset["inputs"].shape)
    save_puzzles(dataset)

run()

