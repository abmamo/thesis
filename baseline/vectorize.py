# transfrom bassline puzzles into tensors. Each letter is 
# described by a one hot vector. Strings are then a 
# combination of these one hot vectors stacked up. 

# read puzzles from pickle file
import torch
import pickle
# figure out a way to use pytorch tensors
import numpy as np


all_letters = '0123456789'
data = pickle.load( open( "puzzles.pkl", "rb" ) )

def letter_to_index(letter):
    '''
       Function that takes a letter and returns its index in our alphabet
    '''
    # return the index of the letter in our alphabet
    return all_letters.find(letter)


def line_to_tensor(line):
    '''
       Function to convert an entire string into an n dimensional array where
       each row is a one hot vector representation fo the ith character
    '''
    # create a 1x10 zero tensor
    tensor = np.zeros((1, len(all_letters)))
    for letter in line:
        # for the ith and jth indices, where i and j are the digits of our
        # 2 digit number set the value to 1
        tensor[0][letter_to_index(letter)] = 1
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
    save_puzzles(v)

run()

