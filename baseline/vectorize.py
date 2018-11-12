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

def line_to_tensor(line):
    '''
       Function to convert an entire string into an n dimensional array where
       each row is a one hot vector representation fo the ith character
    '''
    # create a 1x10 zero tensor
    tensor = [0] * len(all_digits)
    # Set the ith entry to be 1
    tensor[0][line] = 1
    return tensor

def get_inputs(data):
    ''' 
       GEt a list of lists that is our input
    '''
    inputs, output = [], []
    for d in data:
        inputs.append(d["input"])
        output.append(d["output"])
    return inputs, output

def vectorize_data(data):
    '''
       vectorize inputs
    '''
    
    inputs, output = get_inputs(data)
    v_inputs = torch.Tensor(inputs)
    v_output = torch.Tensor(output)
    return v_inputs, v_output
      

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
    i, o = vectorize_data(data)
    dataset = {"inputs" : i, "output" : o}
    save_puzzles(dataset)

run()
