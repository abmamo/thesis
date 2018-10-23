# transfrom bassline puzzles into tensors. Each letter is 
# described by a one hot vector. Strings are then a 
# combination of these one hot vectors stacked up. 

# read puzzles from pickle file
import torch
import pickle

all_letters = 'abcdefghijklmnopqrstuvwxyz'
data = pickle.load( open( "puzzles.pkl", "rb" ) )

def letter_to_index(letter):
    '''
       Function that takes a letter and returns its index in our alphabet
    '''
    return all_letters.find(letter)

def letter_to_tensor(letter):
    '''
       Take a letter and convert it into a one hot vector
    '''
    # initialise an empty tensor
    tensor = torch.zeros(1, len(all_letters))
    # use the index of the letter in our alphabet to convert it into a vector
    tensor[0][letter_to_index(letter)] = 1
    return tensor 

def line_to_tensor(line):
    '''
       Function to convert an entire string into an n dimensional array where
       each row is a one hot vector representation fo the ith character
    '''
    # create empty tensor of dimension len(line) x 1 x len(alphabet)
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def vectorize_data(data):
    '''
       Vectorize generated data
    '''
    vectorized_data = []
    for d in data:
        v_d = {"similar": [line_to_tensor(x) for x in d["similar"]], "odd": [line_to_tensor(x) for x in d["odd"]]}
        vectorized_data.append(v_d)
    return vectorized_data

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
