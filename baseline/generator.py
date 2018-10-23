# Randommly generate puzzles using letters from the english alphabet
# We will have 5 strings where 4 strings will not contain a letter
# in some other string. The other string will be out odd man out.
# We then try to see if if a neural network can learn the odd man out
import random
import pickle
import itertools

def get_alphabet():
    '''
       Function to generate an alphabet. In this case the English alphabet
    '''
    alphabet = '0123456789'
    return list(alphabet)

def generate_not_containing(alphabet, length = 5):
    '''
       For every letter in our alphabet generate a list of strings that 
       do not contain that alphabet. Return the result as a dictionary
       with the list of strings as values and the letter they don't contain
       as key.
    '''
    # initialise dictionary
    not_containing = {}
    for i in range(len(alphabet)):
        # get list of letters from alphabet that doesn't include the ith letter
        cands = [x for x in alphabet if x != alphabet[i]]
        # add it to the dictionary
        not_containing[alphabet[i]] = [''.join(x) for x in itertools.combinations(cands, length)]
    return not_containing

def generate_containing(alphabet, not_containing, length = 5):
    '''
       For every letter in our alphabet generate list of strings that contain
       the given letter. Return the result as a dictionary indexed by the
       letter.
    '''
    # initialise dictionary
    containing = {}
    for i in not_containing:
        containing[i] = [''.join(x) for x in itertools.combinations(alphabet, length) if i in x]
    return containing

def generate_puzzles(containing, not_containing):
    '''
       Function to generate simple puzzles using the english alphabet.
       One entry in our list doesn't contain a letter the other strings
       in our list
    '''
    # empty list to store our list of puzzles
    puzzles = []
    # for every letter in our alphabet
    for letter in containing:
        # for every string containing letter
        for string in containing[letter]:
            # get 4 random strings containing string and 1 string not containing letter
            puzzles.append({"odd" : random.sample(not_containing[letter], 1), "similar": random.sample(containing[letter], 4)})
    return puzzles

def save_puzzles(puzzles):
    '''
       Dumps our dictionary to a pickle file
    '''
    # open file
    f = open("puzzles.pkl", "wb")
    # save file
    pickle.dump(puzzles,f)
    f.close()

def run():
    length = 2
    alphabet = get_alphabet()
    not_containing = generate_not_containing(alphabet, length)
    containing = generate_containing(alphabet, not_containing, length)
    puzzles = generate_puzzles(containing, not_containing)
    converted_puzzles = convert_to_int(puzzles)
    print("%d puzzles generated!" % len(converted_puzzles))
    save_puzzles(converted_puzzles)
     
run()

