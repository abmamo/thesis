
import copy
import random
import pickle
import itertools
from copy import deepcopy

class Generator():
    def __init__(self, alphabet, length, choices):
        self.alphabet = alphabet
        self.length = length
        self.choices = choices
        self.puzzles = None

    def generate_puzzle(self):
        # randomly select a letter all similar strings must contain
        rand_letter = random.sample(self.alphabet, 1)[0]
        # define a list to store the puzzle
        puzzle = []
        # copy the alphabet since we are going to avoid replacement
        alphabet = self.alphabet
        for i in range(self.choices - 1):
            # generate 4 random strings to be appended to the random letter above
            remaining = ''.join(random.sample(alphabet, self.length - 1))
            # create word from random letter and the remaining string
            word = rand_letter + remaining
            # shuffle the word to avoid bias
            word = ''.join(random.sample(word, len(word)))
            # append word to puzzle list
            puzzle.append(word)
            # remove the letters in the word above to avoid redundancy
            for letter in word:
                # remove the letter from the alphabet
                alphabet = alphabet.replace(letter, '')
        # remove the common letter from the alphabet
        alphabet = self.alphabet
        alphabet = alphabet.replace(rand_letter, '')
        # randomly generate a string using the above alphabet
        odd = ''.join(random.sample(alphabet, self.length))
        puzzle.append(odd)

        # create puzzle index which shows where the odd word is
        puz_ind = [0] * (self.choices - 1) + [1]
        # combine the puzzle and its index for shuffling
        combined = list(zip(puzzle, puz_ind))
        # shuffle
        random.shuffle(combined)
        # get the puzzle and index back
        puzzle[:], puz_ind[:] = zip(*combined)
        # return puzzle,index tuple
        return (puzzle, puz_ind.index(1))

    def generate_data(self, size):
        # create list to store puzzle, index tuple pairs
        puzzles = []
        # until the desired training data size is reached randomly generate a puzzle
        for i in range(size):
            puzzles.append(self.generate_puzzle())
        return puzzles
