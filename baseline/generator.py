import random
import pickle
import itertools

class Generator():
    def __init__(self, alphabet, length):
        self.alphabet = alphabet
        self.length = length
        self.puzzles = None

    def alphabet_to_list(self, alphabet):
        return list(alphabet)

    def get_similar(self, alphabet, length):
        similar = {}
        for i in range(len(alphabet)):
            # get alphabet letters different than the ith letter
            cands = [x for x in alphabet if x != alphabet[i]]
            # generate all 2 letter combination strings and index them by the ith element
            similar[alphabet[i]] = [''.join(x) for x in itertools.permutations(cands, length)]
        return similar

    def get_odd(self, alphabet, similar, length):
        odd = {}
        for i in similar:
            # get all the numbers that contain i
            odd[i] = [''.join(x) for x in itertools.permutations(alphabet, length) if i in x]
        return odd

    def get_puzzles(self, similar, odd, length, num):
        # this generates 18 * num of puzzles
        puzzles = []
        for i in range(num):
            for key in odd:
                for number in odd[key]:
                    # get string that doesn't contain key
                    random_odd = [x for x in random.sample(similar[key], 1)]
                    # get <length> number of strings that contain key
                    random_sim = [x for x in random.sample(odd[key], length)]
                    # make puzzle
                    puzzle = random_odd + random_sim
                    # make puzzle index for shuffling
                    # the index for the odd one is always at index 0 initially
                    puz_ind = [1] + [0] * length
                    # combine the puzzle and its index for shuffling
                    combined = list(zip(puzzle, puz_ind))
                    # shuffle
                    random.shuffle(combined)
                    # get puzzle and index back
                    puzzle[:], puz_ind[:] = zip(*combined)
                    # append the puzzle and index of the odd string as a tuple to
                    # the puzzle list
                    puzzles.append((puzzle, puz_ind.index(1)))
        return puzzles

    def generate_data(self, size):
        alphabet = self.alphabet_to_list(self.alphabet)
        similar = self.get_similar(alphabet, self.length)
        odd = self.get_odd(alphabet, similar, self.length)
        puzzles = self.get_puzzles(similar, odd, self.length, size)
        self.puzzles = puzzles
        return puzzles

    def save(self, filename):
        # open file
        f = open(filename, "wb")
        # save file
        pickle.dump(self.puzzles,f)
        f.close()


g = Generator('0123456789', 2)
train_data = g.generate_data(200)
g.save("train_data.pkl")
test_data = g.generate_data(10)
g.save("test_data.pkl")
