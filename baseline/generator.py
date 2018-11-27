import random
import pickle
import itertools

class Generator():
    def __init__(self, alphabet, length, choices):
        self.alphabet = list(alphabet)
        self.length = length
        self.choices = choices
        self.puzzles = None

    def get_similar(self):
        similar = {}
        for i in range(len(self.alphabet)):
            # get alphabet letters different than the ith letter
            cands = [x for x in self.alphabet if x != self.alphabet[i]]
            # generate all 2 letter combination strings and index them by the ith element
            similar[self.alphabet[i]] = [''.join(x) for x in itertools.permutations(cands, self.length)]
        return similar

    def get_odd(self, similar):
        odd = {}
        for i in similar:
            # get all the numbers that contain i
            odd[i] = [''.join(x) for x in itertools.permutations(self.alphabet, self.length) if i in x]
        return odd

    def get_puzzles(self, similar, odd, num):
        # this generates 18 * num of puzzles
        puzzles = []
        for i in range(num):
            for key in odd:
                for number in odd[key]:
                    # get string that doesn't contain key
                    random_odd = [x for x in random.sample(similar[key], 1)]
                    # get <length> number of strings that contain key
                    random_sim = [x for x in random.sample(odd[key], self.choices)]
                    # make puzzle
                    puzzle = random_odd + random_sim
                    # make puzzle index for shuffling
                    # the index for the odd one is always at index 0 initially
                    puz_ind = [1] + [0] * self.choices
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
        similar = self.get_similar()
        odd = self.get_odd(similar)
        puzzles = self.get_puzzles(similar, odd, size)
        self.puzzles = puzzles
        return puzzles

    def save(self, filename):
        # open file
        f = open(filename, "wb")
        # save file
        pickle.dump(self.puzzles,f)
        f.close()


g = Generator('0123456789', 2, 3)
train_data = g.generate_data(200)
g.save("train_data.pkl")
test_data = g.generate_data(10)
g.save("test_data.pkl")
