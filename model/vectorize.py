# import dependencies
import torch

# use cuda if available
if torch.cuda.is_available():
    print("vectorize using gpu")
    print("-------------------")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    def cudaify(model):
        model.cuda()
else:
    print("vectorize using cpu")
    cuda = torch.device('cpu')
    print("-------------------")
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    def cudaify(model):
        pass

# seed the random generation to replicate experiment
torch.manual_seed(1)


def oneHot(word, vocab):
    '''
       Function to turn a word into a one hot vector
       given the word and the vocabulary
    '''
    # initialise a zero vector of the expected size
    vec = [0]*len(vocab)
    # change the specific index
    vec[vocab[word]] = 1
    # return vector
    return vec


def makePuzzleVector(puzzle, vocab):
    '''
       Function to make a puzzle matrix which is just
       a vector form of all the words in a puzzlee
    '''
    # get the words from the puzzle
    choices, _ = puzzle
    # define empty list to store vectors
    result = []
    # iterate through each word and vectorize it
    # and append it to the matrix
    for choice in choices:
        result = result + oneHot(choice, vocab)
    # reshape the matrix and return it
    return FloatTensor(result).view(1, -1)


def makePuzzleTarget(label):
    '''
       function given a list of lables converts them
       to pytorch tensors
    '''
    return LongTensor([label])


def makePuzzleMatrix(puzzles, vocab):
    '''
       function to convert a collection of puzzles
       into a collection of puzzle matrices which
       are just a vector form of all the words in 
       a puzzle.
    '''
    # define empty list to store matrix values
    matrix = []
    # iterate through each puzzle
    for puzzle in puzzles:
        # get words from puzzle
        choices, _ = puzzle
        # vector to store inidivdual puzzle
        oneHotVec = []
        # go through each word and vectorize it
        for choice in choices:
            oneHotVec += oneHot(str(choice), vocab)
        # add the puzzle to the matrix
        matrix.append(oneHotVec)
    # return tensor matrix
    return FloatTensor(matrix, device=cuda)


def makePuzzleTargets(labels):
    return LongTensor(labels)


def buildVocab(puzzles):
    '''
       function to build a vocabulary from a given set of puzzles.
    '''
    # define empty dictionary to store words and their index
    word_to_ix = {}
    # iterate through each words in each puzzle
    for choices, _ in puzzles:
        for word in choices:
            # if word is not already in the dictionary add it
            # with an increasing index
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # return vocabulary
    return word_to_ix
