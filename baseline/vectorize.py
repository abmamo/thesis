import torch

if torch.cuda.is_available():
    print("vectorize using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    def cudaify(model):
        model.cuda()
else:
    print("vectorize using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        pass

torch.manual_seed(1)


def oneHot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def makePuzzleVector(puzzle, vocab):
    choices, _ = puzzle
    result = []
    for choice in choices:
        result = result + oneHot(choice, vocab)
    return FloatTensor(result).view(1, -1)


def makePuzzleTarget(label):
    return torch.LongTensor([label])

def makePuzzleMatrix(puzzles, vocab):
    matrix = []
    for puzzle in puzzles:
        choices, _ = puzzle
        oneHotVec = []
        for choice in choices:
            oneHotVec += oneHot(str(choice), vocab)
        matrix.append(oneHotVec)
    return FloatTensor(matrix, device=cuda)

def makePuzzleTargets(labels):
    return LongTensor(labels)


def buildVocab(puzzles):
    word_to_ix = {}
    for choices, _ in puzzles:
        for word in choices:
            if word not in word_to_ix:
               word_to_ix[word] = len(word_to_ix)
    return word_to_ix


