import torch

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
    return torch.FloatTensor(result).view(1, -1)


def makePuzzleTarget(label):
    return torch.LongTensor([label])

def buildVocab(puzzles):
    word_to_ix = {}
    for choices, _ in puzzles:
        for word in choices:
            if word not in word_to_ix:
               word_to_ix[word] = len(word_to_ix)
    return word_to_ix
