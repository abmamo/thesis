import torch

torch.manual_seed(1)


def oneHot(word, vocab):
    vec = [0]*len(vocab)
    vec[vocab[word]] = 1
    return vec

def makePuzzleVector(puzzle, vocab):
    (num1, num2, num3), _ = puzzle
    oneHot1 = oneHot(str(num1), vocab)
    oneHot2 = oneHot(str(num2), vocab)
    oneHot3 = oneHot(str(num3), vocab)
    return torch.FloatTensor(oneHot1 + oneHot2 + oneHot3).view(1, -1)


def makePuzzleTarget(label):
    return torch.LongTensor([label])

def buildVocab(puzzles):
    word_to_ix = {}
    for choices, _ in puzzles:
        for word in choices:
            if word not in word_to_ix:
               word_to_ix[word] = len(word_to_ix)
    return word_to_ix
