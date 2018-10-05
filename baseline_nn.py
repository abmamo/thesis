# Import dependencies
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F

# Define demo corpus
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

def tokenize_sent(sent):
    '''
       Given a sentence return a list of tokens.
    '''
    # Break sentence into words
    return sent.split()

def tokenize_corpus(corpus):
    '''
       Tokenize all the sentences in a corpus.
    '''
    # Tokenize each sentence in a corpus
    return [tokenize_sent(sent) for sent in corpus]

def get_vocabulary(tokenized_corpus):
    '''
       Get all the unique words in a tokenized corpus.
    '''
    vocab = []
    for sent in tokenized_corpus:
        for token in sent:
            if token not in vocab:
               vocab.append(token)
    return vocab

# Why do we need to index the vocabulary? Does the indexing affect performance?
def index_vocabulary(vocab):
    '''
       Given a vocabulary index it word to index and index to word.

       Returns two dictionaries that contain the index.
    '''
    return {w: idx for (idx, w) in enumerate(vocab)}, {idx: w for (idx, w) in enumerate(vocab)}

def get_word_pairs(word2idx, tokenized_corpus, window_size):
    '''
       Take a tokenized corpus and window size parameters and return
    
       index of pairs.
    '''
    idx_pairs = []
    # Go through each sentence in the document
    for sentence in tokenized_corpus:
        # Get vocablulary indices
        indices = [word2idx[word] for word in sentence]
        for center_word_pos in range(len(indices)):
            # For every word withing the window
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # Error handling make sure we don't look for an index that's outside the sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                # Get current word index
                context_word_idx = indices[context_word_pos]
                # Append to the list of pairs
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    return idx_pairs

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1;
    return x

t_corp = tokenize_corpus(corpus)
vocab = get_vocabulary(t_corp)
word2idx, idx2word = index_vocabulary(vocab)
idx_pairs = get_word_pairs(word2idx, t_corp, 2)
vocabulary_size = len(vocab)

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data[0]
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
       print("Loss at epo {" + str(epo) + "}:" + str(loss_val/len(idx_pairs))) 
 


