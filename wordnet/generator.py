import random
from nltk.corpus import wordnet as wn

def get_hypernyms(word, depth = None):
    '''
       Method that takes a given word and returns
       the hypernyms of all that word.

       If a depth parameter is passed it gets hypernyms
       up to that depth.
    '''
    if depth == None:
       # if we have reached the depth we want or 
       # if we have reached a leaf stop
       if wn.synsets(word)[0].hypernyms() == []:
          # return accumulated list
          return []
       # create empty list to store hypernyms
       hypernyms = []
       # otherwise recursively add hypernyms
       for sense in wn.synsets(word):
           # get the hypernyms of current sense
           for h in sense.hypernyms():
               hypernyms.append(h)
               for z in get_hypernyms(word):
                   hypernyms.append(z)
       return hypernyms
    else:
       if depth == 0:
          return []
       # create empty list to store hypernyms
       hypernyms = []
       # get the hypernyms for each sense of the given word
       for sense in wn.synsets(word):
           # get the hypernyms of teh current sense
           for h in sense.hypernyms():
               hypernyms.append(h)
               for z in get_hypernyms(word, depth - 1):
                   hypernyms.append(z)
       return hypernyms

def get_hyponyms(word, depth = None):
    '''
       Method that takes a given word and returns
       the hyponyms of all that word.

       If a depth parameter is passed it gets hypernyms
       up to that depth.
    '''
    if depth == None:
       # if we have reached the depth we want or 
       # if we have reached a leaf stop
       if wn.synsets(word)[0].hyponyms() == []:
          # return accumulated list
          return []
       # create empty list to store hypernyms
       hyponyms = []
       # otherwise recursively add hypernyms
       for sense in wn.synsets(word):
           # get the hypernyms of current sense
           for h in sense.hyponyms():
               hyponyms.append(h)
               for z in get_hyponyms(word):
                   hyponyms.append(z)
       return hyponyms
    else:
       if depth == 0:
          return []
       # create empty list to store hypernyms
       hyponyms = []
       # get the hypernyms for each sense of the given word
       for sense in wn.synsets(word):
           # get the hypernyms of teh current sense
           for h in sense.hyponyms():
               hyponyms.append(h)
               for z in get_hyponyms(word, depth - 1):
                   hyponyms.append(z)
       return hyponyms

def get_similar(root, depth = 3):
    '''
       given a root word return four random hyponyms
       of the root word

       depth is artibrarily set
    '''
    # get hyponyms
    hyponyms = get_hyponyms(root, depth = depth)
    # randomly select 4 words from the list 
    return random.sample(hyponyms, 4)

def is_leaf(word, depth = 1):
    '''
       function that checks to see if a given word is 
       a leaf in wordnet
    '''
    return get_hyponyms(word, depth = depth) == []

def get_odd(root, depth = 3):
    '''
       given a root word go up a given number of steps
       and generate a word with a similar ancestor
    '''
    # get hypernym of the root word
    hypernyms = get_hypernyms(root, depth = depth)
    # randomly select hypernym
    odd_root = random.choice(hypernyms).lemmas()[0].name()
    # find hyponyms of the root word
    hyponyms = get_hyponyms(odd_root, depth = 3)
    # check if a hyponym is a leaf
    leaves = [x for x in hyponyms if is_leaf(x.lemmas()[0].name())]
    # randomly select a leaf
    return random.choice(leaves)
    
def generate_puzzle(root):
    ''' given a root word get 4 random hyponyms and a hypernym connected to it through a common anscetor'''
    similar = get_similar(root)
    odd = get_odd(root)
    return {"similar" : similar, "odd" : odd}


#hypernyms = get_hypernyms('dog', depth = 3)
#print(hypernyms)
#hyponyms = get_hyponyms('dog', depth = 3)
#print(hyponyms)
sim = get_similar('dog')
print(sim)
#odd = get_odd('dog')
#print(odd)
#puzzles = generate_puzzle('dog')
#print(puzzles)
