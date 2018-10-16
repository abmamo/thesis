# when getting list of words from wordnet add a function that returns words with similar specificity

# generate hypernyms function

# for 4 hyponym words go one step above and to the the left or to the right and get one different word
# save as a pickle file


# import wordnet
from nltk.corpus import wordnet as wn


class PuzzleGenerator():
      def __init__(self):
          self.puzzles = {}

      def get_words(self, n = 500):
          '''
             Method to get all the words from wordnet
             and return them as a list
          '''
          # used set to make sure the words are unique
          return list(set(list(wn.words())))[:n]
      
      def get_hypernyms(self, word, depth=1):
          '''
             Method takes a given word and returns
             the hypernyms of all the senses of that word
          '''
          # if we have reached the depth we want stop
          if depth == 0:
             return set()
          # create empty list to store hypernyms
          hypoernyms = set()
          # get the hypernyms for each sense of the given word
          for sense in wn.synsets(word):
              # get the hypernyms of teh current sense
              for h in sense.hypernyms():
                  hyperms.add(h)
                  for z in get_hypoernyms(word, depth - 1):
                      hypernyms.add(z)
          return hypernyms

      def generate_hypernyms(self):
          '''
             Method gets a list of words from wordnet and
             returns a mapping of each word and its hyponyms
          '''
          # define dictionary to store mapping
          mapping = {}
          # get all teh words from wordnet
          words = self.get_words()
          # for each word get its hyponyms
          for word in words:
              mapping[word] = self.get_hypernyms(word)
          # return word : hyponyms pairs
          return mapping 
 
      def get_hyponyms(self, word, depth=1):
          '''
             Method takes a given word and returns 
             the hyponyms of all the senses of that word.
          '''
          # if we have reached the depth we want stop
          if depth == 0:
             return set()
          # create empty list to store hyponyms
          hyponyms = set()
          # get the hyponym for each sense of the given word
          for sense in wn.synsets(word):
              # get the hyponyms of the current sense
              for h in sense.hyponyms():
                  hyponyms.add(h)
                  # get the hyponyms on the next depth level
                  for z in get_hyponyms(word, depth -1):
                      hyponyms.add(z)
          return hyponyms

      def generate_hyponyms(self):
          '''
             Method gets a list of words from wordnet and
             returns a mapping of each word and its hyponyms
          '''
          # define dictionary to store mapping
          mapping = {}
          # get all teh words from wordnet
          words = self.get_words()
          # for each word get its hyponyms
          for word in words:
              mapping[word] = self.get_hyponyms(word)
          # return word : hyponyms pairs
          return mapping

      def filter_puzzles(self, mapping):
          '''
             Method takes a dictionary containing key value
             pairs of words and hyponyms and returns pairs
             that have 4 or more hyponyms.
          '''
          # define a dictionary to store pairs that satisfy our condition
          filtered_mapping = {}
          # iterate through our mapping to see if it satisfies the condition
          for key, value in mapping:
              if len(value) >= 4:
                 filtered_mapping[key] = value
          # return the filtered mapping
          return filtered_mapping
     
      def generate(self):
          puzzles = self.generate_hyponyms()
          for the least common anscestor of similar words:
              get hyponym of that word:
                  get hypernym of that word:
                      use that as the odd word

puzzle_generator = PuzzleGenerator()
puzzles = puzzle_generator.generate_hyponyms()
for puzzle in puzzles:
    print(puzzle)
    print(puzzles[puzzle])
    print()
