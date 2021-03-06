# Scalable Learning for the Odd-Man-Out Task with Applications to Word Vector Induction

## Overview

word2vec is a popular software package used to generate vector representation of words. It implements several models that attempt to predict a given word based on its neighbors. One major drawback of generating vector representations by leveraging window based occurrence information in a given corpus is the frequency at which words occur in a text may not be an accurate representation of their real word occurrence frequencies (reporting bias). In addition, words with different meanings but similar syntactic properties might occur in the same context. And given a small enough window size, the vector representations generated by window based models can be similar (good & bad). Last but not least window based methods do not generally address the issue of polysemy. A word is considered polysemous if it can give two different meanings based on the context it is used in. When training word vectors using window based methods, the context surrounding a given word determines its vector representation; which implies two polysemous words can have completely different representations. 
 
In my dissertation I explored the effects of changing the objective function used for training word2vec models and explicitly requiring it to disambiguate between the different senses of words and in the process generate word embeddings that capture the sense relationships between words. The novel task is the odd-man-out task.

The task is the following: given a puzzle consisting of 3 or more words, choose the one which does not belong with the others.
```
	P1 =[car, motorcycle, bus, chariot, apple] 
```
where the first four words are related, as they are a means of transport, and can be grouped under the class vehicle. The word “apple”, on the other hand, is a fruit and cannot be grouped in the same category and therefore is the odd word out. 

## Quickstart

To run all the experiments and see the relationships between various parameters such as hidden layer size vs classifier accuracy you can:

- Clone the repository
```
   git clone https://github.com/abmamo/thesis.git
```
- create a python virtual environment
```
   python3 -m venv env
```
- install all requirements in requirements.txt
```
   pip install -r requirements.txt
```
- to run all the experiments exploring the effects various parameters from scrach you can 
```
   python experiments.py
```
the experiments.py file uses the python multiprocessing module to execute all experiments simultaneously and optimize the code runtime for CPUs.
- to look at the results of the experiments in a Tkinter gui
```
   python visualize.py
```

## Documentation
The model is divided into four major components/classes: the generator, the vectorizer, the neural network, and the trainer.

### Model
#### Generator

The main purpose of the generator class is to generate synthetic data for the novel task. We decided to generate synthetic data since it is easy to generate in large quantities and allows us to create unambiguous puzzles. In the synthetic data generated two words in a puzzle are “associated” with a “sense” if they share a particular letter. For instance in the puzzle [ABC,BDE,BGA,CBD,AFH] the words ABC, BDE, BGA, CBD are associated with the sense B. Similarly the words ABC, BGA, and AFH are associated with the sense A. 

The Generator class has 4 attributes:

- Alphabet - the alphabet for generating novel odd man out puzzles
- Length - length of each string in an odd man out puzzle
- Choices - number of choices in an odd man out puzzle
- Puzzles - originally initialized as NULL used to store generated puzzles

and 2 methods:

- generate_puzzle - generates one odd man out puzzle of the form [ABC,BDE,BGA,CBD,AFH] from a given alphabet
- generate_data - given an integer size = n, it generates n odd man out puzzles.


#### Vectorize 

Is a module that serves as a middleware to convert the primarily string based puzzles generated by the above class into pytorch friendly tensor format. It is comprised of 6 functions:

- _oneHot_: is a function given a word and a vocabulary containing that word that returns a one hot vector representation of the word

- _makePuzzleVector_: is a function that accepts a puzzle and a vocabulary and returns a puzzle matrix made up of the one hot encoding of the words in the puzzle.

- _makePuzzleTarget_: convert list of labels associated with the data into a pytorch tensor. For instance for the puzzle P5 = (AB,AF,AI,AK,HE) the position of “HE” can be encoded by P5 = [0, 0, 0, 0, 1] which is then converted into a tensor.

- _makePuzzleMatrix_: function to convert a collection fo puzzles into a collection of puzzle matrices

- _buildVocab_: is a function to build a vocabulary from a given set of puzzles. It iterates through each puzzle and gets the words in each puzzle.

#### Neural Network

The model used in this experiment is a pytorch multilayer classifier initialized with a vocabulary, number of labels/choices, hidden layer size, and number of hidden layers. It takes the input (The size of the input layer is euqal to the size of the vocab, |V|, times the puzzle size.), feeds it through several layers one after the other, and then finally gives the output which is done through the forward method. The dump method is used to get the weights of the neural network at a given time. It also has a static method _initialize_from_model_and_vocab_ that is used to initialize the weights of the network from a preexisting neural network.

#### Trainer

The trainer module is a collection of functions used to train and evaluate the network. The class primarily serves as a way of interfacing with the network and training it with different parameters. It accepts training data, testing data, parameters for training such as epochs batch size and an old model to initialize weights from if available. It has two training methods: _train_(iterative) and batch_train. In addition it has a evaluate model that tests the data on a testing dataset.

### Experiments & Visualization

The _experiments.py_ module serves as a gridsearch function that trains the model with various parameters for training data size, puzzle length and so on and stores the training and testing accuracy of the classifier with different parameters in a pickle file. The _visualize.py_ module then reads this pickle file and displays the trends in training and testing classifier accuracy across various experiments.
