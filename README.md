# Exploring Training Regimens for Various Neural Network Architectures Using Curriculum Learning: A Case Study of the Odd-Man-Out Task

The amount of textual information available to us has increase significantly with the expansion of the World Wide Web. This has led to issues of managing and understanding the vast amount of textual data. One approach is to use various neural network architectures to perform natural language processing tasks. This code explores various training regimens that can be used on a given neural network with the ultimate goal of creating sense aware vector word embeddings.

More specifically, we initially explore the relationship between various parameters like number of words in a given puzzle, the training data size are explored initially. Various experiments for different training regimens are going to be added based on the results of these experiments and network behavior.

To run the initial set of experiments you can:

```sh
python run.py`
```
This script runs multiple experiments with different parameters for generating our data and handles the generation and vectorization of data in addition to training and evaluating the model. The main areas explored are the relationship between the model accuracy training data size, number of hidden layers and number of neurons in the hidden layer. The results of the experiments will be in the results folder grouped by experiment parameter. Their names will be indicative of which experiment they were outputed from. For instance the file

```sh
   training_size_experiment_base10_choice3.csv
```

is a result of running the network with various training sizes and a fixed based or size of alphabet and number of words or choices in the puzzles.

All the libraries in requirements.txt need to be installed to run the model. This can be done using pip
```sh
   pip install -r requirements.txt
```


