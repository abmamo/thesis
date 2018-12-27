# thesis

To run the experiment do:

`python run.py`

This script runs multiple experiments with different parameters for generating our data. The main areas explored are the relationship between the model accuracy training data size, number of hidden layers and number of neurons in the hidden layer. 

The results of the experiments will be in the results folder. Their names will be indicative of which experiment they were outputed from. For instance the file

```sh
   training_size_experiment_base10_choice3.csv
```

is a result of running the network with various training sizes and a fixed based or size of alphabet and number of words or choices in the puzzles.

All the libraries in requirements.txt need to be installed.
