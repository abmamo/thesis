# experiment to test the relationship between accuracy and
# hidden layer size

# import  csv module to save results of experiment

import csv

# Ipmort the model and the generator
from model.trainer import Trainer
from model.generator import Generator


# set experiment parameters
HIDDEN_SIZES = [100, 200, 400]
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno'

def train_model(base, training_size, length=2, choice=5, epochs=2000, batch_size=1000, dimension = 100, testing_size=100):
    # initialise generator
    alphabet = ALPHABET[:base]
    g = Generator(alphabet, length, choice)

    # generate data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)

    # initialise trainer
    trainer = Trainer(train_data, test_data, epochs, dimension)

    # run model on generated data
    model = trainer.batch_train()

    # evaluate model on generated data
    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    # return base size and results
    return (base, train_acc, test_acc)

def run_experiment():
    print('experimenting with hidden layer sizes')
    # define list to store results for all parameters
    results = []
    # iterate through each hidden layer size
    for hidden_size in HIDDEN_SIZES:
        # initialise the model with the hidden size parameter
        result = train_model(base=36, training_size = 500000, dimension = hidden_size)
        # add the results to the end
        results.append(result)
        # save the results to a file
        with open("baseline/results/hidden_size_experiment/hidden_size_experiment_hidden_size_" + str(hidden_size) + ".csv", "w") as f:
             writer = csv.writer(f)
             writer.writerows(results)

# don't run it when imported
if __name__ == '__main__':
    run_experiment()
