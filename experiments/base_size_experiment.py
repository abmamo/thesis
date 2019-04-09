# experiment to test the relationship between accuracy and
# trainding data size

# import csv module to save results of experiment
import csv

# Ipmort the model, the trainer and the generator
from model.trainer import Trainer
from model.generator import Generator

# set experiment parameters

BASE_SIZES = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
TRAINING_SIZES = [200000, 500000]
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno'

def train_model(base, training_size, length=2, choice=5, epochs=2000, batch_size=1000, dimension = 100, testing_size=100):
    '''
       Function to generate data and train model.
    '''
    # initialise generator with given alphabet
    alphabet = ALPHABET[:base]
    g = Generator(alphabet, length, choice)

    # generate training and test data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)

    # initialise trainer class
    trainer = Trainer(train_data, test_data, epochs, dimension)

    # run model on generated data
    model = trainer.batch_train()
 
    # evaluate the model on the training and testing data
    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    # return the base size and the accuracies achieved with that base size
    return (base, train_acc, test_acc)

def run_experiment():
    print('experimenting with base size')
    # iterate through all specified training sizes
    for training_size in TRAINING_SIZES:
        # define an empty list to store results of each base size experiment
        results = []
        for base_size in BASE_SIZES:
            result = train_model(base=base_size, training_size = training_size)
            # add result to lsit
            results.append(result)
        # save results list as CSV
    for training_size in TRAINING_SIZES:
        results = []
        for base_size in BASE_SIZES:
            result = train_model(base=base_size, training_size = training_size)
            results.append(result)
        with open("baseline/results/base_size_experiment/base_size_experiment_training_size_" + str(training_size) + ".csv", "w") as f:
             writer = csv.writer(f)
             writer.writerows(results)

# do not run the experiment when this is imported
if __name__ == '__main__':
    run_experiment()
