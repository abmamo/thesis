# experiment to test the relationship between accuracy and
# trainding data size

# import csv module to save results of experiment
import csv

# import pickle module to save python data s python objects


# Ipmort the model, the trainer and the generator
from model.trainer import Trainer
from model.generator import Generator

# set experiment parameters
#HIDDEN_SIZES = [100, 200, 400]
HIDDEN_SIZES = [100, 200]
#BASE_SIZES = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
BASE_SIZES = [16, 18]
#TRAINING_SIZES = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000, 500000]
TRAINING_SIZES = [1000, 2000]
#NUM_CHOICES = [2, 3, 4, 5, 6, 7, 8]
NUM_CHOICES = [2, 3]
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno'


def train_model(base, training_size=200000, length=2, choice=5, epochs=2000, batch_size=1000, dimension=100, testing_size=100):
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
    return (train_acc, test_acc)

# Experiments


def run_base_size_experiment():
    print('----------------------------')
    print('experimenting with base size')
    print('----------------------------')
    # dict to save the results of different base sizes
    results = {}
    for base_size in BASE_SIZES:
        print('----------------------------')
        print('base size: ' + str(base_size))
        print('----------------------------')
        result = train_model(base=base_size)
        # add result to lsit
        results[base_size] = result
    return results


def run_hidden_layer_size_experiment():
    print('-------------------------------------')
    print('experimenting with hidden layer sizes')
    print('-------------------------------------')
    # define list to store results for all parameters
    results = {}
    # iterate through each hidden layer size
    for hidden_size in HIDDEN_SIZES:
        print('--------------------------------')
        print('hidden size: ' + str(hidden_size))
        print('--------------------------------')
        # initialise the model with the hidden size parameter
        result = train_model(base=36, training_size=500000,
                             dimension=hidden_size)
        # add the results to the end
        results[hidden_size] = result
    return results


def run_num_choices_experiment():
    print('-------------------------------------')
    print('experimenting with hidden layer sizes')
    print('-------------------------------------')
    # define list to store results for all parameters
    results = {}
    # iterate through each hidden layer size
    for choice in NUM_CHOICES:
        print('--------------------------------')
        print('num. of choices: ' + str(choice))
        print('--------------------------------')
        result = train_model(base=36, training_size=500000, choice=choice)
        # Add result to results list
        results[choice] = result
    return results


def run_training_size_experiment():
    print('--------------------------------')
    print('experimenting with training size')
    print('--------------------------------')
    # define dict to store results
    results = {}
    # iterate through each training size
    for training_size in TRAINING_SIZES[::-1]:
        print('-------------------------------------')
        print('training data size: ' + str(training_size))
        print('-------------------------------------')
        result = train_model(base=36,
                             training_size=training_size)
        results[training_size] = result
    return results


def run_curriculum_learning_experiment():
    pass


if __name__ == '__main__':
    # run all experiments
    base_size_results = run_base_size_experiment()
    hidden_layer_size_results = run_hidden_layer_size_experiment()
    num_choices_results = run_num_choices_experiment()
    training_size_results = run_training_size_experiment()
    # aggregate all results in a dictionary
    all_results = {'base_size': base_size_results, 'hidden_layer_size': hidden_layer_size_results,
                   'num_choices': num_choices_results, 'training_size': training_size_results}
    # save dictionary to pickle file for easy visualization
    with open('all_results.pickle', 'wb') as f:
        pickle.dump(all_results, f)
