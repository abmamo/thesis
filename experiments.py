# experiment to test the relationship between accuracy and
# trainding data size
from datetime import datetime
# import pickle module to save python data s python objects
import pickle
# Import the model, the trainer and the generator
from model.trainer import Trainer
from model.generator import Generator
# import mprocessing module to run multiple experiments at the same time
from multiprocessing import Process, Manager

# set experiment parameters
HIDDEN_SIZES = [100, 200, 400]
BASE_SIZES = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
TRAINING_SIZES = [1000, 2000, 4000, 8000,
                  16000, 32000, 64000, 128000, 200000, 500000]
NUM_CHOICES = [2, 3, 4, 5, 6, 7, 8]
STRIDES = [2, 4, 8]
START_BASE_SIZES = [8, 16, 32]
MAX_BASE = 40
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789'


def train_model(base, training_size=200000, length=2, choice=5, epochs=2000, batch_size=1000, dimension=100, testing_size=100, old_model=None):
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
    # initialise model
    if old_model == None:
        trainer = Trainer(train_data, test_data, epochs, dimension)
    else:
        trainer = Trainer(train_data, test_data, epochs, dimension, old_model)

    # run model on generated data
    model = trainer.batch_train()

    # evaluate the model on the training and testing data
    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    # return the base size and the accuracies achieved with that base size
    return (train_acc, test_acc)

# Experiments


def run_base_size_experiment(all_results):
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
    all_results['base_size'] = results


def run_hidden_layer_size_experiment(all_results):
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
    all_results['hidden_layer_size'] = results


def run_num_choices_experiment(all_results):
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
    all_results['num_choices'] = results


def run_training_data_size_experiment(all_results):
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
    all_results['training_data_size'] = results


def run_curriculum_learning_experiment():
    # currently being modified
    print('--------------------------------')
    print('experimenting with curriculum learning')
    print('--------------------------------')
    # define dict to store results
    results = {}
    for base_size in START_BASE_SIZES:
        prev_model, prev_trainer, train_acc, test_acc = train_model(
            base=base_size)
        for stride in STRIDES:
            while base_size <= MAX_BASE:
                base_size = base_size + stride
                old_model = MultiLayerClassifier.initialize_from_model_and_vocab(
                    prev_model, prev_trainer.vocab)
                model, trainer, train_acc, test_acc = train_model(
                    base=base_size, old_model=old_model)
        results[base_size]
        results.append((train_acc, test_acc))


if __name__ == '__main__':
    # start timer for entire experiment
    start = datetime.now()
    # define processing manager
    manager = Manager()
    # define shared variable to communicate
    all_results = manager.dict()
    # define a list to keep track of experiments
    experiments = []
    # experiment with base size parameter
    base_size_process = Process(
        target=run_base_size_experiment, args=(all_results,))
    experiments.append(base_size_process)
    base_size_process.start()
    # experiment with hidden layer size parameter
    hidden_layer_size_process = Process(
        target=run_hidden_layer_size_experiment, args=(all_results,))
    experiments.append(hidden_layer_size_process)
    hidden_layer_size_process.start()
    # experiment with num choices parameter
    num_choices_process = Process(
        target=run_num_choices_experiment, args=(all_results,))
    experiments.append(num_choices_process)
    num_choices_process.start()
    # experiment with training data size parameters
    training_data_size_process = Process(
        target=run_training_data_size_experiment, args=(all_results,))
    experiments.append(training_data_size_process)
    training_data_size_process.start()
    # re join spawned processes to get value
    for experiment in experiments:
        experiment.join()
    # show time
    print('-----------------------------')
    print('Time elapsed: {}'.format(datetime.now()-start))
    print('-----------------------------')
    # save dictionary to pickle file for easy visualization
    with open('all_results.pickle', 'wb') as f:
        pickle.dump(all_results, f)
