import pickle
from model import Trainer
from generator import Generator

BASE_10_TRAINING_SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
BASE_16_TRAINING_SIZES = [800000, 400000, 200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]
CHOICES = [3, 5] 
BASES = [10, 16]

def run_experiment(base, training_size, length=2, choice=5, epochs=10, batch_size=1000, dimension = 100, testing_size=100):
    # initialise generator
    if base == 10:
        g = Generator('0123456789', length, choice)
    elif base == 16:
        g = Generator('abcdefghijklmnop', length, choice)
    else:
        print("You have to manually add the alphabet for the given base size")

    # generate data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)
    
    print("data generated")

    # initialise model
    trainer = Trainer(train_data, test_data, epochs, dimension)

    # run model on generated data
    model = trainer.train()
    print('length of the alphabet is = {}'.format(base))
    print('length of individual strings is = {}'.format(length))
    print('number of choices in a puzzle is = {}'.format(choice))
    print('number of epochs is = {}'.format(epochs))
    print('training data size is = {}'.format(training_size))
    print('testing data size is = {}'.format(testing_size))
    print('training accuracy = {}'.format(trainer.evaluate(model, trainer.train_data)))
    print('test accuracy = {}'.format(trainer.evaluate(model, trainer.test_data)))


def run():
    for training_size in BASE_10_TRAINING_SIZES:
            for base in BASES:
                run_experiment(base=base, training_size=training_size)



run() 
