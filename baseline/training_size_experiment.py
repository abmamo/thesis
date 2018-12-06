# experiment to test the relationship between accuracy and
# trainding data size

import csv

# Ipmort the model and the generator
from model import Trainer
from generator import Generator

import os
cwd = os.getcwd()
print(cwd)

BASE_10_TRAINING_SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
BASE_16_TRAINING_SIZES = [800000, 400000, 200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]
BASES = [10, 16]
CHOICES = [3, 5]

def train_model(base, training_size, length=2, choice=3, epochs=2000, batch_size=1000, dimension = 100, testing_size=100):
    # initialise generator
    if base == 10:
        g = Generator('abcdefghij', length, choice)
    elif base == 16:
        g = Generator('abcdefghijklmnop', length, choice)
    else:
        print("You have to manually add the alphabet for the given base size")

    # generate data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)

    # initialise model
    trainer = Trainer(train_data, test_data, epochs, dimension)

    # run model on generated data
    model = trainer.batch_train(1000)

    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    return (training_size, train_acc, test_acc)

def run():
    for base in BASES:
        for choice in CHOICES:
            results = []
            for training_size in BASE_16_TRAINING_SIZES[::-1]:
                result =train_model(base = base, choice = choice, training_size = training_size )
                results.append(result)
            with open("results/training_size_experiment/training_size_experiment_base" + str(base) + "_choice" + str(choice) +  ".csv", "w") as f:
                 writer = csv.writer(f)
                 writer.writerows(results)
run()


