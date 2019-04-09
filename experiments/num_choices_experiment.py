# experiment to test the relationship between accuracy and
# number of choices in the puzzle

import csv

# Ipmort the model and the generator
from model.trainer import Trainer
from model.generator import Generator


# Define experiment parameters
BASE_SIZES = [16]
TRAINING_SIZES = [500000]
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno'
NUM_CHOICES = [2,3,4,5,6,7,8]

def train_model(base, training_size, length=2, choice=5, epochs=2000, batch_size=1000, dimension = 100, testing_size=100):
    # initialise generator
    alphabet = ALPHABET[:base]
    g = Generator(alphabet, length, choice)

    # generate data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)

    # initialise model
    trainer = Trainer(train_data, test_data, epochs, dimension)

    # run model on generated data
    model = trainer.batch_train()

    # evaluate the model on training and testing data
    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    # return the choice and the accuracy
    return (choice, train_acc, test_acc)

def run_experiment():
    print('experimenting with base size')
    # iterate through training sizes
    for training_size in TRAINING_SIZES:
        results = []
        # iterate through all base sizes
        for base_size in BASE_SIZES:
            # iterate through all choice
            for choice in NUM_CHOICES:
                result = train_model(base=base_size, training_size = training_size, choice=choice)
                # Add result to results list
                results.append(result)
            # Save the list as a csv
            with open("results/num_choices_experiment/num_choices_experiment.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(results)

# do not run when imported
if __name__ == '__main__':
    run_experiment()
