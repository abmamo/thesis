# experiment to test the relationship between accuracy and
# trainding data size

import csv

# Ipmort the model and the generator
from model import Trainer
from generator import Generator


BASE_SIZES = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
TRAINING_SIZES = [200000, 500000]
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno'

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
    model = trainer.batch_train(1000)

    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    return (base, train_acc, test_acc)

def run():
    for training_size in TRAINING_SIZES:
        results = []
        for base_size in BASE_SIZES:
            result = train_model(base=base_size, training_size = training_size)
            results.append(result)
        with open("baseline/results/base_size_experiment/base_size_experiment_training_size_" + str(training_size) + ".csv", "w") as f:
             writer = csv.writer(f)
             writer.writerows(results)
if __name__ == '__main__':
    run()
