# experiment to test the relationship between accuracy and
# trainding data size

import csv

# Ipmort the model and the generator
from model.nn import Trainer
from model.generator import Generator


LARGE_TRAINING_SIZE = [1000000, 2000000, 4000000]
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmno'

def train_model(base, training_size, length=2, choice=5, epochs=2000, batch_size=1000, dimension = 100, testing_size=100, num_hidden = 2):
    # initialise generator
    alphabet = ALPHABET[:base]
    g = Generator(alphabet, length, choice)

    # generate data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)

    # initialise model
    trainer = Trainer(train_data, test_data, epochs, dimension, num_hidden)

    # run model on generated data
    model = trainer.batch_train(1000)

    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    return (base, train_acc, test_acc)

def run_experiment():
    results = []
    for training_size in LARGE_TRAINING_SIZE:
        result = train_model(base=42, training_size = training_size)
        results.append(result)
        with open("baseline/results/num_hidden_experiments/large_training_size_experiment_training_size_" + str(training_size) + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(results)

if __name__ == '__main__':
    print('experimenting with number of neurons in hidden layer') 
    run_experiment()
