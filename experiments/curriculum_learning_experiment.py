# experiment to test the relationship between accuracy and
# trainding data size

import csv

# Ipmort the model and the generator
from model.nn import MultiLayerClassifier
from model.trainer import Trainer
from model.generator import Generator


STRIDES = [2, 4, 8]
START_BASE_SIZES = [8, 16, 32]
MAX_BASE = 40
TRAINING_SIZE = 200000
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789'

def train_model(base, training_size, length=2, choice=5, epochs=2000, batch_size=1000, dimension = 100, testing_size=100, old_model = None):
    # initialise generator
    alphabet = ALPHABET[:base]
    g = Generator(alphabet, length, choice)

    # generate data
    train_data = g.generate_data(training_size)
    test_data = g.generate_data(testing_size)

    # initialise model
    if old_model == None:
       trainer = Trainer(train_data, test_data, epochs, dimension)
    else:
       trainer = Trainer(train_data, test_data, epochs, dimension, old_model)

    # run model on generated data
    model = trainer.batch_train()

    train_acc = trainer.evaluate(model, train_data[:200])
    test_acc = trainer.evaluate(model, test_data)

    return (model, trainer, train_acc, test_acc)

def run_experiment():
    print('experimenting curriculum learning')
    results = []
    for base_size in START_BASE_SIZES:
        prev_model, prev_trainer, train_acc, test_acc = train_model(base = base_size, training_size = TRAINING_SIZE)
        results.append((train_acc, test_acc))
        for stride in STRIDES:
            while base_size <= MAX_BASE:
                  base_size = base_size + stride
                  old_model = MultiLayerClassifier.initialize_from_model_and_vocab(prev_model, prev_trainer.vocab)
                  model, trainer, train_acc, test_acc = train_model(base = base_size, training_size = TRAINING_SIZE, old_model = old_model)
            results.append((train_acc, test_acc))
        with open("baseline/results/curriculum_learning_experiment/curriculum_learning_experiment.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(results)

if __name__ == '__main__':
    run_experiment()
