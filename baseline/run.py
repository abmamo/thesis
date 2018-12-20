from experiments import base_size_experiment, hidden_size_experiment, training_size_experiment, large_training_size_experiment, num_hidden_experiment

def main():
    base_size_experiment.run_experiment()
    num_hidden_experiment.run_experiment()
    hidden_size_experiment.run_experiment()
    training_size_experiment.run_experiment()
    large_training_size_experiment.run_experiment()

if __name__ == '__main__':
    main()
