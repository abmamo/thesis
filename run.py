from experiments import base_size_experiment, hidden_size_experiment, training_size_experiment, large_training_size_experiment, num_hidden_experiment, curriculum_learning_experiment, num_choices_experiment

def main():
    num_choices_experiment.run_experiment()
    #base_size_experiment.run_experiment()
    #num_hidden_experiment.run_experiment()
    #hidden_size_experiment.run_experiment()
    #training_size_experiment.run_experiment()
    #large_training_size_experiment.run_experiment()
    #curriculum_learning_experiment.run_experiment()

if __name__ == '__main__':
    main()
