# import all experiments from the experiments folder
from experiments import base_size_experiment, hidden_size_experiment, training_size_experiment, large_training_size_experiment, num_choices_experiment

# define function to run all imported experiments


def main():
    # num_choices_experiment.run_experiment()
    base_size_results = base_size_experiment.run_experiment()
    print(base_size_results)
    # hidden_size_experiment.run_experiment()
    # training_size_experiment.run_experiment()
    # large_training_size_experiment.run_experiment()
    # curriculum_learning_experiment.run_experiment()


# do not run when imported
if __name__ == '__main__':
    main()
