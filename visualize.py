import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pickle

from pandas import DataFrame

import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")


# import data
# open a file, where you stored the pickled data
file = open('all_results.pickle', 'rb')
# dump information to that file
all_results = pickle.load(file)
# close the file
file.close()

root = tk.Tk()

for experiment_name in all_results:
    # get the results
    results = all_results[experiment_name]
    # move results into a pandas dataframe
    df = DataFrame(results).T
    df.columns = ['training acc.', 'testing acc.']
    df.index.name = experiment_name
    # define figure
    figure = plt.Figure(figsize=(5, 4), dpi=100)
    # add plot
    ax = figure.add_subplot(111)
    # define line chart canvas figure
    line = FigureCanvasTkAgg(figure, root)
    line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    # plot dataframe
    df.plot(kind='line', legend=True, ax=ax,
            marker='o', fontsize=10)
    ax.set_xlabel(experiment_name)
    ax.set_ylabel('classifier_accuracy')
    ax.set_title(experiment_name + ' vs. classifier_accuracy')
root.mainloop()
