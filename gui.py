import os
# get results directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = BASE_DIR + '/results'
# get all csv files recursively from directory
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(RESULTS_DIR):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
