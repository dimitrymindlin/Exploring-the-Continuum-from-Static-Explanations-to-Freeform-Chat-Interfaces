import os
import pandas as pd


def load_data(path):
    # Load all the csv files, and store them in a dictionary
    data = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            data[file.split(".")[0]] = pd.read_csv(os.path
                                                   .join(path, file), delimiter=',')
    return data
