import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.io import arff


def load_data(dataset_name, root_dir='../'):
    name_to_function = {'ALL', lambda: load_bioconductors_files(os.path.join(root_dir, 'ALL.csv'))}
    X, y = name_to_function[dataset_name]()
    return X, y


def scale_y_values(y):
    y_unique_values = y.unique()
    y_unique_values.sort()
    y_unique_values = y_unique_values.tolist()
    for i in range(len(y_unique_values)):
        y.replace({y_unique_values[i]: i}, inplace=True)
    return y


def load_bioconductors_files(filename):
    data = pd.read_csv(filename)
    X, y = data.loc[:, data.columns != 'Y'], data['Y']
    y = scale_y_values(y)
    return X, y


def load_skfeatures_files(filename):
    data = sio.loadmat(filename)
    X, y = pd.DataFrame(data['X'], columns=[f'X{i}' for i in range(1, data['X'].shape[1] + 1)]), data['Y']
    y = scale_y_values(y)
    return X, y


def load_ARFF_files(filename):
    data = pd.DataFrame(arff.loadarff(filename)[0])
    y_column = list(filter(lambda x: x.lower() == 'class', data.columns))[0]
    X, y = data.loc[:, data.columns != y_column], data[y_column]
    y = scale_y_values(y)
    return X, y


def load_Misc_files(filename):
    data = pd.read_csv(filename)
    y_column = list(filter(lambda x: x.lower() == 'class' or x.lower() == 'response', data.columns))[0]
    X, y = data.loc[:, data.columns != y_column], data[y_column]
    y = scale_y_values(y)
    return X, y
