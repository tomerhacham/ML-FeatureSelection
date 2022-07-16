import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.io import arff

def scale_y_values(y):
    y_unique_values = y.unique()
    y_unique_values.sort()
    y_unique_values = y_unique_values.tolist()
    for i in range(len(y_unique_values)):
        y.replace({y_unique_values[i]: i}, inplace=True)
    return y

root = '../datasets/ARFF'
files_path = [os.path.join(root, x) for x in os.listdir(root)]
files_path = list(filter(lambda x: x.endswith('.arff'), files_path))
for file in files_path:
    # data = pd.read_csv(file)
    data = pd.DataFrame(arff.loadarff(file)[0])
    y_column = list(filter(lambda x: x.lower() == 'class', data.columns))[0]

    X, y = data.loc[:, data.columns != y_column],data[y_column]
    y = scale_y_values(y)
    print()
    # y_unique_values = data.Class.unique()
    # print(f'-----{file}-----\ncolumns:{data.shape[1]}, rows:{data.shape[0]}, classes:{data.Class.nunique()}\n')
    # y_unique_values.sort()
    # y_unique_values=y_unique_values.tolist()
    # for i in range(len(y_unique_values)):
    #     data.replace({y_unique_values[i]:i},inplace=True)




def load_bioconductors_files(filename):
    data = pd.read_csv(filename)
    X, y = data.loc[:, data.columns != 'Y'],data[y_column]
    y = scale_y_values(y)
    return X,y


def load_skfeatures_files(filename):
    data = sio.loadmat(filename)
    X,y = pd.DataFrame(data['X'], columns=[f'X{i}' for i in range(1, data['X'].shape[1] + 1)]),data['Y']
    y = scale_y_values(y)
    return X,y


def load_ARFF_files(filename):
    data = pd.DataFrame(arff.loadarff(filename)[0])
    y_column = list(filter(lambda x: x.lower() == 'class', data.columns))[0]
    X, y = data.loc[:, data.columns != y_column], data[y_column]
    y = scale_y_values(y)
    return X,y
