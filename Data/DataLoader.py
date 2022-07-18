import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.io import arff


def load_data(dataset_name, root_dir='../datasets'):
    name_to_function = {
        'ALL': lambda: load_bioconductors_files(os.path.join(root_dir, 'bioconductors/ALL.csv')),
        'ayeastCC': lambda: load_bioconductors_files(os.path.join(root_dir, 'bioconductors/ayeastCC.csv')),
        'bcellViper': lambda: load_bioconductors_files(os.path.join(root_dir, 'bioconductors/bcellViper.csv')),
        'bladderbatch': lambda: load_bioconductors_files(os.path.join(root_dir, 'bioconductors/bladderbatch.csv')),
        'CLL': lambda: load_bioconductors_files(os.path.join(root_dir, 'bioconductors/CLL.csv')),

        'Breast': lambda: load_ARFF_files(os.path.join(root_dir, 'ARFF/Breast.arff')),
        'CNS': lambda: load_ARFF_files(os.path.join(root_dir, 'ARFF/CNS.arff')),
        'Leukemia_4c': lambda: load_ARFF_files(os.path.join(root_dir, 'ARFF/Leukemia_4c.arff')),
        'Lymphoma': lambda: load_ARFF_files(os.path.join(root_dir, 'ARFF/Lymphoma.arff')),
        'SRBCT': lambda: load_ARFF_files(os.path.join(root_dir, 'ARFF/SRBCT.arff')),

        'ALLAML': lambda: load_skfeatures_files(os.path.join(root_dir, 'charliec443 scikit-feature master skfeature-data/ALLAML.mat')),
        'BASEHOCK': lambda: load_skfeatures_files(os.path.join(root_dir, 'charliec443 scikit-feature master skfeature-data/BASEHOCK.mat')),
        'CLL-SUB-111': lambda: load_skfeatures_files(os.path.join(root_dir, 'charliec443 scikit-feature master skfeature-data/CLL-SUB-111.mat')),
        'colon': lambda: load_skfeatures_files(os.path.join(root_dir, 'charliec443 scikit-feature master skfeature-data/colon.mat')),
        'GLIOMA': lambda: load_skfeatures_files(os.path.join(root_dir, 'charliec443 scikit-feature master skfeature-data/GLIOMA.mat')),

        'GDS4824': lambda: load_Misc_files(os.path.join(root_dir, 'Misc/GDS4824.csv')),
        'journal.pone.0246039.s002': lambda: load_Misc_files(os.path.join(root_dir, 'Misc/journal.pone.0246039.s002.csv')),
        'NCI60_Affy': lambda: load_Misc_files(os.path.join(root_dir, 'Misc/NCI60_Affy.csv')),
        'NCI60_Ross': lambda: load_Misc_files(os.path.join(root_dir, 'Misc/NCI60_Ross.csv')),
        'pone.0246039.s001': lambda: load_Misc_files(os.path.join(root_dir, 'Misc/pone.0246039.s001.csv')),
                        }
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
    X = pd.DataFrame(data['X'], columns=[f'X{i}' for i in range(1, data['X'].shape[1] + 1)])
    y = data['Y'].flatten()
    y = pd.Series(y)
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
