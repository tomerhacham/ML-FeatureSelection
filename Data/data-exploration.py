import os
import scipy.io as sio
import numpy as np
import pandas as pd

root = '../datasets/bioconductors'
files_path = [os.path.join(root,x) for x in os.listdir(root)]
files_path = list(filter(lambda x:x.endswith('.csv'),files_path))
#print(files_path)
for file in files_path:
    data = pd.read_csv(file)

    y_unique_values = data.Y.unique()
    print(f'-----{file}-----\ncolumns:{data.shape[1]}, rows:{data.shape[0]}, classes:{data.Y.nunique()}\n')
    # y_unique_values.sort()
    # y_unique_values=y_unique_values.tolist()
    # for i in range(len(y_unique_values)):
    #     data.replace({y_unique_values[i]:i},inplace=True)


def scale_y_values(data):
    y_unique_values = data.Y.unique()
    y_unique_values.sort()
    y_unique_values=y_unique_values.tolist()
    for i in range(len(y_unique_values)):
        data.replace({y_unique_values[i]:i},inplace=True)

def load_bioconductors_files(filename):
    data = pd.read_csv(filename)
    scale_y_values(data)
    return data

def load_skfeatures_files(filename):
    data = sio.loadmat(filename)
    X = pd.DataFrame(data['X'],columns=[f'X{i}' for i in range(1,data['X'].shape[1]+1)])
    y = pd.DataFrame(data['Y'],columns=['Y'])
    data = pd.concat([X,y],axis=1)
    scale_y_values(data)
    return data