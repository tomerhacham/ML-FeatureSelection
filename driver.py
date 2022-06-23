from AlgorithmsImpl.FAST import FAST
import pandas as pd

dataset = pd.read_csv('datasets/X_y_train.csv')
X, y = dataset.loc[:, dataset.columns != 'y'], dataset['y']
features = FAST(X,y)