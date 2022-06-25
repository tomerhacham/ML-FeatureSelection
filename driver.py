from AlgorithmsImpl.FAST import FAST
from AlgorithmsImpl.bSSA import bSSA
import pandas as pd
from ReliefF import ReliefF


dataset = pd.read_csv('datasets/X_y_train.csv')
X, y = dataset.loc[:, dataset.columns != 'y'], dataset['y']
#features = FAST(X.to_numpy(),y.to_numpy(),0.07)
features = bSSA(X.to_numpy(),y.to_numpy(),verbose=1)
print(features)