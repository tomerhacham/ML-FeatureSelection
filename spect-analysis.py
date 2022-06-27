import pandas as pd
from AlgorithmsImpl.FAST import FAST
from AlgorithmsImpl.bSSA import bSSA

dataset = pd.read_csv('spect/SPECTF.train',header=None)
X, y = dataset.iloc[:, 1:], dataset[0]
# features = FAST(X.to_numpy(),y.to_numpy())
features = bSSA(X.to_numpy(),y.to_numpy(), binarization_threshold=0.6, population_size=3, maxIter=3)

dataset.head()
