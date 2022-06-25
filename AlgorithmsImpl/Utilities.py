import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_selection import f_classif


def ScoreWrapper(binaryFeatureSelector):
    def activeSelector(X, y, *args, **kwargs):
        selectedFeaturesVector = binaryFeatureSelector(X, y, *args, **kwargs)
        invertedVector = np.invert(np.array(selectedFeaturesVector).astype(bool)).astype(int)
        scores, pvalues = f_classif(X,y)
        (scores * invertedVector) + (np.finfo(float).max * selectedFeaturesVector)
