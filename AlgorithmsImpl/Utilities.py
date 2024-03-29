import numpy as np
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.feature_selection import f_classif
from functools import wraps
from ReliefF import ReliefF


def WithScores(func):
    '''Util function the generate decorator in order to impute scores
     for algorithm that return features using binary representation'''
    @wraps(func)
    def score_imputer(X, y, *args, **kwargs):
        selectedFeaturesVector = func(X, y, *args, **kwargs)
        invertedVector = np.invert(np.array(selectedFeaturesVector).astype(bool)).astype(int)
        scores, pvalues = f_classif(X,y)
        amplified_scores = (scores * invertedVector) + ((np.max(scores)+1) * np.array(selectedFeaturesVector))
        return amplified_scores

    return score_imputer

def ReliefFFitter(X,y):
    '''Util function to wrap the fitting process of ReliefF in order to adhere the Sklearn convention'''
    reliefF = ReliefF() if X.shape[0] > 100 else ReliefF(n_neighbors=X.shape[0] // 5)
    reliefF.fit(X, y)
    return reliefF