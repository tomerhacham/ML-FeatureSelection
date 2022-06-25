import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_selection import f_classif
from functools import wraps


def WithScores(func):
    @wraps(func)
    def score_amplifier(X, y, *args, **kwargs):
        selectedFeaturesVector = func(X, y, *args, **kwargs)
        invertedVector = np.invert(np.array(selectedFeaturesVector).astype(bool)).astype(int)
        scores, pvalues = f_classif(X,y)
        amplified_scores = (scores * invertedVector) + ((np.max(scores)+1) * np.array(selectedFeaturesVector))
        return amplified_scores

    return score_amplifier
