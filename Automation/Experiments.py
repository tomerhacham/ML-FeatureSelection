import time

from sklearnex import patch_sklearn

patch_sklearn()
from ReliefF import ReliefF
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, make_scorer
from AlgorithmsImpl.FAST import FAST
from AlgorithmsImpl.bSSA import bSSA, bSSA__New
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFdr, RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, cross_validate
from skfeature.function.information_theoretical_based.MRMR import mrmr
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer

# NB, SVM, LogisticsRegression, RandomForest, k-nearest neighbors (K-NN
classifiers = [('NB', lambda: Pipeline([('minMaxScaler', MinMaxScaler()), ('nb', MultinomialNB())])),
               ('SVM', lambda: SVC()),
               ('LogisticsRegression', lambda: LogisticRegression()),
               ('RandomForest', lambda: RandomForestClassifier()),
               ('K-NN', lambda: KNeighborsClassifier()),
               ]
# mRMR, f_classIf, RFE, ReliefF
fs_methods = [  # /('bSSA', lambda X, y: bSSA(X,y)),
    # ('bSSA_New', lambda X, y: bSSA__New(X, y)),
    # ('FAST', lambda X, y: FAST(X, y)),
    ('mRMR', lambda X, y: mrmr(X, y)),
    ('SelectFdr', lambda X, y: SelectFdr(alpha=0.1).fit(X, y).scores_),
    ('RFE', lambda X, y: RFE(estimator=SVC()).fit(X, y).scores_),
    ('ReliefF',
     lambda X, y: ReliefF().fit(X, y).feature_scores if X.shape[0] > 100 else ReliefF(n_neighbors=X.shape[0] // 5).fit(
         X, y).feature_scores)
]

# Define pipeline
preprocess_pipeline = Pipeline([('simpleImputer', SimpleImputer()),
                                ('varianceThreshold', VarianceThreshold()),
                                ('powerTransformer', PowerTransformer())])
datasets = ['as']


def get_CV_generator(X):
    n_sample = X.shape[0]
    if n_sample > 1000:
        return KFold(n_splits=5)
    elif n_sample > 100:
        return KFold(n_splits=10)
    elif n_sample > 50:
        return LeaveOneOut()
    else:
        return LeavePOut(2)


def load_dataset():
    dataset = pd.read_csv('../datasets/X_y_train.csv')
    dataset = dataset.sample(200)
    X, y = dataset.loc[:, dataset.columns != 'y'], dataset['y']
    return X, y


def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    .. deprecated:: 0.24

        The _pairwise attribute is deprecated in 0.24. From 1.1
        (renaming of 0.26) and onward, this function will check for the
        pairwise estimator tag.

    Labels y will always be indexed only along the first axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.

    """
    if _is_pairwise(estimator):
        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


for dataset in datasets:
    X, y = load_dataset()
    features_names = list(X.columns)
    _X = preprocess_pipeline.fit_transform(X, y)
    _y = y.to_numpy()
    for fs_method_name, fs_method in fs_methods:
        selectKBest = SelectKBest(score_func=fs_method, k=100)
        selectKBest.fit(_X, _y)
        score = selectKBest.scores_
        for K in list([100, 50, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]):
            k_best_features = np.argpartition(score, -K)[-K:]
            for name, generate_func in classifiers:
                clf = generate_func()
                cv_method = get_CV_generator(_X)
                #  ACC, MCC ,AUC,  PR-AUC. יצוין כי בבעיות Multi-class
                n_classes = y.nunique(dropna=False)
                print(f'n_classes:{n_classes}')
                metrics = {'ACC': make_scorer(accuracy_score),
                           'MCC': make_scorer(matthews_corrcoef)}
                # 'PR-AUC':make_scorer(precision_recall_curve)}
                metrics = {}
                if n_classes > 2:
                    metrics['AUC'] = make_scorer(roc_auc_score, multi_class='ovr', average='micro')
                else:
                    metrics['AUC'] = make_scorer(roc_auc_score)

                # clf.fit(X=_X[:,k_best_features],y=_y,)
                # prediction = clf.predict_proba(X=_X[:,k_best_features])
                # cv_score = cross_validate(estimator=clf,X=_X[:,k_best_features],y=_y,cv=cv_method,scoring=metrics,error_score='raise')
                # print(cv_score)
                X, y = _X[:, k_best_features], _y
                for train, test in cv_method.split(X,y):
                    X_train=X[train,:]
                    y_train = y[train]
                    X_test = X[test,:]
                    y_test = y[test]
                    start_time = time.time()


                    result = {}
                    clf.fit(X_train, y_train)
                    fit_time = time.time() - start_time
                    y_pred = clf.predict_proba(X_test)
                    test_scores = roc_auc_score(y_true=y_test, y_score=y_pred,multi_class = 'ovr', average = 'macro')
                    score_time = time.time() - start_time - fit_time
