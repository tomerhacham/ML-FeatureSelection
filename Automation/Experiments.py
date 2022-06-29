import time

from sklearnex import patch_sklearn

patch_sklearn()
from ReliefF import ReliefF
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, make_scorer, \
    average_precision_score
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
                metrics = {'ACC': lambda y_true, y_score:accuracy_score(y_true,np.argmax(y_score,axis=1)),
                           'MCC': lambda y_true, y_score:matthews_corrcoef(y_true,np.argmax(y_score,axis=1)),
                            'PR-AUC':lambda y_true, y_score: average_precision_score(np.identity(n_classes)[y_true],y_score)}
                if n_classes > 2:
                    metrics['AUC'] = lambda y_true, y_score: roc_auc_score(y_true, y_score,multi_class='ovr', average='micro')
                else:
                    metrics['AUC'] = roc_auc_score

                X, y = _X[:, k_best_features], _y
                result = {'fit-time':[],
                          'inference-time':[],
                          'ACC':[],
                          'AUC':[],
                          'MCC':[],
                          'PR-AUC':[],
                          }
                for train, test in cv_method.split(X,y):
                    X_train=X[train,:]
                    y_train = y[train]
                    X_test = X[test,:]
                    y_test = y[test]
                    start_time = time.time()

                    clf.fit(X_train, y_train)
                    fit_time = time.time() - start_time

                    y_pred = clf.predict_proba(X_test)
                    inference_time = time.time() - start_time - fit_time

                    for metric in metrics:
                        score = metrics[metric](y_test,y_pred)
                        result[metric].append(score)


