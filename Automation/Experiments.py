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
from statistics import mean

from sklearn.metrics import get_scorer

# NB, SVM, LogisticsRegression, RandomForest, k-nearest neighbors (K-NN
classifiers = [#('NB', lambda: Pipeline([('minMaxScaler', MinMaxScaler()), ('nb', MultinomialNB())])),
               #('SVM', lambda: SVC(probability=True)),
               #('LogisticsRegression', lambda: LogisticRegression()),
               #('RandomForest', lambda: RandomForestClassifier()),
               ('K-NN', lambda: KNeighborsClassifier()),
               ]
# mRMR, f_classIf, RFE, ReliefF
fs_methods = [  # /('bSSA', lambda X, y: bSSA(X,y)),
    # ('bSSA_New', lambda X, y: bSSA__New(X, y)),
    # ('FAST', lambda X, y: FAST(X, y)),
    ('mRMR', lambda X, y: mrmr(X, y)),
    ('SelectFdr', lambda X, y: SelectFdr(alpha=0.1).fit(X, y).scores_),
    ('RFE', lambda X, y:RFE(estimator=SVC(kernel="linear")).fit(X, y).ranking_),
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
        return '5Fold', KFold(n_splits=5)
    elif n_sample > 100:
        return '10Fold', KFold(n_splits=10)
    elif n_sample > 50:
        return 'Leave One Out', LeaveOneOut()
    else:
        return 'Leave PairOut', LeavePOut(2)


def load_dataset():
    dataset = pd.read_csv('../datasets/X_y_train.csv')
    dataset = dataset.sample(200)
    X, y = dataset.loc[:, dataset.columns != 'y'], dataset['y']
    return X, y


def get_metrics(n_classes):
    metrics = {'ACC': lambda y_true, y_score: accuracy_score(y_true, np.argmax(y_score, axis=1)),
               'MCC': lambda y_true, y_score: matthews_corrcoef(y_true, np.argmax(y_score, axis=1)),
               'PR-AUC': lambda y_true, y_score: average_precision_score(np.identity(n_classes)[y_true], y_score)}
    if n_classes > 2:
        metrics['AUC'] = lambda y_true, y_score: roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    else:
        metrics['AUC'] = roc_auc_score
    return metrics


def split_test_train(train_indexes, test_indexes, X, y):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    return X_train, y_train, X_test, y_test


def cross_validate(clf, X, y, cv_method, metrics):
    result = {'fit-time': [],
              'inference-time': [],
              'ACC': [],
              'AUC': [],
              'MCC': [],
              'PR-AUC': [],
              }
    for train, test in cv_method.split(X, y):
        X_train, y_train, X_test, y_test = split_test_train(train, test, X, y)
        start_time = time.time()  # start timer
        clf.fit(X_train, y_train)
        fit_time = time.time() - start_time  # measure fit time
        result['fit-time'].append(fit_time)
        y_pred = clf.predict_proba(X_test)
        inference_time = (time.time() - start_time - fit_time) / X_test.shape[0]  # measure inference time
        result['inference-time'].append(inference_time)
        for metric in metrics:
            score = metrics[metric](y_test, y_pred)
            result[metric].append(score)  # append score for each metric

    return result


def get_new_record_to_results():
    return {
        'Dataset Name': '',
        'Number of samples': '',
        'Original Number of features': '',
        'Filtering Algorithm': '',
        'Learning Algorithm': '',
        'Number of features selected (K)': '',
        'CV Method': '',
        'Fold': '',  # on all cv methods there is fold?
        'Measure Type': '',
        'Measure Value': '',
        'List of Selected Features Names': '',
        'Selected Features scores': '',
        'Feature Selection time': '',
        'Fit time': '',
        'Inference time per record': '',
    }


results_table = pd.DataFrame(columns=[key for key in get_new_record_to_results()])
for dataset in datasets:
    X, y = load_dataset()
    features_names = list(X.columns)
    _X, _y = preprocess_pipeline.fit_transform(X, y), y.to_numpy()
    for fs_method_name, fs_method in fs_methods:
        selectKBest = SelectKBest(score_func=fs_method, k=100)
        start_time = time.time()  # start timer
        selectKBest.fit(_X, _y)
        feature_selection_time = time.time() - start_time  # measure feature selection time
        score = selectKBest.scores_
        for K in list([100, 50, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]):
            k_best_features = np.argpartition(score, -K)[-K:]
            for clf_name, generate_func in classifiers:
                print(f'classifier:{clf_name}, FS:{fs_method_name}')
                clf = generate_func()
                cv_method_name,cv_method = get_CV_generator(_X)
                n_classes = y.nunique(dropna=False)
                print(f'n_classes:{n_classes}')
                metrics = get_metrics(n_classes)
                # region oldCV
                # result = {'fit-time': [],
                #           'inference-time': [],
                #           'ACC': [],
                #           'AUC': [],
                #           'MCC': [],
                #           'PR-AUC': [],
                #           }
                # for train, test in cv_method.split(_X[:, k_best_features], _y):
                #
                #     X_train, y_train, X_test, y_test = split_test_train(train, test, _X[:, k_best_features], _y)
                #     start_time = time.time()  # start timer
                #     clf.fit(X_train, y_train)
                #     fit_time = time.time() - start_time  # measure fit time
                #
                #     y_pred = clf.predict_proba(X_test)
                #     inference_time = (time.time() - start_time - fit_time) / X_test.shape[0]  # measure inference time
                #
                #     for metric in metrics:
                #         score = metrics[metric](y_test, y_pred)
                #         result[metric].append(score)  # append score for each metric
                # endregion

                cv_result = cross_validate(clf, _X[:, k_best_features], _y if clf_name !='KNN' else _y.ravel(), cv_method, metrics)
                avg_fit_time, avg_inference_time = mean(cv_result['fit-time']), mean(cv_result['inference-time'])
                for metric in metrics.keys():
                    record = {
                        'Dataset Name': dataset,
                        'Number of samples': X.shape[0],
                        'Original Number of features': X.shape[1],
                        'Filtering Algorithm': fs_method_name,
                        'Learning Algorithm': clf_name,
                        'Number of features selected (K)': K,
                        'CV Method': cv_method_name,
                        'Fold': '',  # on all cv methods there is fold?
                        'Measure Type': metric,
                        'Measure Value': mean(cv_result[metric]),
                        'List of Selected Features Names': tuple([features_names[i] for i in k_best_features]),
                        'Selected Features scores': tuple([score[i] for i in k_best_features]),
                        'Feature Selection time': feature_selection_time,
                        'Fit time': avg_fit_time,
                        'Inference time per record': avg_inference_time,
                    }
                    results_table.append(record,ignore_index=True)
    results_table.to_csv('results.csv')
