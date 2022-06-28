from ReliefF import ReliefF
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFdr, f_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, cross_validate
from skfeature.function.information_theoretical_based.MRMR import mrmr
import numpy as np

# NB, SVM, LogisticsRegression, RandomForest, k-nearest neighbors (K-NN
classifiers = [('NB',lambda: MultinomialNB()),
               ('SVM', lambda: SVC()),
               ('LogisticsRegression', lambda: LogisticRegression()),
               ('RandomForest', lambda: RandomForestClassifier()),
               ('K-NN', lambda: KNeighborsClassifier()),
               ]

fs_methods = [('mRMR', lambda X, y: mrmr(X, y)),
              ('f_classIf', lambda X, y: SelectFdr(alpha=0.1).fit(X,y).scores_),
              ('RFE', lambda X, y: RFE(estimator=SVC()).fit(X,y).scores_),
              ('ReliefF', lambda X, y: ReliefF().fit(X,y).feature_scores if X.shape[0] > 100 else ReliefF(n_neighbors=X.shape[0] // 5).fit(X,y).feature_scores )
              ]

# Define pipeline
preprocess_pipeline = Pipeline([('simpleImputer', SimpleImputer()),
                                ('varianceThreshold', VarianceThreshold()),
                                ('powerTransformer', PowerTransformer())])
datasets = []


def load_dataset(dataset):
    pass


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


for dataset in datasets:
    X, y = load_dataset(dataset)
    _X, _y = preprocess_pipeline.fit_transform(X, y)
    for fs_method_name, fs_method in list([]):
        selectKBest = SelectKBest(score_func=fs_method, k=100).fit(_X, _y)
        score = selectKBest.scores_
        for K in list([100, 50, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]):
            k_best_features = np.argpartition(score, -K)[-K:]
            for name, generate_func in classifiers:
                clf = generate_func()
                cv_method = get_CV_generator(_X)
