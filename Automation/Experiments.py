from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Loading the dataset

# Define pipeline
preprocess_pipeline = Pipeline([('simpleImputer', SimpleImputer()),
                 ('varianceThreshold', VarianceThreshold()),
                 ('powerTransformer',PowerTransformer())])
datasets=[]


def load_dataset(dataset):
    pass


def get_CV_method(_X):
    pass


for dataset in datasets:
    dataframe=load_dataset(dataset)
    X,y=
    _X,_y = preprocess_pipeline.fit_transform(X,y)
    for fs_method in list([]):
        score, p_value = SelectKBest(score_func=fs_method,k=100)
        for K in list([100,50,30,25,20,15,10,5,4,3,2,1]):
            k_best_features =  np.argpartition(score, -K)[-K:]
            for clf in clfList:
                cv_method = get_CV_method(_X)
