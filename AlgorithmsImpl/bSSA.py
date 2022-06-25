from sklearnex import patch_sklearn
patch_sklearn()
from random import random
import numpy as np
import math

from numpy.random._generator import default_rng
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def dataTransformation(X):
    _X = PCA().fit_transform(X)
    _X = FastICA().fit_transform(_X)
    return _X


def generateUpdateLeaderFunction(Ymax, Ymin):
    def updateLeader(leaderLocation, a1):
        a2, a3 = random(), random()
        if a3 >= 0.5:
            newLocation = leaderLocation + (a1 * a2 * Ymax)
        else:
            newLocation = leaderLocation - (a1 * a2 * Ymax)
        return newLocation
    return updateLeader


def generateFitnessFucntion(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    def calculateFitness(salp):
        if np.all((salp == 0)):
            return 0
        selected_features = np.where(salp == 1)[0].tolist()
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(X_train[:, selected_features], y_train)
        score = knn_classifier.score(X_test[:, selected_features], y_test)
        return score

    return calculateFitness


def bSSA(X, y, binarization_threshold=0.6, population_size=20, maxIter=10,verbose=0):
    #X = dataTransformation(X)  # data transformation
    calculateFitness = generateFitnessFucntion(X, y)  # generate lazy score function
    salps = [np.where(default_rng(42).random(X.shape[1]) > binarization_threshold, 1, 0) for _ in
             range(population_size)]  # creates initial population
    Ymax = np.ones(X.shape[1])
    Ymin = np.zeros(X.shape[1])

    updateLeader = generateUpdateLeaderFunction(Ymax, Ymin)  # generate lazy update function
    for currIter in range(1,maxIter):
        if verbose>0:
            print(f'{currIter}/{maxIter}')
        salps.sort(key=lambda salp: calculateFitness(salp), reverse=True)
        a1 = 2 * (math.e ** (-(4 * currIter / maxIter) ** 2))  # Eq.3
        for j in range(len(salps)):
            newLocation = updateLeader(salps[j], a1) if j == 0 else (salps[j] + salps[j - 1]) / 2
            salps[j] = np.where(newLocation > binarization_threshold, 1, 0) # convert them into binary using threshold δ

    salps.sort(key=lambda salp: calculateFitness(salp), reverse=True)
    selectedFeatures = salps[0].tolist()
    return selectedFeatures
