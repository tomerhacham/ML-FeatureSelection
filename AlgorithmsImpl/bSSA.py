from random import random, randint
import numpy as np
import math

from numpy.random._generator import default_rng
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def dataTransformation(X):
    _X = PCA().fit_transform(X)
    _X = FastICA().fit_transform(_X)
    return _X


def updateLeader(leaderLocation, a3):
    pass


def generateUpdateLeaderFunction(Ymax, Ymin):
    def updateLeader(leaderLocation, a1):
        a2, a3 = random(), random()
        if a3 >= 0.5:
            return leaderLocation + a1 * (Ymax * a2 + Ymin)
        else:
            return leaderLocation - a1 * (Ymax * a2 + Ymin)

    return updateLeader

def generateFitnessFucntion(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    def calculateFitness(salp):
        selected_features = np.where(salp==1)[0].tolist()
        knnClassifier = NearestNeighbors()
        knnClassifier.fit(X[:,selected_features],y_train)
        score = knnClassifier.score(X_test[:,selected_features,y_test])
        return score

    return calculateFitness

def bSSA(X, y, binarization_threshold=0.6, population_size=20, maxIter=70):
    X = dataTransformation(X)                               # data transformation
    calculateFitness = generateFitnessFucntion(X,y)         # generate lazy score function
    salps = [np.where(default_rng(42).random(X.shape[1]) > binarization_threshold, 1, 0)[0] for _ in
             range(population_size)]  # creates initial population
    Ymax = np.ones(X.shape[1])
    Ymin = np.zeros(X.shape[1])

    updateLeader = generateUpdateLeaderFunction(Ymax, Ymin)     # generate lazy update function
    currIter = 0
    while currIter < maxIter:
        salps.sort(key=lambda salp: calculateFitness(salp), reverse=True)
        a1 = 2 * (math.e ** (-(4 * currIter / maxIter) ** 2))  # Eq.3
        for j in range(len(salps)):
            newLocation = updateLeader(salps[j], a1) if j == 0 else (salps[j] + salps[j - 1])/2
            salps[j] = np.where(newLocation > binarization_threshold, 1, 0)[0]  # convert them into binary using threshold δ

    salps.sort(key=lambda salp: calculateFitness(salp), reverse=True)
    return salps[0].tolist()
