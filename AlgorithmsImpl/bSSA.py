from random import random, randint
import numpy as np
import math

from numpy.random._generator import default_rng
from sklearn.decomposition import PCA, FastICA


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
            pass
        else:
            pass

        pass

    pass


def calculateFitness(salp, X, y):
    pass


def updateFollower(param, param1):
    pass


def bSSA(X, y, binarization_threshold=0.6, population_size=20, maxIter=70):
    # A. FEATURE TRANSFORMATIO
    X = dataTransformation(X)

    # B. BINARY SALP SWARM ALGORITHM
    # salps = [[randint(0, 1) for _ in range(X.shape[1])] for _ in range(population_size)]  # creates initial population
    salps = [np.where(default_rng(42).random(X.shape[1]) > binarization_threshold, 1, 0) for _ in
             range(population_size)]
    Ymax = np.ones(X.shape[1])
    Ymin = np.zeros(X.shape[1])
    updateLeader = generateUpdateLeaderFunction(Ymax, Ymin)
    currIter = 0
    while currIter < maxIter:
        salps.sort(key=lambda salp: calculateFitness(salp, X, y), reverse=True)
        a1 = 2 * (math.e ** (-(4 * currIter / maxIter) ** 2))  # Eq.3
        for j in range(len(salps)):
            if j == 0:
                newLocation = updateLeader(salps[j], a1)  # Modify position of leader salp using Eq. 2
                salps[j] = np.where(newLocation>binarization_threshold,1,0)  # convert them into binary using threshold δ
            else:
                newLocation = updateFollower(salps[j], salps[j - 1])
                salps[j] = np.where(newLocation>binarization_threshold,1,0)  # convert them into binary using threshold δ
