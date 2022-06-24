from random import random, randint
import numpy as np
import math

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

        pass

    pass


def bSSA(X, y, binarization_threshold=0.6, population_size=20, maxIter=70):
    # A. FEATURE TRANSFORMATIO
    X = dataTransformation(X)

    # B. BINARY SALP SWARM ALGORITHM
    salps = [[randint(0, 1) for _ in range(X.shape[1])] for _ in range(population_size)]  # creates initial population
    Ymax = [1 for _ in range(X.shape[1])]
    Ymin = [0 for _ in range(X.shape[1])]
    updateLeader = generateUpdateLeaderFunction(Ymax, Ymin)
    currIter = 0
    while currIter < maxIter:
        a1 = 2 * (math.e ** (-(4 * currIter / maxIter) ** 2))  # Eq.3
        for i in range(len(salps)):
            if i == 0:
                salps[i] = updateLeader(salps[i], a1) # Modify position of leader salp using Eq. 2
