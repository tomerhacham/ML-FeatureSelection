from sklearnex import patch_sklearn
patch_sklearn()

import math
from ReliefF import ReliefF
from skfeature.utility.mutual_information import su_calculation
from .PrimMST import Graph
import itertools
from joblib import Parallel, delayed

def generateGraph(X, S, nodes_map):
    def calculateWeight(pair):
        fi, fj = pair
        fi_tag, fj_tag = nodes_map[fi], nodes_map[fj]
        f_correlation = su_calculation(X[fi_tag], X[fj_tag])
        return (fi, fj, f_correlation)

    graph = Graph(len(S))
    pairs_of_features = list(itertools.combinations([i for i in range(len(S))], 2))
    print(f'pairs: {len(pairs_of_features)}')
    edges = Parallel(n_jobs=-1,
                     verbose=10)(delayed(calculateWeight)(pair) for pair in pairs_of_features)

    for edge in edges:
        fi, fj, f_correlation = edge
        graph.addEdge(fi, fj, f_correlation)

    return graph

def createTrees(edges):
    def getNeighbors(v):
        neighbors = set([edge[1] for edge in filter(lambda edge: edge[0] == v, edges)])
        return neighbors

    def DFS(temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in getNeighbors(v):
            if visited[i] == False:
                # Update the list
                temp = DFS(temp, i, visited)
        return temp

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(nodes):
        visited = {}
        for v in nodes:
            visited[v] = False
        cc = []
        for v in nodes:
            if not visited[v]:
                temp = []
                cc.append(DFS(temp, v, visited))
        return cc

    nodes = set([edge[0] for edge in edges]).union(set([edge[1] for edge in edges]))
    cc = connectedComponents(nodes)
    return cc

#A Fast Clustering Based Feature Subset Selection
def FAST(X, y, t_relevance_threshold=None):
    '''
    A Fast Clustering Based Feature Subset Selection
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.
    y : array-like of shape (n_samples,)
        The target vector
    :param t_relevance_threshold: float or None, minimum t-relevant threshold for irrelevant feature removal
    :return: vector represent for each feature if it was selected or not
    '''
    S = set()
    #_X = X.to_numpy()
    #_y = y.to_numpy()
    if t_relevance_threshold is None:
        rf = ReliefF()
        rf.fit(X, y)
        index = math.floor(math.sqrt(len(X))*math.log(len(X)))
        feature = rf.top_features[index]
        t_relevance_threshold=su_calculation(X[feature],y)

    # # ==== Part 1: Irrelevant Feature Removal ====
    # features = list(X.columns)
    features = [i for i in range(X.shape[1])]
    for f in features:
        t_relevance = su_calculation(X[f], y)
        if t_relevance > t_relevance_threshold:
            S.add(f)
        else:
            print(f'{f}:{t_relevance}')

    # ==== Part 2: Minimum Spanning Tree Construction ====
    nodes_map = [node for node in S]
    graph = generateGraph(X, S, nodes_map)
    _forest = graph.primMST()
    forest = [(nodes_map[edge[0]], nodes_map[edge[1]]) for edge in _forest]

    # ==== Part 3: Tree Partition and Representative Feature Selection ====
    for edge in forest.copy():
        i, j = edge
        su_fifj = su_calculation(X[i], X[j])
        su_fic = su_calculation(X[i], y)
        su_fjc = su_calculation(X[j], y)
        if su_fifj < su_fic and su_fifj < su_fjc:
            forest.remove(edge)

    S.clear()
    trees = createTrees(forest)
    for tree in trees:
        su_list = [(f, su_calculation(X[f], y)) for f in tree]
        su_list.sort(key=lambda x: x[1], reverse=True)
        fr = su_list[0][0]
        S.add(fr)
    vector = [1 if f in S else 0 for f in features]
    return vector
    return S
