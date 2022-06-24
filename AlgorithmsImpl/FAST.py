from skfeature.utility.mutual_information import su_calculation
from .PrimMST import Graph
import itertools
from joblib import Parallel, delayed



def generateGraph(X,S):
    def generateEdge(pair,graph):
        fi, fj = pair
        f_correlation = su_calculation(X[fi], X[fj])
        graph.addEdge(fi, fj, f_correlation)
        return
    graph = Graph(len(S))
    pairs_of_features = list(itertools.combinations([i for i in range(len(S))], 2))
    Parallel(require='sharedmem')(delayed(generateEdge)(pair,graph) for pair in pairs_of_features)

    # for pair in pairs_of_features:
    #     fi, fj = pair
    #     f_correlation = su_calculation(X[fi], X[fj])
    #     graph.addEdge(fi, fj, f_correlation)
    return graph


def createTrees(nodes, edges):
    def getNeighbors(v):
        neighbors = set([edge[1]] for edge in filter(lambda edge: edge[0] == v, edges))
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
        visited = [False for i in nodes]
        cc = []
        for v in nodes:
            if not visited[v]:
                temp = []
                cc.append(DFS(temp, v, visited))
        return cc

    cc = connectedComponents(nodes)
    return cc


def FAST(X, y, t_relevance_threshold=0.06):
    S = set()
    _X=X.to_numpy()
    _y=y.to_numpy()
    # ==== Part 1: Irrelevant Feature Removal ====
    features = list(X.columns)
    for f in range(len(features)):
        t_relevance = su_calculation(_X[f], _y)
        if t_relevance > t_relevance_threshold:
            S.add(f)
        else:
            print(f'{f}:{t_relevance}')

    # ==== Part 2: Minimum Spanning Tree Construction ====
    graph = generateGraph(_X,S)
    forest = graph.PrimMST()

    # ==== Part 3: Tree Partition and Representative Feature Selection ====
    for edge in forest.copy():
        i, j = edge
        su_fifj = su_calculation(_X[i], _X[j])
        su_fic = su_calculation(_X[i], _y)
        su_fjc = su_calculation(_X[j], _y)
        if su_fifj < su_fic and su_fifj < su_fjc:
            forest.remove(edge)

    S.clear()
    trees = createTrees(features, forest)
    for tree in trees:
        su_list = [(f, su_calculation(_X[f], _y)) for f in tree]
        su_list.sort(key=lambda x: x[1], reverse=True)
        fr = su_list[0]
        S.add(fr)
    return S
