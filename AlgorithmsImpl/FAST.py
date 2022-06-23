from skfeature.utility.mutual_information import su_calculation
from .PrimMST import Graph
import itertools


def generateGraph(features):
    graph = Graph(len(features))
    pairs_of_features = list(itertools.combinations([i for i in range(len(features))], 2))
    for pair in pairs_of_features:
        fi, fj = pair
        f_correlation = su_calculation(fi, fj)
        graph.addEdge(fi, fj, f_correlation)
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


def FAST(X, y, t_relevance_threshold=0.0):
    S = set()
    # ==== Part 1: Irrelevant Feature Removal ====
    features = list(X.columns)
    for f in features:
        t_relevance = su_calculation(X[f].value, y.value)
        if t_relevance > t_relevance_threshold:
            S.add(f)

    # ==== Part 2: Minimum Spanning Tree Construction ====
    graph = generateGraph(features)
    forest = graph.PrimMST()

    # ==== Part 3: Tree Partition and Representative Feature Selection ====
    for edge in forest.copy():
        i, j = edge
        su_fifj = su_calculation(X[features[i]].value, X[features[j]].value)
        su_fic = su_calculation(X[features[i]].value, y.value)
        su_fjc = su_calculation(X[features[j]].value, y.value)
        if su_fifj < su_fic and su_fifj < su_fjc:
            forest.remove(edge)

    S.clear()
    trees = createTrees(features, forest)
    for tree in trees:
        su_list = [(f, su_calculation(X[f].value, y.value)) for f in tree]
        su_list.sort(key=lambda x: x[1], reverse=True)
        fr = su_list[0]
        S.add(fr)
    return S
