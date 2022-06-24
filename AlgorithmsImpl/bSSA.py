from sklearn.decomposition import PCA, FastICA


def dataTransformation(X):
    _X=PCA().fit_transform(X)
    _X = FastICA().fit_transform(_X)
    return _X

def bSSA(X,y,binarization_threshold=0.6):
    # A. FEATURE TRANSFORMATIO
    X = dataTransformation(X)

    # B. BINARY SALP SWARM ALGORITHM

