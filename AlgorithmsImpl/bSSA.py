from sklearn.decomposition import PCA, FastICA


def dataTransformation(X,y):
    _X=PCA().fit_transform(X)
    _X = FastICA().fit_transform(_X)
    return _X

def bSSA(X,y,binarization_threshold):
    _X = X.to_numpy()
    _y = y.to_numpy()


