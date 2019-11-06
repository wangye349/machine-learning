import numpy as np

class StandardScaler():

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        assert X.ndim == 2, "the shape of X must be 2"
        self.mean_ = np.array([np.mean(X[:,i]) for i in X.shape[1]])
        self.scale_ = np.array([np.std(X[:,i]) for i in X.shape[1]])
        return self

    def transform(self, X):
        assert X.ndim == 2, "the shape of X must be 2"
        assert self.mean_ != None and self.scale_ != None, "the array must be fitted before transform"
        assert self.mean_.shape[1] == self.scale_.shape[1] == X.shape[1], "the shape of X must be the same as its mean adn std"
        X_temp = np.empty(shape=X.shape, dtype=float);
        for i in range(X.shape[1]):
            X_temp[:,i] = (X[:,i] - self.mean_[i]) / self.scale_[i]
        return X_temp