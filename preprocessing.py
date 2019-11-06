import numpy as np

class StandardScaler():

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = [np.mean(X[:,i]) for i in X.shape[1]]
        self.scale_ = [np.std(X[:,i]) for i in X.shape[1]]
        return self

    def transform(self, X):
        X_temp = np.empty(shape=X.shape, dtype=float);
        for i in range(X.shape[1]):
            X_temp[:,i] = (X[:,i] - self.mean_[i]) / self.scale_[i]
        return X_temp