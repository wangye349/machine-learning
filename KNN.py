from math import sqrt
import numpy as np
from collections import Counter
from metrics import accuracy_score

class KNNClassifier:

    def __init__ (self, k):
        assert k > 1, "k must be valid"
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to size of y"
        assert self.k < X.shape[0], \
        "the size of X must be at least k"
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X_predict):
        assert self.X_train is not None and self.y_train is not None, \
        "must be fitted before"
        assert X_predict.shape[1] == self.X_train.shape[1], \
        "the feature number of x must be the same as X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return y_predict

    def _predict(self, x):
        assert x.shape[0] == self.X_train[1], \
        "the shape of x must be the same as X_train"
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        nearest = np.argsort(distances)

        topK_y = [self.y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_predict, y_test)

    def __repr__(self):
        return "KNN(k=%d)" % self.k