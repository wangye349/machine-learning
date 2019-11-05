import numpy as np
def train_test_split(X, y, test_size, seed):
    assert X.shape[0] == y.shape[0], "X'size must be the same as y's"
    assert 0 <= test_size <= 1, "test_size must be between 0 and 1"

    if seed:
        np.random.seed(seed)

    shuffle_index = np.random.permutation(len(X))
    test_size = int(test_size * len(X))
    X_train = X[shuffle_index[:test_size]]
    X_test = X[shuffle_index[test_size:]]

    y_train = y[shuffle_index[:test_size]]
    y_test = y[shuffle_index[test_size:]]

    return X_train, X_test, y_train, y_test
