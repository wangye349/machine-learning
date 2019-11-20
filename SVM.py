import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class SVM(object):
    def __init__(self, kernel="linear", max_iter=100):
        self.kernel = kernel
        self.max_iter = max_iter

    def init_args(self, features, labels):
        self.m, self.n = features.shape # m represents the numbers of input, n represents the dimension of input
        self.X = features
        self.y = labels
        self.alpha = np.zeros(self.m) #[1 for i in range(self.m)]
        self.b = 0.0
        self.E = [self._E(i) for i in range(self.m)]
        self.C = 1

    def _g(self, i):
        g_temp = self.b
        for j in range(self.m):
            g_temp += self.alpha[j] * self.y[j] * self._kernel(self.X[i], self.X[j])
        return g_temp

    def _E(self, i):
        return self._g(i) - self.y[i]

    def _kernel(self, x, y):
        if self.kernel == "linear":
            return sum([x[k] * y[k] for k in range(len(x))])
        if self.kernel == "poly":
            return (sum([x[k] * y[k] for k in range(len(x))] + 1)) ** 2
        return 0

    def _KKT(self, i):
        yg = self.y[i] * self._g(i)
        if self.alpha[i] == 0:
            return yg >= 1
        elif 0 < self.alpha[i] < self.C:
            return yg == 1
        elif self.alpha[i] == self.C:
            return yg <= 1
        else:
            return 0

    def init_alpha(self):
        index_list = [i for i in range(len(self.alpha)) if 0 <= self.alpha[i] <= self.C]
        index_not_satisfied = [i for i in range(len(self.alpha)) if i not in index_list]
        index_list.extend(index_not_satisfied)

        for i in index_list:
            if self._KKT(i):
                continue
            EI = self.E[i]
            if EI >= 0:
                max_ = min(range(self.m), key=lambda x: self.E[x])
            else:
                max_ = max(range(self.m), key=lambda x: self.E[x])
            # pre = abs(self.E[index_list[0]] - EI)
            # max_ = 0
            # for j in index_list:
            #     if j == i:
            #         continue
            #     if abs(self.E[j] - EI) > pre:
            #         max_ = j
            return i, max_

    def fit(self, features, labels):
        self.init_args(features, labels)
        for i in range(self.max_iter):
            i1, i2 = self.init_alpha()
            alpha1_old = self.alpha[i1]
            alpha2_old = self.alpha[i2]
            if self.y[i1] != self.y[i2]:
                L = max(0, alpha2_old - alpha1_old)
                H = min(self.C, self.C + alpha2_old - alpha1_old)
            else:
                L = max(0, alpha2_old + alpha1_old - self.C)
                H = min(self.C, alpha2_old + alpha1_old)
            E1 = self.E[i1]
            E2 = self.E[i2]
            K11 = self._kernel(self.X[i1], self.X[i1])
            K22 = self._kernel(self.X[i2], self.X[i2])
            K12 = self._kernel(self.X[i1], self.X[i2])
            K21 = self._kernel(self.X[i2], self.X[i1])
            eta = K11 + K22 - K12
            alpha2_new_unc = alpha2_old + self.y[i2] * (E1 - E2) / eta
            if alpha2_new_unc > H:
                alpha2_new = H
            elif L <= alpha2_new_unc <= H:
                alpha2_new = alpha2_new_unc
            else:
                alpha2_new = L

            alpha1_new = alpha1_old + self.y[i1] * self.y[i2] * (alpha2_old - alpha2_new)
            b_old = self.b
            b1_new = -E1 - self.y[i1] * K11 * (alpha1_new - alpha1_old) - self.y[i2] * K21 * (alpha2_new - alpha2_old) + b_old
            b2_new = -E2 - self.y[i1] * K12 * (alpha1_new - alpha1_old) - self.y[i2] * K22 * (alpha2_new - alpha2_old) + b_old
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return "train done"

    def predict(self, feature):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.y[i] * self._kernel(self.X[i], feature)
        return 1 if r >= 0 else -1

    def score(self, X_test, y_test):
        score = 0
        test_number = len(X_test)
        print(sum([self.alpha[i] * self.y[i] for i in range(self.m)]))
        for i in range(test_number):
            y_predict = self.predict(X_test[i])
            if y_predict == y_test[i]:
                score += 1
        return score / float(test_number)



def create_data():
    iris = load_iris()
    print(type(iris))
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]

def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:, 0], X[50:, 1], label='1')
    plt.show()

    score = []
    svm = [i for i in range(10)]
    svm[1] = SVM(max_iter=200)
    svm[1].fit(X_train, y_train)
    score.append(svm[1].score(X_test, y_test))
    print(score)

    svm_sklearn = SVC(gamma='auto')
    svm_sklearn.fit(X_train, y_train)
    print (svm_sklearn.score(X_test, y_test))

if __name__ == "__main__":
    main()