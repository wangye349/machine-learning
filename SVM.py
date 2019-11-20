class SVM(object):
    def __init__(self, kernel="linear", max_iter=100):
        self.kernel = kernel
        self.max_iter = max_iter

    def init_args(self, features, labels):
        self.m, self.n = features.shape # m represents the numbers of input, n represents the dimension of input
        self.X = features
        self.y = labels
        self.alpha = [0 for i in range(self.m)]
        self.b = 0
        self.E = [self._E(i) for i in range(self.m)]
        self.C = 1

    def _g(self, i):
        g_temp = self.b
        for j in range(self.m):
            g_temp += self.alpha[j] * self.y[j] * self.kernel(self.j, self.i)
        return g_temp

    def _E(self, i):
        return self._g(i) - self.y[i]

    def kernel(self, x, y):
        if (self.kernel == "linear"):
            return sum([x[k] * y[k] for k in range(len(x))])
        if (self.kernel == "poly"):
            return sum([x[k] * y[k] for k in range(len(x))] ** 2)
        return 0

    def _KKT(self, i):
        yg = self.y[i] * self._g(i)
        if (self.alpha[i] == 0):
            return yg >= 1
        elif (0 < self.alpha[i] < self.C):
            return yg == 1
        elif (self.alpha[i] == self.C):
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
            pre = abs(self.E[index_list[0]] - EI)
            for j in index_list:
                if j == i:
                    continue
                if abs(self.E[j] - EI) > pre:
                    max_ = j
            return i, j
