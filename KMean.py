import random
import numpy as np
import math
import copy

class KMean(object):

    def __init__(self):
        self.label_class = None
        self.contains = None
        self.labels = 2
        self.epsilon = 0.001

    def distance(self, a, b):
        return math.sqrt(np.sum(np.array([a[i] - b[i] for i in range(len(a))]) ** 2))

    def estimate(self, features, label):
        c = []
        for i in range(len(features)):
            max_like = 0
            distance = self.distance(features[i], label[max_like])
            for j in range(len(label)):
                if self.distance(features[i], label[j]) < distance:
                    max_like = j
            c.append(max_like)
        return c


    def maximize(self, features, label, con):
        l = copy.copy(label)
        for i in range(len(label)):
            numerator = 0
            denominator = 0
            for j in range(len(features)):
                if con[j] == i:
                    numerator += features[j]
                    denominator += 1
            if denominator != 0:
                l[i] = numerator / denominator
            else:
                l[i] = 0
        return l

    def fit(self, features):
        self.label_class = np.random.random((self.labels, len(features[0])))
        for i in range(len(features[0])):
            self.label_class[:, i] = self.label_class[:, i] * (max(features[:, i]) - min(features[:, i])) + min(features[:, i])
        while True:
            previous = copy.copy(self.label_class)
            self.contains = self.estimate(features, self.label_class)
            self.label_class = self.maximize(features, self.label_class, self.contains)
            label_class_score = np.sum(self.label_class - previous)
            if label_class_score < self.epsilon:
                break
        return "train done!"


if __name__ == "__main__":
    kmean = KMean()
    a = np.array([1,2,6,2,2,7,7,3,8,7,7])
    c = np.array([[random.normalvariate(1, 0.1), random.normalvariate(5,0.1)] for _ in range(100)])
    d = np.array([[random.normalvariate(4, 0.1), random.normalvariate(3,0.1)] for _ in range(100)])
    e = np.concatenate((c, d))
    # c = kmean.estimate(a,np.array([4.3, 4.2]))
    # print(c)
    # print(kmean.maximize(a, np.array([2.2, 6.5]), c))
    kmean.fit(e)
    print(kmean.label_class)