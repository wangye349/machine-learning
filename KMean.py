import random
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

class KMean(object):

    def __init__(self):
        self.label_class = None # label_class is a 2-dim array, each example of it is a coordinate, the label of it is its serial number
        self.contains = None # contains is an array to classify each example, like [1,1,0,1] represent\
                             # the 1st 2nd 4th example is classified to 1 label, the 3rd example is classified to 0 label
        self.labels = 2 # how many labels would you like to classify the inputs
        self.epsilon = 0.00001 # mimimum receptable score

    # distance function compute the EuclideanDistance between vector a and b
    def distance(self, a, b):
        return math.sqrt(np.sum(np.array([a[i] - b[i] for i in range(len(a))]) ** 2))

    # estimate function helps get self.contains
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

    # given features, label class and contains to update label class
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
        # loop the E-M step till the score is less than the previous given number
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
    c = np.array([[random.normalvariate(1, 1), random.normalvariate(3,1)] for _ in range(100)])
    d = np.array([[random.normalvariate(4, 1), random.normalvariate(1,1)] for _ in range(100)])
    plt.scatter(c[:,0], c[:,1],c='red')
    plt.scatter(d[:,0], d[:,1],c='blue')
    plt.show()
    e = np.concatenate((c, d))
    # c = kmean.estimate(a,np.array([4.3, 4.2]))
    # print(c)
    # print(kmean.maximize(a, np.array([2.2, 6.5]), c))
    kmean.fit(e)
    print(kmean.label_class)
#TODO: finish the code and make it clear