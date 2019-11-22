import numpy as np
import math

class NaiveBayes(object):

    def __init__(self, alpha=1):
        self._condition_prob = None
        self._prior_prob = None
        self._label = None
        self.alpha = alpha

    def _calculate_prob(self, feature):
        label_X = np.unique(feature)
        classify = {x:0 for x in label_X}
        for x in feature:
            classify[x] += 1
        for x in label_X:
            classify[x] = (classify[x] + self.alpha) / float(len(feature) + self.alpha * (label_X))
        return classify

    def fit(self, features, labels):
        self._label = np.unique(labels)
        self._prior_prob = self._calculate_prob(labels)
        self._condition_prob = {x:{} for x in self._label}
        for label in self._label:
            for feature_mark in range(len(features[0])):
                self._condition_prob[label][feature_mark] = self._calculate_prob(features[:,feature_mark][labels == label])

    def predict(self, new_feature):
        prediction = self._label[0]
        max_prob = 0
        for label in self._label:
            prior_prob = self._prior_prob[label]
            condition_prob = 1
            for feature_mark in range(len(new_feature)):
                condition_prob *= self._condition_prob[label][feature_mark][new_feature[feature_mark]]
            if prior_prob * condition_prob > max_prob:
                prediction = label
                max_prob = prior_prob * condition_prob
        return prediction

#TODO: test GassianNB and improve it
class GassianNB(NaiveBayes):

    def __init__(self):
        self._condition_prob = None
        self._prior_prob = None
        self._label = None

    # calculate mean(mu) and standard deviation(sigma) of the given feature
    def _calculate_mean_std(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return (mu, sigma)

    # the probability density for the Gaussian distribution
    def _prob_gaussian(self, mu, sigma, x):
        return (1.0 / (sigma * np.sqrt(2 * np.pi)) *
                np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))

    def fit(self, features, labels):
        self._label = np.unique(labels)
        self._prior_prob = self._calculate_prob(labels)
        self._condition_prob = {x:{} for x in self._label}
        for label in self._label:
            for feature_mark in range(len(features[0])):
                self._condition_prob[label][feature_mark] = self._calculate_mean_std(features[:feature_mark][labels == label])

    def predict(self, new_feature):
        prediction = self._label[0]
        max_prob = 0
        for label in self._label:
            prior_prob = self._prior_prob[label]
            condition_prob = 1
            for feature_mark in range(len(new_feature)):
                condition_prob *= self._prob_gaussian(self._condition_prob[label][feature_mark], new_feature[feature_mark])
            if prior_prob * condition_prob > max_prob:
                prediction = label
                max_prob = prior_prob * condition_prob
        return prediction


#TODO: improve TestNaiveBayes()
class TestNaiveBayes():

    def test_calculate_prob(self, nb):
        a = np.array([1,2,3,4,1,3,1,3,1,2])
        print(nb._calculate_prob(a))

    def test_fit(self):
        features = np.array([[1,'s'],[1,'m'],[1,'m'],[1,'s'],[1,'s'],[2,'s'],[2,'m'],[2,'m'],[2,'l'],[2,'l'],\
                             [3,'l'],[3,'m'],[3,'m'],[3,'l'],[3,'l']])
        labels = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
        self.fit(features, labels)
        print(self._prior_prob)
        print(self._condition_prob)
        feature = np.array(['2','s'])

    def test_prediction(self):
        self._label = np.array([0, 1])
        self._prior_prob = {0:0.5, 1:0.5}
        self._condition_prob = {0:{0:{1:0.3,2:0.7}, 1:{3:0.4,4:0.6}},1:{0:{1:0.6,2:0.4}, 1:{3:0.7,4:0.3}}}
        print(self.predict(np.array([1,3])))

    def testmain(self):
        nb = NaiveBayes()
        self.test_calculate_prob(nb)
        # nb.test_prediction()
        # nb.test_fit()
