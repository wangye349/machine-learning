import numpy as np
import copy

class HMM(object):
    def __init__(self):
        self.A = np.array([[0.5, 0.2, 0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
        self.B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
        self.Q = np.array([1,2,3])
        self.V = np.array(['a', 'b'])
        self.PI = np.array([0.2,0.4,0.4])
        # self.Q = Q
        # self.V = V
        # self.A = A
        # self.B = B
        # self.PI = PI
        self.prob_evidence = None

    def forward(self, O):
        '''
        given an observed array O, return P(O|A,B,PI)
        '''
        observed_length = len(O)
        exist_state = len(self.Q)
        alpha = self.PI * self.B[:,list(self.V).index(O[0])]
        for t in range(1, observed_length):
            temp_alpha = copy.copy(alpha)
            for i in range(len(alpha)):
                alpha[i] = np.sum(temp_alpha * self.A[:,i]) * self.B[i,list(self.V).index(O[t])]
            # alpha = np.sum(alpha * self.A) * self.B[:,list(self.V).index(O[t])]
        self.prob_evidence = np.sum(alpha)
        print(self.prob_evidence)

    def test_forward(self):
        O = np.array(['a', 'b', 'a'])
        self.forward(O)

    def viterbi(self,O):
        observed_length = len(O)
        exist_state = len(self.Q)
        delta = np.zeros((observed_length, exist_state))
        delta[0] = self.PI * self.B[:, list(self.V).index(O[0])]
        phi = np.zeros((observed_length, exist_state))
        for t in range(1,observed_length):
            for i in range(exist_state):
                max_prob = np.max(delta[t-1] * self.A[:,i])
                delta[t][i] = max_prob * self.B[i,list(self.V).index(O[t])]
                phi[t][i] = np.argmax(delta[t-1] * self.A[:,i])
        max_prob = np.max(delta[observed_length - 1])
        optim_route = list(O)
        optim_route[-1] = np.argmax(delta[-1])
        for i in reversed(range(observed_length - 1)):
            optim_route[i] = int(phi[i+1][optim_route[i+1]])
        print(optim_route)

    def test_viterbi(self):
        O = np.array(['a', 'b', 'a', 'b'])
        self.viterbi(O)


if __name__ == "__main__":
    hmm = HMM()
    # hmm.test_forward()
    hmm.test_viterbi()
