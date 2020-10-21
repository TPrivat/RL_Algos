import numpy as np


class MC:
    def __init__(self, n_action, type="FV"):
        self.n_actions = n_action
        self.visit = type
        self.v = dict()
        self.q = dict()
        self.returns = dict()
        self.eps = 0.9

    def RandomPolicy(self):
        return np.random.randint(0, self.n_actions-1)

    def LearnedPolicy(self, state, epoch):
        eps = self.eps * ((1000 - epoch) / 1000)
        if np.random.random_sample() < eps:
            return np.random.randint(0, self.n_actions-1)
        return np.random.randint(0, self.n_actions-1)  # Needs to be argmax(Q)

    def TakeAction(self, state, epoch):
        # if the state hasn't been visited before create a new
        # entry so we can keep track of returns ans state value
        if tuple(state) not in self.returns.keys():  # update returns dict
            self.returns.update({tuple(state): []})

        if tuple(state) not in self.v.keys():  # update state value dict with initial v(s) = 0
            self.v.update({tuple(state): 0})

        if epoch == 0:
            action = self.RandomPolicy()
        action = self.LearnedPolicy(state, epoch)

        return action

    def Update(self, episode):
        g = 0
        for idx, str in enumerate(episode[::-1]):
            g = 0.8 * g + str[2]
            if str[0] in episode[:idx-1]:
                continue
            else:
                self.returns[str[0]].append(g)
                self.v[str[0]] += np.average(self.returns[str[0]])
