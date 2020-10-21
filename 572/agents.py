import numpy as np


class MC:
    def __init__(self, n_action, type="FV"):
        self.n_actions = n_action
        self.visit = type
        self.v = dict()
        self.q = dict()
        self.vreturns = dict()
        self.qreturns = dict()
        self.eps = 0.9

    def RandomPolicy(self):
        return np.random.randint(0, self.n_actions)

    def LearnedPolicy(self, state, epoch):
        eps = self.eps * ((1000 - epoch) / 1000)
        if np.random.random_sample() < eps:
            return self.RandomPolicy()                      # Random pick based on epsilon
        if tuple(state) in self.q.keys():
            return np.argmax(self.q[tuple(state)])          # Pick best action
        else:
            return self.RandomPolicy()                      # If new state pick random action

    def TakeAction(self, state, epoch):
        # if the state hasn't been visited before create a new
        # entry so we can keep track of returns and state value
        if tuple(state) not in self.vreturns.keys():  # update returns dict
            self.vreturns.update({tuple(state): []})

        if tuple(state) not in self.v.keys():  # update state value dict with initial v(s) = 0
            self.v.update({tuple(state): 0})

        if tuple(state) not in self.q.keys():
            self.q.update({tuple(state): [0, 0]})

        # if tuple(state) not in self.qreturns.keys():
        #     self.qreturns.update({tuple(state): [0, 0]})

        # action = self.RandomPolicy()
        if epoch == 0:
            action = self.RandomPolicy()
        else:
            action = self.LearnedPolicy(state, epoch)

        # If this state, action pair has not been visit create entry in q to
        # # keep track of value
        # if (tuple(state), action) not in self.q.keys():
        #     self.q.update({(tuple(state), action): 0})
        #
        if (tuple(state), action) not in self.qreturns.keys():
            self.qreturns.update({(tuple(state), action): []})

        return action

    def UpdateV(self, episode):
        g = 0
        for idx, str in enumerate(episode[::-1]):
            g = 0.8 * g + str[2]
            if str[0] in episode[:idx-1]:
                continue
            else:
                self.vreturns[str[0]].append(g)
                self.v[str[0]] += np.average(self.vreturns[str[0]])

    def UpdateQ(self, episode):
        g = 0
        T = len(episode) - 1
        for idx, str in enumerate(episode[::-1]):
            g = 0.8 * g + str[2]
            # g += str[2]
            if str in episode[:(T-1)-idx]:
                continue
            else:
                self.qreturns[(str[0], str[1])].append(g)
                self.q[str[0]][str[1]] += np.average(self.qreturns[(str[0], str[1])])


class Sarsa:
    def __init__(self, n_actions, epsilon=0.8, alpha=0.75, gamma=0.75):
        self.q = dict()
        self.n_actions = n_actions
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def Policy(self, state):
        if tuple(state) not in self.q.keys():
            return np.random.randint(0, self.n_actions)     # Random based on epsilon
        if np.random.random_sample() > self.eps:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q[tuple(state)])

    def TakeAction(self, state):
        # Check if we visited state yet
        # if not then add it to q table
        if tuple(state) not in self.q.keys():
            self.q.update({tuple(state): [0, 0]})

        return self.Policy(state)

    def UpdateQ(self, state, action, next_state, next_action, reward):
        self.q[tuple(state)][action] += self.alpha * (reward + self.gamma*self.q[tuple(next_state)][next_action]
                                                      - self.q[tuple(state)][action])


class MC2:
    def __init__(self, n_action, type="FV"):
        self.n_actions = n_action
        self.visit = type
        self.v = dict()
        self.q = dict()
        self.vreturns = dict()
        self.qreturns = dict()
        self.eps = 0.9

    def RandomPolicy(self):
        return np.random.randint(0, self.n_actions)

    def LearnedPolicy(self, state, epoch):
        eps = self.eps * ((1000 - epoch) / 1000)
        if np.random.random_sample() < eps:
            return self.RandomPolicy()                      # Random pick based on epsilon
        if tuple(state) in self.q.keys():
            return np.argmax(self.q[tuple(state)])          # Pick best action
        else:
            return self.RandomPolicy()                      # If new state pick random action

    def TakeAction(self, state, epoch):
        # if the state hasn't been visited before create a new
        # entry so we can keep track of returns and state value
        if tuple(state) not in self.vreturns.keys():  # update returns dict
            self.vreturns.update({tuple(state): []})

        if tuple(state) not in self.v.keys():  # update state value dict with initial v(s) = 0
            self.v.update({tuple(state): 0})

        if tuple(state) not in self.q.keys():
            self.q.update({tuple(state): [0, 0]})

        # if tuple(state) not in self.qreturns.keys():
        #     self.qreturns.update({tuple(state): [0, 0]})

        # action = self.RandomPolicy()
        if epoch == 0:
            action = self.RandomPolicy()
        else:
            action = self.LearnedPolicy(state, epoch)

        # If this state, action pair has not been visit create entry in q to
        # # keep track of value
        # if (tuple(state), action) not in self.q.keys():
        #     self.q.update({(tuple(state), action): 0})
        #
        if (tuple(state), action) not in self.qreturns.keys():
            self.qreturns.update({(tuple(state), action): []})

        return action

    def UpdateV(self, episode):
        g = 0
        for idx, str in enumerate(episode[::-1]):
            g = 0.8 * g + str[2]
            if str[0] in episode[:idx-1]:
                continue
            else:
                self.vreturns[str[0]].append(g)
                self.v[str[0]] += np.average(self.vreturns[str[0]])

    def UpdateQ(self, episode):
        g = 0
        T = len(episode) - 1
        for idx, str in enumerate(episode[::-1]):
            g = 0.8 * g + str[2]
            # g += str[2]
            if str in episode[:(T-1)-idx]:
                continue
            else:
                self.qreturns[(str[0], str[1])].append(g)
                self.q[str[0]][str[1]] += np.average(self.qreturns[(str[0], str[1])])
