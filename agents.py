import numpy as np
import torch.nn as nn


class SarsaDict:
    def __init__(self, n_actions, epsilon=0.10, alpha=0.001, gamma=0.9):
        self.q = dict()
        self.n_actions = n_actions
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def Policy(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q[state])

    def TakeAction(self, state):
        # key = hash(state.flatten().tostring())
        key = hash(state)
        if key not in self.q.keys():
            self.q[key] = np.zeros(self.n_actions)

            return np.random.randint(0, self.n_actions)
        return self.Policy(key)

    def Update(self, state, action, next_state, next_action, reward, terminate):
        # state = hash(state.flatten().tostring())
        # next_state = hash(next_state.flatten().tostring())
        state = hash(state)
        next_state = hash(next_state )
        if terminate:
            try:
                self.q[state][action] += self.alpha * (reward - self.q[state][action])
            except:
                print("Current state not a key apparently, guess i'll add it")
                print("Reward: ", reward)
                print()
                self.q[state] = np.zeros(self.n_actions)
        else:
            self.q[state][action] += self.alpha * (reward + self.gamma *
                                                                self.q[next_state][next_action] -
                                                                self.q[state][action])

    def SetQ(self, dict):
        self.q = dict


class SarsaNN(nn.Module):
    def __init__(self, outputs):
        super(SarsaNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.Conv2d(16, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Linear(7*7*32, 512),
            nn.LeakyReLU(),
            nn.Linear(512, outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Sarsa:
    def __init__(self, n_actions, epsilon=0.10, alpha=0.001, gamma=0.9):
        # self.q = np.zeros((21, 10, 2)) #+ np.random.random()
        self.q = np.random.randn(21, 10, 2)
        self.n_actions = n_actions
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def Policy(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q[state[0]-1, state[1]-1, :])

    def TakeAction(self, state):
        return self.Policy(state)

    def Update(self, state, action, next_state, next_action, reward, terminate):
        if terminate:
            self.q[state[0]-1, state[1]-1, action] += self.alpha * (reward -
                                                                    self.q[state[0]-1, state[1]-1, action])
        else:
            self.q[state[0]-1, state[1]-1, action] += self.alpha * (reward + self.gamma *
                                                                self.q[next_state[0]-1, next_state[1]-1, next_action] -
                                                                self.q[state[0]-1, state[1]-1, action])


class QLearning:
    def __init__(self, n_actions, epsilon=0.15, alpha=0.1, gamma=0.8):
        self.q = np.zeros((21, 10, 2)) #+ np.random.random()
        self.n_actions = n_actions
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.count = 1

    def Policy(self, state):
        self.count += 1
        eps = self.eps * ((1000000 - self.count)/1000000)
        if np.random.random() < eps:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q[state[0]-1, state[1]-1, :])

    def TakeAction(self, state):
        return self.Policy(state)

    def Update(self, state, action, next_state, reward):
        self.q[state[0]-1, state[1]-1, action] += self.alpha * (reward + self.gamma *
                                                                np.argmax(self.q[next_state[0]-1, next_state[1]-1, :]) -
                                                                self.q[state[0]-1, state[1]-1, action])


class TDZero:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, policy=None):
        self.alpha = alpha
        self.gamma = gamma
        if policy is None:
            self.policy = lambda x: np.random.randint(0, n_actions)
        else:
            self.policy = policy
        self.v = np.zeros((21, 10))


    def TakeAction(self, state):
        return self.policy(state)

    def Update(self, state, next_state, reward):
        tderror = reward + self.gamma * self.v[next_state[0]-1, next_state[1]-1] - self.v[state[0]-1, state[1]-1]
        tderror *= self.alpha
        # print(tderror)
        self.v[state[0]-1, state[1]-1] += tderror


class SarsaLinearApprox:
    def __init__(self, n_actions, alpha=0.00000001, gamma=0.8 , epsilon=0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.n_actions = n_actions
        self.weights = np.random.randn(n_actions, n_actions)

    def Policy(self, state):
        q = np.matmul(self.weights.T, state)
        # print("Q ", q)
        action = np.argmax(q)
        return action

    def TakeAction(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.n_actions)
        return self.Policy(state)

    def Update(self, state, action, next_state, next_action, reward, terminal):
        if terminal:
            error = reward - np.matmul(self.weights.T, state)[action]
            error = error * np.array(state).astype(np.float32) * self.alpha
            self.weights += error
        else:
            error = (reward + self.gamma * np.matmul(self.weights.T, next_state)[next_action] -
                     np.matmul(self.weights.T, state)[action])
            error = error * np.array(state).astype(np.float32) * self.alpha
            self.weights += error


class TDLamda:
    def __init__(self, n_actions, lamda=0.75, alpha=0.1, gamma=0.9, policy=None):
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.n_actions = n_actions
        if policy is None:
            self.policy = lambda x: np.random.randint(0, n_actions)
        else:
            self.policy = policy
        self.v = np.zeros((21, 10))
        self.z = np.zeros((21, 10))
        self.count = 0

    def TakeAction(self, state):
        if np.random.random() < 0.15:
            return np.random.randint(0, self.n_actions)
        return self.policy(state)

    def Update(self, state, next_state, reward, terminal):
        # alpha = (1000000 - self.count) / 1000000
        self.z[state[0]-1, state[1]-1] += self.gamma * self.lamda * self.z[state[0]-1, state[1]-1]
        tderror = reward + self.gamma * self.v[next_state[0]-1, next_state[1]-1] - self.v[state[0]-1, state[1]-1]
        if terminal:
            self.z[state[0] - 1, state[1] - 1] = 1
        self.v[state[0]-1, state[1]-1] += self.alpha * tderror * self.z[state[0]-1, state[1]-1]

    def ResetZ(self):
        self.z = np.zeros((21, 10))



class MC:
    def __init__(self, n_action, epochs, type="FV"):
        self.n_actions = n_action
        self.visit = type
        self.q = 2 * np.random.randn(21, 10, 2)
        self.qreturns = np.zeros((21, 10, 2))
        self.qcount = np.zeros((21, 10, 2))
        self.eps = 0.999
        self.epoch = epochs

    def RandomPolicy(self):
        return np.random.randint(0, self.n_actions)

    def LearnedPolicy(self, state, epoch):
        eps = self.eps * ((self.epoch - epoch) / self.epoch)
        if np.random.random_sample() < eps:
            return self.RandomPolicy()                          # Random pick based on epsilon
        return np.argmax(self.q[state[0]-1, state[1]-1, :])     # Pick best action

    def TakeAction(self, state, epoch):
        if epoch == 0:
            action = self.RandomPolicy()
        else:
            action = self.LearnedPolicy(state, epoch)

        return action

    def updateQ(self, states, rewards):
        g = 0
        T = len(states) - 1
        for idx, st in enumerate(states[::-1]):
            g = 0.1 * g + rewards[T-idx]
            if st in states[:(T-1)-idx]:
                continue
            else:
                self.qreturns[st[0][0] - 1, st[0][1] - 1, st[1]] += g
                self.qcount[st[0][0] - 1, st[0][1] - 1, st[1]] += 1
                self.q[st[0][0]-1, st[0][1]-1, st[1]] += (self.qreturns[st[0][0]-1, st[0][1]-1, st[1]] /
                                                          self.qcount[st[0][0]-1, st[0][1]-1, st[1]])

        self.qreturns = np.zeros((21, 10, 2))
        self.qcount = np.zeros((21, 10, 2))
