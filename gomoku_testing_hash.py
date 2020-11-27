import numpy as np
import pickle
from agents import SarsaDict
import gym
import gym_gomoku
import matplotlib.pyplot as plt

game = 'Gomoku9x9_5-v0'
num_epochs = 1000

env = gym.make(game)
env.reset()
env.opponent = 'naive3'


agent = SarsaDict(82, epsilon=0.1, alpha=0.005, gamma=0.8)

with open('sarsa_qdict.pkl', 'rb') as f:
    q_table = pickle.load(f)

agent.SetQ(q_table)


def boardHash(board):
    color = board[0]
    full = board[1]
    for i in range(9):
        for j in range(9):
            if color[i, j] == 1:
                full[i, j] = -1

    full = full.flatten()
    h = 0.0
    for i in range(full.size):
        h *= 100.3
        h += full[i]

    return h


done = False
win = 0
loss = 0

for i in range(num_epochs):
    if (i+1) % 2000 == 0:
        print("Epoch: ", i+1)
        print("Progress: ", (i/num_epochs) * 100, "%")
        print()

    terminate = False
    reward = 0
    state = env.reset()
    stateH = boardHash(state)
    action = agent.TakeAction(stateH)

    while not terminate:
        state, reward, terminate, _ = env.step(action)
        stateH = boardHash(state)
        action = agent.TakeAction(stateH)

    if reward > 0:
        win += 1
    elif reward < 0:
        loss += 1

print()
print("Win rate: ", win/num_epochs*100, "%")
print("Loss rate: ", loss/num_epochs*100, "%")