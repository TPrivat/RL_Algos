import gym
import gym_gomoku
import numpy as np
from agents import SarsaDict
import matplotlib.pyplot as plt
import pickle

game = 'Gomoku9x9_5-v0'
num_epochs = 10000

env = gym.make(game)
env.reset()
agent = SarsaDict(82, epsilon=0.1, alpha=0.005, gamma=0.8)


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
rewards = list()
avg_rewards = list()




for i in range(num_epochs):
    if (i+1) % 2000 == 0:
        print("Epoch: ", i+1)
        print("Progress: ", (i/num_epochs) * 100, "%")
        print()

    terminate = False
    state = env.reset()
    stateH = boardHash(state)
    action = agent.TakeAction(stateH)

    while not terminate:
        next_state, reward, terminate, _ = env.step(action)
        next_stateH = boardHash(next_state)
        if terminate:
            agent.Update(stateH, action, stateH, action, reward, terminate)
            if reward > 0:
                win += 1
            elif reward < 0:
                loss += 1
            rewards.append(reward)
            break
        next_action = agent.TakeAction(next_stateH)
        agent.Update(stateH, action, next_stateH, next_action, reward, terminate)
        state = next_state
        stateH = next_stateH
        action = next_action

    if i % 5000 == 0:
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards)/rewards.size)
        rewards = list()

print()
print("Win rate: ", win/num_epochs*100, "%")
print("Loss rate: ", loss/num_epochs*100, "%")
print("Unique state: ", len(agent.q))


plt.plot(np.arange(len(avg_rewards)), avg_rewards)
plt.show()

with open('sarsa_qdict.pkl', 'wb') as f:
    pickle.dump(agent.q, f)
