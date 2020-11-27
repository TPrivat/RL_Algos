from sum21_env import Sum21
from agents import TDZero
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000000

env = Sum21()
agent = TDZero(2)
avg_rewards = list()
rewards = list()
win = 0
loss = 0

for i in range(num_epochs):
    if (i+1) % 10000 == 0:
        print("Epoch: ", i+1)
        print("Progress: ", (i/num_epochs) * 100, "%")
        print()

    terminate = False
    env.reset()
    states = list()

    state = env.get_state()
    prev_state = state

    while not terminate:
        prev_state = state
        action = agent.TakeAction(state)
        terminate, reward = env.step(action)
        if terminate:
            agent.Update(prev_state, state, reward)
            if reward > 0:
                win += 1
            elif reward < 0:
                loss += 1

            rewards.append(reward)
            break

        next_state = env.get_state()
        agent.Update(state, next_state, reward)
        state = next_state

    if (i+1)%20000 == 0:
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards)/rewards.size)
        rewards = list()

print()
print("Win rate: ", win/num_epochs)

avg_rewards = np.array(avg_rewards)
np.save("TD0_avg_rewards.npy", avg_rewards)

plt.imshow(agent.v, cmap='bwr')
plt.colorbar()
plt.title("TD(0) Optimal State Values")
plt.ylabel("Player hand")
plt.xlabel("Dealer hand")
plt.yticks(np.arange(0, agent.v.shape[0]))
plt.xticks(np.arange(0, agent.v.shape[1]))
plt.grid()
plt.show()