from sum21_env import Sum21
from agents import QLearning
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000000

env = Sum21()
agent = QLearning(2, epsilon=0.9, alpha=0.6, gamma=0.9)
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
    count = 0
    while not terminate:
        action = agent.TakeAction(state)
        if count == 0:
            prev_action = action
            count += 1
        terminate, reward = env.step(action)
        if terminate:
            if env.get_state()[0] == 21:
                prev_state = state
                state = env.get_state()

            agent.Update(prev_state, prev_action, state, reward)
            if reward > 0:
                win += 1
            elif reward < 0:
                loss += 1

            rewards.append(reward)
            break
        next_state = env.get_state()
        agent.Update(state, action, next_state, reward)
        prev_state = state
        prev_action = action
        state = next_state

    if (i + 1) % 20000 == 0:
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards) / rewards.size)
        rewards = list()

print()
print("Win rate: ", win / num_epochs)

plt.title("Avg rewards")
plt.plot(avg_rewards)
plt.show()

avg_rewards = np.array(avg_rewards)
np.save("QLearning_avg_rewards", avg_rewards)

hitstay = np.argmax(agent.q, axis=2)

print(hitstay)
print()
print(agent.q[20])
np.save("qq.npy", hitstay)

plt.imshow(hitstay, cmap='bwr')
plt.title("Q Learning Optimal Policy")
plt.colorbar(ticks=range(2))
plt.ylabel("Player hand")
plt.xlabel("Dealer hand")
plt.yticks(np.arange(0, hitstay.shape[0]))
plt.xticks(np.arange(0, hitstay.shape[1]))
plt.grid()
plt.show()