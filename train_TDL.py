from sum21_env import Sum21
from agents import TDLamda
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000000

# qp = np.load("qq.npy")
# policy = lambda x: np.argmax(qp[x[0]-1, x[1]-1])

env = Sum21()
agent = TDLamda(2, lamda=0.4, alpha=0.6, gamma=0.8)
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
    agent.ResetZ()
    states = list()

    state = env.get_state()
    prev_state = state

    while not terminate:
        prev_state = state
        action = agent.TakeAction(state)
        terminate, reward = env.step(action)
        if terminate:
            agent.Update(prev_state, state, reward, terminate)
            if reward > 0:
                win += 1
            elif reward < 0:
                loss += 1

            rewards.append(reward)
            break

        next_state = env.get_state()
        agent.Update(state, next_state, reward, terminate)

        state = next_state

    if (i+1)%20000 == 0:
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards)/rewards.size)
        rewards = list()

print()
print("Win rate: ", win/num_epochs)

plt.plot(avg_rewards)
plt.show()

avg_rewards = np.array(avg_rewards)
np.save("TDL_avg_rewards.npy", avg_rewards)

plt.imshow(agent.v, cmap='bwr')
plt.colorbar()
plt.title("TD(lambda) Optimal State Value")
plt.ylabel("Player hand")
plt.xlabel("Dealer hand")
plt.yticks(np.arange(0, agent.v.shape[0]))
plt.xticks(np.arange(0, agent.v.shape[1]))
plt.grid()
plt.show()

# policy = agent.v.copy()
#
# for i in range(agent.v.shape[0]):
#     for j in range(agent.v.shape[1]):
#         if policy[i, j] >= -1:
#             policy[i, j] = 1
#         else:
#             policy[i, j] = 0
#
# plt.imshow(policy)
# plt.colorbar(ticks=range(2))
# plt.title("TD(lambda) Optimal State Value")
# plt.ylabel("Player hand")
# plt.xlabel("Dealer hand")
# plt.yticks(np.arange(0, agent.v.shape[0]))
# plt.xticks(np.arange(0, agent.v.shape[1]))
# plt.grid()
# plt.show()