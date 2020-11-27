from sum21_env import Sum21
from agents import SarsaLinearApprox
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000000

env = Sum21()
agent = SarsaLinearApprox(2, alpha=0.000005, gamma=0.7, epsilon=0.1)
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
    action = agent.TakeAction(state)
    # print(action)
    prev_state, prev_action = state, action
    while not terminate:
        # prev_state, prev_action = state, action
        terminate, reward = env.step(action)
        if terminate:
            agent.Update(state, action, state, action, reward, terminate)
            if reward > 0:
                win += 1
            elif reward < 0:
                loss += 1

            rewards.append(reward)
            break
        next_state = env.get_state()
        next_action = agent.TakeAction(next_state)
        agent.Update(state, action, next_state, next_action, reward, terminate)
        prev_state, prev_action = state, action
        state = next_state
        action = next_action

    if (i+1)%20000 == 0:
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards)/rewards.size)
        rewards = list()

print()
print("Win rate: ", win/num_epochs)

plt.plot(avg_rewards)
plt.show()

np.array(avg_rewards)
np.save("Approx_avg_rewards.npy", avg_rewards)

hitstay = np.zeros((21, 10))
for i in range(21):
    for j in range(10):
        q = np.argmax(np.matmul(agent.weights.T, np.array([i, j])))
        hitstay[i, j] = q

if np.sum(hitstay) == 0:
    print(agent.weights)

np.save("approx_q.npy", hitstay)

plt.imshow(hitstay, cmap='bwr')
plt.colorbar(ticks=range(2))
plt.title("TD Linear Approximation Optimal Policy")
plt.ylabel("Player hand")
plt.xlabel("Dealer hand")
plt.yticks(np.arange(0, hitstay.shape[0]))
plt.xticks(np.arange(0, hitstay.shape[1]))
plt.grid()
plt.show()
