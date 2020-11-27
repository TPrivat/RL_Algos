from sum21_env import Sum21
from agents import MC
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000000

env = Sum21()
agent = MC(env.n_actions(), num_epochs)
avg_rewards = list()
returns = list()
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
    rewards = list()

    while not terminate:
        state = env.get_state()
        action = agent.TakeAction(state, i)
        terminate, reward = env.step(action)
        states.append([state, action])
        rewards.append(reward)

    if sum(rewards) > 0:
        win += 1
    elif sum(rewards) < 0:
        loss += 1

    returns.append(sum(rewards))
    if i % 10000 == 0 and i != 0:
        # print(winlose)
        avg_rewards.append(sum(returns)/10000)
        returns = list()

    agent.updateQ(states, rewards)



hitstay = np.zeros((21, 10))
print(hitstay.shape)

hitstay = np.argmax(agent.q, axis=2)

plt.imshow(hitstay)
plt.colorbar()
plt.ylabel("Player hand")
plt.xlabel("Dealer hand")
plt.yticks(np.arange(0, hitstay.shape[0]))
plt.xticks(np.arange(0, hitstay.shape[1]))
plt.grid()
plt.show()

print("Win rate: ", win/num_epochs * 100)
print("Loss rate: ", loss/num_epochs * 100)
plt.plot(avg_rewards)
plt.show()