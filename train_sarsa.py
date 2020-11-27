from sum21_env import Sum21
from agents import Sarsa
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000000

env = Sum21()
agent = Sarsa(2, epsilon=0.1, alpha=0.005, gamma=0.8)
avg_rewards = list()
returns = list()
win = 0
loss = 0
rewards = list()

for i in range(num_epochs):
    if (i+1) % 10000 == 0:
        print("Epoch: ", i+1)
        print("Progress: ", (i/num_epochs) * 100, "%")
        print()

    terminate = False
    env.reset()
    state = env.get_state()
    action = agent.TakeAction(state)
    prev_state, prev_action = state, action
    while not terminate:
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
        # prev_state, prev_action = state, action
        state = next_state
        action = next_action



    if (i+1) % 20000 == 0:
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards)/rewards.size)
        rewards = list()

print()
print("Win rate: ", win/num_epochs*100, "%")
hitstay = np.argmax(agent.q, axis=2)

avg_rewards = np.array(avg_rewards)
plt.plot(np.arange(avg_rewards.size), avg_rewards)
plt.show()


np.save("Sarsa_avg_rewards.npy", avg_rewards)



plt.imshow(hitstay, cmap='bwr')
plt.colorbar(ticks=range(2))
plt.title("SARSA Optimal Policy")
plt.ylabel("Player hand")
plt.xlabel("Dealer hand")
plt.yticks(np.arange(0, hitstay.shape[0]))
plt.xticks(np.arange(0, hitstay.shape[1]))
plt.grid()
plt.show()
