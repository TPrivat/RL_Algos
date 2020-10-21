from sum21_env import Sum21
from agents import MC
import numpy as np
import matplotlib.pyplot as plt

env = Sum21()
agent = MC(env.n_actions())

num_epochs = 3000000

for i in range(num_epochs):
    if ((i+1) % 100000 == 0):
        print("Epoch: ", i+1)

    terminate = False
    env.reset()
    episode = list()

    while not terminate:
        state = env.get_state()
        action = agent.TakeAction(state, i)
        terminate, reward = env.step(action)
        episode.append((tuple(state), action, reward))

    # print(reward)

    agent.UpdateQ(episode)

print(agent.q)



hit = 0
stay = 0
player_states = list()
dealer_states = list()
value_states = list()
for i in agent.q:
    player_states.append(i[0])
    dealer_states.append(i[1])
    value = agent.q[i]
    if np.argmax(value) == 0:
        hit+=1
    else:
        stay+=1
    value_states.append(np.argmax(value))
print("Hit: ", hit)
print("Stay: ", stay)
player_states = np.array(player_states)
dealer_states = np.array(dealer_states)
value_states = np.array(value_states)

save = np.vstack((player_states, dealer_states))
save = np.vstack((save, value_states))

print(save.shape)

np.save('round3_sameas1_longer.npy', save)