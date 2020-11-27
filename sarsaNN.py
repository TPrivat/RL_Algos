import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_gomoku
import matplotlib.pyplot as plt

device = "cuda:0"
torch.set_anomaly_enabled(True)


class SarsaNN(nn.Module):
    def __init__(self, outputs):
        super(SarsaNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1),
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.LeakyReLU(),
            nn.Linear(128, outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def Policy(action):
    if np.random.random() < 0.2:
        return np.random.randint(0, 82)
    return action


game = 'Gomoku9x9_5-v0'
num_epochs = 10000
n_actions = 82
learning_rate = 0.001

env = gym.make(game)
env.reset()
agent = SarsaNN(n_actions)
# agent.to(device)
agent.cuda()
gamma = 0.8

done = False
win = 0; lose = 0
rewards = list()
avg_rewards = list()

# optimizer = optim.RMSprop(agent.parameters())
optimizer = optim.SGD(agent.parameters(), lr=learning_rate)
# optimizer = optim.Adam(agent.parameters(), lr=0.02)
# loss = nn.SmoothL1Loss()
loss = nn.MSELoss()

for i in range(num_epochs):
    if (i+1) % 100 == 0:
        print("Epoch: ", i+1)
        print("Progress: ", (i/num_epochs) * 100, "%")
        print()
        rewards = np.array(rewards)
        avg_rewards.append(np.sum(rewards)/rewards.size)
        rewards = list()

    terminate = False
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0)
    state = state.type(torch.float32).to(device)
    q_values = agent(state)
    action = torch.argmax(q_values)
    action = Policy(action)

    while not terminate:
        next_state, reward, terminate, _ = env.step(action)
        if terminate:
            wrapped_reward = torch.tensor(reward, dtype=torch.float32).cuda()
            q1 = agent(state)[0, action]
            error = loss(wrapped_reward, q1)
            # error *= torch.tensor(-1.0)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            # Update the weights
            for f in agent.parameters():
                f.data -= (f.grad.data * learning_rate)

            if reward > 0:
                win += 1
            elif reward < 0:
                lose += 1
            rewards.append(error)
            break

        next_state = torch.from_numpy(next_state).unsqueeze(0)
        next_state = next_state.type(torch.float32).cuda()
        next_q_values = agent(next_state)
        next_action = torch.argmax(next_q_values)
        next_action = Policy(next_action)
        q1 = agent(state)[0, action]

        target = reward + gamma * next_q_values[0, next_action]
        error = loss(target, q1)
        # error *= torch.tensor(-1.0)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

        # Update the weights
        for f in agent.parameters():
            f.data -= (f.grad.data * learning_rate)

        state = next_state
        action = next_action



print()
print("Win rate: ", win/num_epochs*100, "%")
print("Loss rate: ", lose/num_epochs*100, "%")
plt.plot(np.arange(len(avg_rewards)), avg_rewards)
plt.show()