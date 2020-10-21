import numpy as np
from sum21_env import Sum21

n_states = 10

for i in range(21):
    n_states += i + 1

n_actions = 2
epochs = 100
env = Sum21()


def random_policy(n_actions):
    return np.random.randint(0, n_actions)


v = dict()
returns = dict()
memory = list()

for i in range(epochs):
    terminate = False
    env.reset()

    # Create each episode
    while not terminate:
        state = env.get_state()
        action = random_policy(2)
        terminate, reward = env.step(action)
        tup = (tuple(state), action, reward)
        memory.append(tup)

        # if the state hasn't been visited before create an new entry for V(s)
        if tup not in v:
            v.update({tup:0})

        # if the state hasn't been visited before create a new
        # entry so we can keep track of returns
        if tuple(state) not in returns.keys():
            returns.update({tuple(state):[]})

    g = 0
    past_states = list()
    for str in memory:
        g = 0.8*g + str[2]                          # Update the return
        if str[0] in past_states:                   # Do not update if this isn't
            continue                                # the first visit to the state

        returns[(str[0])].append(g)
        v[str] += np.average(returns[(str[0])])

print(v)
