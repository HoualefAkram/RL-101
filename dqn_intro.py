# DQN
from gymnasium_qtables_intro import FrozenLake
from torch import nn
import random
import torch
import numpy as np
import torch.optim as optim

env = FrozenLake()


class QNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(16, 10),  # input is the number of states (one hot encoded)
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 4),  # output is the number of actions
        )

    def forward(self, x):
        return self.net(x)


def _onehot_tensor(state_num: int):
    onehot = np.zeros(16)
    onehot[state_num] = 1.0
    return torch.tensor(onehot, dtype=torch.float32).unsqueeze(0)


epoches = 500
lr = 0.01
epsilon = 1
decay_val = 0.985
min_epsilon = 0.05
gamma = 0.99
network = QNetwork()

criterion = nn.MSELoss()
adam = optim.Adam(network.parameters(), lr=lr)

for epoche in range(epoches):
    done = False
    state, _ = env.reset()
    print(f"Epoche {epoche} / {epoches}. init state: {state}")
    while not done:
        input_state = _onehot_tensor(state)
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(network(input_state)).item()

        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        done = terminated or truncated

        # train the network
        pred = network(input_state)[0, action]
        new_input_state = _onehot_tensor(new_state)
        next_optimal_q = torch.max(network(new_input_state)).item()
        bellman_target = reward + gamma * next_optimal_q * (1 - int(done))
        loss = criterion(pred, torch.tensor(bellman_target))
        adam.zero_grad()
        loss.backward()
        adam.step()

    epsilon = max(min_epsilon, epsilon * decay_val)


# Test the agent
state, _ = env.reset()
action_names = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
path = []
print(f"--- Agent started at state {state} ---")

done = False

while not done:
    action = torch.argmax(network(_onehot_tensor(state)))
    path.append(action_names[action.item()])
    state, reward, terminated, truncated, info = env.step(action=action)
    done = terminated or truncated

print(f"Path taken: {' -> '.join(path)}")
