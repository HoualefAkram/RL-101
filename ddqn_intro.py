# DDQN
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


def hard_update(target_net, policy_net):
    target_net.load_state_dict(policy_net.state_dict())


epoches = 500
lr = 0.01
epsilon = 1
decay_val = 0.985
min_epsilon = 0.05
gamma = 0.99
update_rate = 100  # 100 steps to update the target_network

policy_network = QNetwork()
target_network = QNetwork()

hard_update(target_network, policy_network)

criterion = nn.MSELoss()
adam = optim.Adam(policy_network.parameters(), lr=lr)

counter = 0
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
            # Use Policy network to pick an action
            with torch.no_grad():
                action = torch.argmax(policy_network(input_state)).item()

        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        done = terminated or truncated

        # train the network (DDQN)
        # 1- policy takes the state and gives best action (already done above), gives new_state
        # 2- policy takes new_state and find the optimal q_val. BUT it will not use, it needs to only know what action it is
        # 3- Target network takes new_state, gets the Q-Values of the 4 actions, it doesnt pick the best, it picks the index from the previous step

        # For DQN (with target network). Note: This isnt DDQN, just target net extension:
        # 1- policy takes the state and gives best action (already done above), gives new_state
        # 2- target takes the new_state and gives the 4 Q values, pick the best one

        with torch.no_grad():
            best_next_action_idx = torch.argmax(
                policy_network(_onehot_tensor(new_state))
            ).item()
            target_optimal_next_q = target_network(_onehot_tensor(new_state))
            v_target = target_optimal_next_q[0, best_next_action_idx].item()

            bellman_target = reward + gamma * v_target * (1 - int(done))

        policy_pred = (policy_network(input_state))[0, action]
        loss = criterion(policy_pred, torch.tensor(bellman_target))
        adam.zero_grad()
        loss.backward()
        adam.step()
        counter += 1
        if counter >= update_rate:
            counter = 0
            hard_update(target_network, policy_network)

    epsilon = max(min_epsilon, epsilon * decay_val)


# Test the agent
state, _ = env.reset()
action_names = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
path = []
print(f"--- Agent started at state {state} ---")

done = False

while not done:
    action = torch.argmax(policy_network(_onehot_tensor(state)))
    path.append(action_names[action.item()])
    state, reward, terminated, truncated, info = env.step(action=action)
    done = terminated or truncated

print(f"Path taken: {' -> '.join(path)}")
