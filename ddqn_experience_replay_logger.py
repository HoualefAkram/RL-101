# DDQN + Experience replay

from gymnasium_qtables_intro import FrozenLake
from torch import nn
import random
import torch
import numpy as np
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
    return F.one_hot(torch.tensor([state_num]), num_classes=16).float()


def hard_update(target_net, policy_net):
    target_net.load_state_dict(policy_net.state_dict())


epoches = 500
lr = 0.01
epsilon = 1
decay_val = 0.985
min_epsilon = 0.05
gamma = 0.99
update_rate = 100  # 100 steps to update the target_network
batch_size = 32
transitions = deque(
    maxlen=10000
)  # state, action, reward, new_state, done.... when reaching 10000 items, old items will get removed

policy_network = QNetwork()
target_network = QNetwork()

hard_update(target_network, policy_network)

criterion = nn.MSELoss()
adam = optim.Adam(policy_network.parameters(), lr=lr)

logger = SummaryWriter("runs/FrozenLake_DDQN")

counter = 0
for epoche in range(epoches):
    done = False
    state, _ = env.reset()

    # Logging
    ep_total_reward = 0  # reward
    ep_steps = 0  # episode length
    ep_total_loss = 0  # loss
    ep_loss_counter = 0
    ep_total_max_q = 0  # max Q

    while not done:
        input_state = _onehot_tensor(state)

        with torch.no_grad():
            ep_total_max_q += torch.max(policy_network(input_state)).item()
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Use Policy network to pick an action
            with torch.no_grad():
                action = torch.argmax(policy_network(input_state)).item()

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay = (state, action, reward, new_state, done)  # tuple
        transitions.append(replay)
        state = new_state

        ep_total_reward += reward
        ep_steps += 1

        # train the network (DDQN)
        # 1- policy takes the state and gives best action (already done above), gives new_state
        # 2- policy takes new_state and finds the optimal q_val. BUT it will not use it, it needs to only know what action it is.
        # 3- Target network takes new_state, gets the Q-Values of the 4 actions, it doesnt pick the best one, it picks the index from the previous step

        # For DQN (with target network). Note: This isnt DDQN, just target net extension:
        # 1- policy takes the state and gives best action (already done above), gives new_state
        # 2- target takes the new_state and gives the 4 Q values, pick the best one

        if len(transitions) >= batch_size:
            batch = random.sample(transitions, batch_size)
            b_states, b_actions, b_rewards, b_new_states, b_dones = zip(*batch)

            b_states_t = torch.tensor(b_states, dtype=torch.int64)
            b_new_states_t = torch.tensor(b_new_states, dtype=torch.int64)

            b_states_tensor = F.one_hot(b_states_t, num_classes=16).float()
            b_new_states_tensor = F.one_hot(b_new_states_t, num_classes=16).float()

            with torch.no_grad():
                # 32 best action IDs
                best_next_action_idxs = torch.argmax(
                    policy_network(b_new_states_tensor), dim=1, keepdim=True
                )
                # 32 optimal next Q values from the target network
                target_optimal_next_qs = target_network(b_new_states_tensor)

                # 32 V-Targets (First idea)
                # v_targets = [
                #     target_optimal_next_qs[i, best_next_action_idxs[i]].item()
                #     for i in range(batch_size)
                # ]
                # 32 V-Targets (optimized)
                v_targets = target_optimal_next_qs.gather(1, best_next_action_idxs)
                reward_t = torch.tensor(b_rewards, dtype=torch.float32).unsqueeze(1)
                done_t = torch.tensor(b_dones, dtype=torch.float32).unsqueeze(1)
                bellman_targets = reward_t + gamma * v_targets * (1 - done_t)

            b_actions_t = torch.tensor(b_actions, dtype=torch.int64).unsqueeze(1)

            policy_preds = (policy_network(b_states_tensor)).gather(1, b_actions_t)

            loss = criterion(policy_preds, bellman_targets)
            ep_total_loss += loss.item()
            ep_loss_counter += 1

            adam.zero_grad()
            loss.backward()
            adam.step()

            counter += 1
            if counter >= update_rate:
                counter = 0
                hard_update(target_network, policy_network)

    # Y,X
    logger.add_scalar("Performance/Episode_Length", ep_steps, epoche)
    logger.add_scalar("Performance/Total_Reward", ep_total_reward, epoche)
    logger.add_scalar("Performance/Average_Max_Q", ep_total_max_q / ep_steps, epoche)
    logger.add_scalar("Training/Epsilon", epsilon, epoche)

    # initial [batch_size] iterations, ep_loss_counter = 0
    if ep_loss_counter > 0:
        logger.add_scalar(
            "Performance/Average_Loss", ep_total_loss / ep_loss_counter, epoche
        )

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
