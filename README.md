# RL 101

A hands-on introduction to Reinforcement Learning using Python, Gymnasium, and PyTorch.

## Contents

### [pytorch_intro.py](pytorch_intro.py)
A minimal PyTorch intro — trains a single linear layer with SGD to fit a linear function (`y = 5x + 2`). Demonstrates the core training loop: forward pass, loss, backward, optimizer step.

### [gymnasium_qtables_intro.py](gymnasium_qtables_intro.py)
Implements a custom **FrozenLake** environment (4x4 grid, deterministic) from scratch using the Gymnasium API, then solves it with a **Q-Table** and epsilon-greedy exploration.

- Grid: 16 states, 4 actions (Left/Up/Right/Down)
- Holes at positions 5, 7, 11, 12 — goal at position 15
- Rewards: +10 (goal), -10 (hole), -1 (step)
- Uses Bellman update with decaying epsilon

### [dqn_intro.py](dqn_intro.py)
Replaces the Q-Table with a **Deep Q-Network (DQN)** built in PyTorch. States are one-hot encoded and fed through a 3-layer MLP to predict Q-values.

- Same custom FrozenLake environment
- Network: `Linear(16→10) → ReLU → Linear(10→10) → ReLU → Linear(10→4)`
- Trained with MSE loss and Adam optimizer
- Epsilon-greedy exploration with decay

### [ddqn_intro.py](ddqn_intro.py)
Extends DQN to **Double DQN (DDQN)** by decoupling action selection from action evaluation using two networks.

- Same custom FrozenLake environment and network architecture as DQN
- **Policy network** selects the best next action; **target network** evaluates its Q-value — reduces overestimation bias
- Target network updated via periodic hard copy (`hard_update`) every 100 steps
- Epsilon-greedy exploration with decay

### [ddqn_experience_replay_intro.py](ddqn_experience_replay_intro.py)
Adds **Experience Replay** to DDQN for more stable and sample-efficient training.

- Stores transitions `(state, action, reward, new_state, done)` in a replay buffer (`deque` with max 10,000 entries)
- At each step, samples a random mini-batch of 32 transitions — breaks temporal correlations and reuses past experience
- DDQN target computed via `gather` on batched tensors for efficiency
- Target network hard-updated every 100 gradient steps (not environment steps)

## Requirements

```
gymnasium
numpy
torch
```

Install with:

```bash
pip install gymnasium numpy torch
```

## Running

```bash
python pytorch_intro.py
python gymnasium_qtables_intro.py
python dqn_intro.py
python ddqn_intro.py
python ddqn_experience_replay_intro.py
```
