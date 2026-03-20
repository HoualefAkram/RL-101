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
```
