# Gymnasium 101 (Q-Tables)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class FrozenLake(gym.Env):
    __TOP = [0, 1, 2, 3]
    __LEFT = [0, 4, 8, 12]
    __RIGHT = [3, 7, 11, 15]
    __BOTTOM = [12, 13, 14, 15]
    __HOLES = [5, 7, 11, 12]

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(16)  # 4x4 grid
        self.action_space = spaces.Discrete(4)  # 0: left, 1: up, 2 right, 3 down
        self.agent_position = None
        self.current_step = None
        self.max_steps = 100

    def __move(self, action) -> int:
        # Illegal moves
        if (
            (action == 0 and self.agent_position in FrozenLake.__LEFT)
            or (action == 1 and self.agent_position in FrozenLake.__TOP)
            or (action == 2 and self.agent_position in FrozenLake.__RIGHT)
            or (action == 3 and self.agent_position in FrozenLake.__BOTTOM)
        ):
            return self.agent_position
        # Legal Moves
        match (action):
            case 0:
                return self.agent_position - 1
            case 1:
                return self.agent_position - 4
            case 2:
                return self.agent_position + 1
            case 3:
                return self.agent_position + 4

    def step(self, action):
        self.current_step += 1

        self.agent_position = self.__move(action)
        terminated = (self.agent_position == 15) or (
            self.agent_position in FrozenLake.__HOLES
        )
        truncated = self.current_step >= self.max_steps

        reward = 10.0 if terminated else -1.0
        if self.agent_position in FrozenLake.__HOLES:
            reward = -10.0
        info = {}

        return self.agent_position, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agent_position = random.choice(
            [x for x in range(15) if x not in FrozenLake.__HOLES]
        )
        self.current_step = 0
        info = {}

        return self.agent_position, info


if __name__ == "__main__":
    env = FrozenLake()

    epoches = 1000
    epsilon = 1.0
    epsilon_decay = 0.985
    min_epsilon = 0.05
    lr = 0.01
    gamma = 0.9

    qvals = np.zeros((16, 4))

    for epoche in range(epoches):
        state, _ = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                # random action
                action = env.action_space.sample()
            else:
                # best action
                action = np.argmax(qvals[state])

            new_state, reward, terminated, truncated, info = env.step(action)

            next_best_q = np.max(qvals[new_state])
            qvals[state][action] = qvals[state][action] + lr * (
                reward + gamma * next_best_q - qvals[state][action]
            )

            state = new_state
            done = terminated or truncated
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("\n--- Training Complete ---")
    print("Testing the trained agent...")

    state, _ = env.reset()
    done = False
    path = []

    print(f"Initial State: {state}")

    action_names = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}

    while not done:
        action = np.argmax(qvals[state])
        path.append(action_names[action])

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    print(f"Path taken: {' -> '.join(path)}")

    if state == 15:
        print("Result: Agent successfully reached the goal!")
    elif state in [5, 7, 11, 12]:
        print("Result: Agent fell in a hole.")
    else:
        print("Result: Agent ran out of time.")

    print("\nFinal Q-Table:")
    print(np.round(qvals, 2))
