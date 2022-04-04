import os
import numpy as np
import json

import gym

import matplotlib.pyplot as plt

from PNDDPG import Agent


def collect_trajectories(env: gym.Env, agent: Agent, n_games: int = 10) -> np.ndarray:

    for _ in range(n_games):
        state = env.reset()
        done: bool = False
        state_history: list[np.ndarray] = []

        while not done:
            state_history.append(state)
            action = agent.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state

    return np.vstack(state_history)


if __name__ == "__main__":

    # Init. path
    data_path = os.path.abspath('DDPG-PNE-PER/data')

    # Init. Environment and agent
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    agent = Agent(env=env, training=False)
    agent.load_models(data_path)

    with open(os.path.join(data_path, 'training_info.json')) as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'testing_info.json')) as f:
        test_data = json.load(f)

    # Load all the data frames
    score = [data["Epidosic Summed Rewards"] for data in train_data]
    average = [data["Moving Mean of Episodic Rewards"] for data in train_data]
    test = [data["Test Score"] for data in test_data]

    trajectory = collect_trajectories(env, agent)

    # Generate graphs
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    axes[0].plot(score, alpha=0.5, label='Episodic summation')
    axes[0].plot(average, label='Moving mean of 100 episodes')
    axes[0].grid(True)
    axes[0].set_xlabel('Training Episodes')
    axes[0].set_ylabel('Rewards')
    axes[0].legend(loc='best')
    axes[0].set_title('Training Profile')

    axes[1].boxplot(test)
    axes[1].grid(True)
    axes[1].set_xlabel('Test Run')
    axes[1].set_title('Testing Profile')

    axes[2].plot(trajectory[:, 0], trajectory[:, 1], label='Position')
    axes[2].plot(trajectory[:, 2], trajectory[:, 3], label='Velocity')
    axes[2].grid(True)
    axes[2].legend(loc='best')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    axes[2].set_title("Trajectory Plot")

    fig.tight_layout()
    plt.savefig(os.path.join(data_path, 'PNDDPG Agent Profiling.png'))
