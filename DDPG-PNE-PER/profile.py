import os
import numpy as np
import json

import torch

from sklearn.decomposition import PCA

import gym

import matplotlib.pyplot as plt

from PNDDPG import Agent


def predict_value(agent: Agent, state: np.ndarray) -> float:
    with torch.no_grad():
        state = torch.as_tensor(state, dtype=torch.float32, device=agent.actor.device)
        action = torch.as_tensor(agent.choose_action(state), dtype=torch.float32, device=agent.actor.device)
        value = agent.critic.forward(state, action)
    return value.item()


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

    # Process network data
    initial_state = 1e6 * np.ones(env.observation_space.shape[0])
    final_state = -1e6 * np.ones(env.observation_space.shape[0])
    steps: int = 500
    states = np.linspace(initial_state, final_state, steps)

    # Compress the states to 2D
    state = np.vstack(states)
    pca = PCA(n_components=1)
    compressed_states = pca.fit_transform(state)
    assert np.allclose(pca.explained_variance_ratio_[0], 1.0)

    # Fetch values
    values = [predict_value(agent, pca.inverse_transform(state)) for state in compressed_states]

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

    axes[2].plot(compressed_states, values)
    axes[2].grid(True)
    axes[2].set_xlabel('State: Principal Axes-1 (VAR = 1.0)')
    axes[2].set_ylabel('State-Action Values')
    axes[2].set_title("Critic Value Estimation")

    fig.tight_layout()
    plt.savefig(os.path.join(data_path, 'PNDDPG Agent Profiling.png'))
