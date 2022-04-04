from typing import Dict, List
import os
import json

import numpy as np
import gym

from torch.utils.tensorboard import SummaryWriter

from DDPG import Agent

# Init. tensorboard summary writer
tb = SummaryWriter(log_dir=os.path.abspath('Vanilla-DDPG/data/tensorboard'))


if __name__ == '__main__':

    # Init. Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    # Init. Datapath
    data_path = os.path.abspath('Vanilla-DDPG/data')

    # Init. Training
    n_games: int = 1500
    best_score = -np.inf
    score_history: List[float] = [] * n_games
    avg_history: List[float] = [] * n_games
    logging_info: List[Dict[str, float]] = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games)

    for i in range(n_games):
        score: float = 0.0
        done: bool = False

        # Initial Reset of Environment
        state = env.reset()

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)

            state = next_state
            score += reward

            agent.optimize()

        score_history.append(score)
        avg_score: float = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(data_path)
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}'
                  f'\t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}')

        episode_info = {
            'Episode': i,
            'Total Episodes': n_games,
            'Epidosic Summed Rewards': score,
            'Moving Mean of Episodic Rewards': avg_score
        }

        logging_info.append(episode_info)

        # Add info. to tensorboard
        tb.add_scalars('training_rewards',
                       {'Epidosic Summed Rewards': score,
                        'Moving Mean of Episodic Rewards': avg_score}, i)

        # Dump .json
        with open(os.path.join(data_path, 'training_info.json'), 'w', encoding='utf8') as file:
            json.dump(logging_info, file, indent=4, ensure_ascii=False)

    # Close tensorboard writer
    tb.close()
