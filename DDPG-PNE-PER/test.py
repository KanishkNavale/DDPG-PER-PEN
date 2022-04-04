from typing import Dict, List
import os
import copy
import json

import numpy as np
import gym
from tqdm import tqdm

from PNDDPG import Agent


if __name__ == '__main__':

    # Init. Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    # Init. Datapath
    data_path = os.path.abspath('DDPG-PNE-PER/data')

    # Init. Testing
    n_games = 10
    test_data: List[Dict[str, np.ndarray]] = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games, training=False)
    agent.load_models(data_path)

    for i in tqdm(range(n_games), desc=f'Testing', total=n_games):
        score_history: List[np.float32] = [] * n_games

        for _ in tqdm(range(n_games), desc=f'Testing', total=n_games):
            score = 0
            done = False

            # Initial Reset of Environment
            state = env.reset()

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                agent.memory.add(state, action, reward, next_state, done)

                state = copy.deepcopy(next_state)
                score += reward

            score_history.append(score)

        print(f'Test Analysis:\n'
              f'Mean:{np.mean(score_history)}\n'
              f'Variance:{np.std(score_history)}')

        test_data.append({'Test Score': score_history})

    # Dump .json
    with open(os.path.join(data_path, 'testing_info.json'), 'w', encoding='utf8') as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
