import copy
import numpy as np
import os
from rl_agents.DDPG import Agent
import gym

if __name__ == '__main__':

    # Init. Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    # Init. Datapath
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

    # Init. Training
    best_score = -np.inf
    score_history = []
    avg_history = []
    n_games = 2000

    # Init. Agent
    agent = Agent(env, data_path, n_games)

    for i in range(n_games):
        score = 0
        distance = 0
        done = False

        # Initial Reset of Environment
        state = env.reset()

        while not done:
            # Render
            # env.render()

            # Choose agent based action & make a transition
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.memory.add(state, action, reward, next_state, done)

            state = copy.deepcopy(next_state)
            score += reward

            # Optimize the agent
            agent.optimize()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}'
                  f'\t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}')

        # Save the score log
        np.save(data_path + 'score_history', score_history, allow_pickle=False)
        np.save(data_path + 'avg_history', avg_history, allow_pickle=False)

    # Close render
    env.close()
