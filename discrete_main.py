"""
Example
"""

import numpy as np
import tensorflow as tf
import os


SHOW_EVERY = 50
LEARNING_RATE = 0.01  # alpha
DISCOUNT = 0.99  # gamma
EPSILON = 0.99
EPSILON_DECR = False
EPSILON_DECREMENT = 0.999
EPSILON_MIN = 0.01
EPSILON_MIN2 = 0.1
max_value_for_Rmax = 100
alpha = 0.85
gamma = 0.95


if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv.envs.discrete_env import DiscreteEnvironment
    from atcenv.algorithms.RL_algorithms import RL_algorithms
    import time
    from tqdm import tqdm
    # writer = tf.summary.create_file_writer("/tmp/mylogs")

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--parameters_sharing', type=bool, default=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(DiscreteEnvironment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = DiscreteEnvironment(**vars(args.env))

    # =============================================================================
    # ALGORITHM PARAMS
    # =============================================================================
    state_space_size = env.state_space_size
    action_space_size = env.action_space_size
    parameters_sharing = args.parameters_sharing
    num_agents = env.num_flights
    state_max_val = env.max_agent_seen
    action_max_val = max(len(env.yaw_angles), len(env.accelleration))

    alg = RL_algorithms(action_space_size, state_space_size,
                state_max_val, action_max_val, num_agents)

    # run episodes
    for e in tqdm(range(args.episodes)):
        print("episode: ", e)
        # reset environment

        state_1 = env.reset()
        action_1 = []
        for obs in state_1:
            action_1.append(env.action_list[alg.choose_action(state_1)])

        # set done status to false
        done = False

        # execute one episode
        while not done:
            print("step: ", env.i)
            env.render()

            rew, state_2, done, info = env.step(action_1)

            # Choosing the next action
            action_2 = []
            for obs in state_1:
                action_2.append(env.action_list[alg.choose_action(state_2)])

            for agent_indx in range(len(state_1)):
                alg.update(state_1[agent_indx], state_2[agent_indx],
                           rew[agent_indx], action_1[agent_indx], action_2[agent_indx])

            action_1 = action_1
            state_1 = state_2

            time.sleep(0.05)

        # close rendering
        env.close()
