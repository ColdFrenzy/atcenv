import numpy as np
import gym

env = gym.make('UAVEnv-v0')
agents = env.agents
ITERATIONS_PER_EPISODE = 30
EPISODES = 10000
N_UAVS = 2

Q_LEARNING = False
SARSA = True
SARSA_lambda = False

SHOW_EVERY = 50
LEARNING_RATE = 0.01  # alpha
DISCOUNT = 0.99  # gamma
EPSILON = 0.01
EPSILON_DECR = False
EPSILON_DECREMENT = 0.999
EPSILON_MIN = 0.01
EPSILON_MIN2 = 0.1
max_value_for_Rmax = 100

best_reward_episode = [[] for uav in range(N_UAVS)]
uavs_episode_rewards = [[] for uav in range(N_UAVS)]
uavs_q_tables = None
if uavs_q_tables is None:

    print("Q-TABLES INITIALIZATION . . .")
    uavs_q_tables = [None for uav in range(N_UAVS)]
    explored_states_q_tables = [None for uav in range(N_UAVS)]
    uavs_e_tables = [None for uav in range(N_UAVS)]


def choose_action(uavs_q_tables, which_uav, obs, agent):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Return the chosen action according to the current set of action which depends on the considered scenario, battery level and other parameters:   #
    # SIDE EFFECT on different agents attributes are performed.                                                                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    obs = tuple([round(ob, 1) for ob in obs])
    all_actions_values = [values for values in uavs_q_tables[which_uav][obs]]
    current_actions_set = agent._action_set
    rand = np.random.random()
    return action


print("\nSTART TRAINING . . .\n")
for episode in range(1, EPISODES + 1):
    q_values_current_episode = [0 for uav in range(N_UAVS)]
    uavs_episode_reward = [0 for uav in range(N_UAVS)]
    for i in range(ITERATIONS_PER_EPISODE):
        for UAV in range(N_UAVS):
            obs = ()
            action = choose_action()
            obs_, reward, done, info = env.step_agent(agents[UAV], action)
            if (Q_LEARNING == True):

                max_future_q = np.max(uavs_q_tables[UAV][obs_])
                current_q = uavs_q_tables[UAV][obs][action]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            else:

                if (SARSA == True):
                    action_ = choose_action(uavs_q_tables, UAV, obs_, agents[UAV])
                    future_reward = uavs_q_tables[UAV][obs_][action_]
                    current_q = uavs_q_tables[UAV][obs][action]
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_reward)

                if (SARSA_lambda == True):
                    """Implements conf.SARSA-Lambda action value function update.
                            e(s,a) = lambda*gamma*e(s,a) + 1(s_t = s, a_t = a).
                            delta_t = reward_t + gamma*q(s_t+1, a_t+1) - q(s_t, a_t).
                            q(s,a) = q(s,a) + alpha*e(s,a)*delta_t.
                            Here we assume gamma=1 (undiscounted).
                            alpha = 1 / N(s,a)."""
                    lambda_value = 0.9
                    action_ = choose_action(uavs_q_tables, UAV, obs_, agents[UAV], battery_in_CS_history[UAV],
                                            env.cells_matrix)
                    future_reward = uavs_q_tables[UAV][obs_][action_]

                    current_q = uavs_q_tables[UAV][obs][action]
                    current_e = uavs_e_tables[UAV][obs][action]
                    # -------------------------PROVA1----------------------------------------------------------------------------
                    '''delta = reward + DISCOUNT * future_reward - current_q
                    print("uavs_e_tables[UAV][obs][action]", uavs_e_tables[UAV][obs][action])
                    current_e += 1
                    print("uavs_e_tables[UAV][obs][action]", uavs_e_tables[UAV][obs][action])

                    new_q = current_q + (1 - LEARNING_RATE) * delta * current_e
                    current_e = DISCOUNT * lambda_value * current_e'''
                    # -------------------------PROVA2----------------------------------------------------------------------------
                    # Computing the error
                    delta = reward + DISCOUNT * future_reward - current_q
                    # Setting the eligibility traces
                    uavs_e_tables[UAV][obs] = [i * DISCOUNT * lambda_value for i in uavs_e_tables[UAV][obs]]
                    uavs_e_tables[UAV][obs][action] = 1
                    # Updating the Q values
                    q_tabl = [i * (1 - LEARNING_RATE) * delta for i in uavs_e_tables[UAV][obs]]
                    uavs_q_tables[UAV][obs] = [x
                                               + y for x, y in zip(uavs_q_tables[UAV][obs], q_tabl)]

                elif (SARSA_lambda == True) and (SARSA == True) or (SARSA_lambda == True) and (
                        Q_LEARNING == True) or (SARSA == True) and (Q_LEARNING == True):

                    assert False, "Invalid algorithm selection."

            if SARSA_lambda == True:
                q_values_current_episode[UAV] = uavs_q_tables[UAV][obs][action]
                uavs_q_tables[UAV][obs][action] = uavs_q_tables[UAV][obs][action]
            else:
                q_values_current_episode[UAV] = new_q
                uavs_q_tables[UAV][obs][action] = new_q

            uavs_episode_reward[UAV] += reward

        if done:
            break