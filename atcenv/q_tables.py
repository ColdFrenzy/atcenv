# It is called Q-table, but if conf.SARSA algorithm is used, it will be actually a conf.SARSA-table:
uavs_q_tables = None
uavs_e_tables = None
if uavs_q_tables is None:

    print("Q-TABLES INITIALIZATION . . .")
    uavs_q_tables = [None for uav in range(conf.N_UAVS)]
    explored_states_q_tables = [None for uav in range(conf.N_UAVS)]
    uavs_e_tables = [None for uav in range(conf.N_UAVS)]
    uav_counter = 0
    # print(uavs_q_tables)
    # breakpoint()
    for uav in range(conf.N_UAVS):
        current_uav_q_table = {}
        current_uav_explored_table = {}

        for x_agent in np.arange(0, map_width + 1, 1):  # range(map_width)

            for y_agent in np.arange(0, map_length + 1, 1):  # range(map_length)

                x_agent = round(x_agent, 1)
                y_agent = round(y_agent, 1)

                # 2D case with UNLIMITED UAVs battery autonomy:


                if (conf.PRIOR_KNOWLEDGE == True):
                    prior_rewards = compute_prior_rewards((x_agent, y_agent), best_prior_knowledge_points)
                    current_uav_q_table[(x_agent, y_agent)] = [prior_rewards[action] for action in range(n_actions)]
                elif (conf.R_MAX == True):
                    current_uav_q_table[(x_agent, y_agent)] = [max_value_for_Rmax for action in range(n_actions)]
                elif (conf.HOSP_SCENARIO == True):
                    for prior in conf.HOSP_PRIORITIES:
                        # current_uav_q_table[((x_agent, y_agent), 1, prior)] = [np.random.uniform(0, 1) for action in range(n_actions)]   #da vedere se mettere 0
                        # current_uav_q_table[((x_agent, y_agent), 0, prior)] = [np.random.uniform(0, 1) for action in range(n_actions)]  # da vedere se mettere 0
                        current_uav_q_table[((x_agent, y_agent), 1, prior)] = [0 for action in range(
                            n_actions)]  # da vedere se mettere 0
                        current_uav_q_table[((x_agent, y_agent), 0, prior)] = [0 for action in range(
                            n_actions)]  # da vedere se mettere 0
                    # current_uav_q_table[((x_agent, y_agent), 1, 0)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                    # current_uav_q_table[((x_agent, y_agent), 0, 0)] = [np.random.uniform(0, 1) for action in range(n_actions)]
                    current_uav_q_table[((x_agent, y_agent), 1, 0)] = [0 for action in range(n_actions)]
                    current_uav_q_table[((x_agent, y_agent), 0, 0)] = [0 for action in range(n_actions)]
                else:
                    current_uav_q_table[(x_agent, y_agent)] = [np.random.uniform(0, 1) for action in
                                                               range(n_actions)]

                current_uav_explored_table[(x_agent, y_agent)] = [False for action in range(n_actions)]



        uavs_q_tables[uav] = current_uav_q_table
        # uavs_e_tables[uav] = current_uav_q_table
        if conf.HOSP_SCENARIO == False:
            explored_states_q_tables[uav] = current_uav_explored_table
        print("Q-Table for Uav ", uav, " created")

    print("Q-TABLES INITIALIZATION COMPLETED.")
else:

    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
