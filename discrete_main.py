"""
Example
"""

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

import numpy as np
if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv.envs.discrete_env import DiscreteEnvironment
    import time
    from tqdm import tqdm

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(DiscreteEnvironment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = DiscreteEnvironment(**vars(args.env))

    #init q_tables
    uavs_q_tables = env.q_table()


    def choose_action_(state):
        action = 0
        if np.random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(uavs_q_tables[state, :])
        return action

    def choose_action(state):
        random_actions = []
        for i in range(env.num_flights):
            if np.random.uniform(0, 1) < EPSILON:
                random_actions.append(
                    (random.randint(0, 2), random.randint(0, 2)))
            else:
                random_actions.append(
                    (np.argmax(uavs_q_tables[i][state, :]), np.argmax(uavs_q_tables[i][state, :])))
        return random_actions


    # Function to learn the Q-value
    def update(state, state2, reward, action, action2):
        for i in range(env.num_flights):

            print(len(uavs_q_tables))
            print(state[i], "state")
            print(action[i], "action")
            print("aoooooooooooooooooooo", uavs_q_tables[i][state[i]])
            print(state)


            state_i = tuple(state[i])
            state_i_2 = tuple(state2[i])

            print("state_i", state_i)
            print("action[i]", action[i])
            print("env.d_states", env.d_states)
            print("\n\nenv.d_states", env.d_actions)
            print("\n\nenv.d_actions[action[i]]", env.d_actions[action[i]])
            print("\n\nenv.d_states[state_i]", env.d_states[state_i])
            print("env.d_actions[action[i]][0]", action[i][0])
            #predict = uavs_q_tables[i][env.d_states[state_i], env.d_actions[action[i]]]
            predict = uavs_q_tables[i][env.d_states[state_i], action[i][0]]
            print("predict", predict)
            target = reward[i] + gamma * uavs_q_tables[i][env.d_states[state_i_2], action2[i][0]]
            #target = reward[i] + gamma * uavs_q_tables[i][env.d_states[state_i_2], env.d_actions[action2[i]]]
            print("target", target)
            uavs_q_tables[i][state_i, action[i][0]] = uavs_q_tables[i][state_i, action[i][0]] + alpha * (target - predict)

            print(uavs_q_tables[i][state_i, action[i][0]])
            print(uavs_q_tables[0])
            #breakpoint()
            #predict = uavs_q_tables[i][state, action]
            #target = reward[i] + gamma * uavs_q_tables[i][state2, action2]
            #uavs_q_tables[i][state, action] = uavs_q_tables[i][state, action] + alpha * (target - predict)

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment

        #obs = env.reset()
        state1 = env.reset()
        action1 = choose_action(state1)
        print("action1", action1)
        print("state1", state1)



        # set done status to false
        done = False

        # execute one episode
        while not done:
            # for i in range(50):
            # the action space of a single agent is a tuple (int,int) where in is
            # a number between {0,1,2,}
            # random_actions = []
            '''for i in range(env.num_flights):
                random_actions.append(
                (random.randint(0, 2), random.randint(0, 2)))'''
            print("action1", action1)

            env.render()

            # perform step with dummy action
            rew, state1, done, info = env.step(action1)
            print("action1", action1)
            print("rew", rew)
            print("obs", state1)
            print("done", done)
            print("info", info)
            #breakpoint()
            state2 = state1
            # Choosing the next action
            action2 = choose_action(state2)
            print("action2", action2)

            # Learning the Q-value
            update(state1, state2, rew, action1, action2)

            state1 = state2
            action1 = action2


            # print(obs[4])
            max_future_q = []
            current_q = []
            new_q = []
            '''for UAV in range(env.num_flights):
                max_future_q.append(np.max(uavs_q_tables[UAV][state2]))
                current_q.append(uavs_q_tables[UAV][state1][action1])
                new_q.append((1 - LEARNING_RATE) * current_q[UAV] + LEARNING_RATE * (rew[UAV] + DISCOUNT * max_future_q[UAV]))

            for UAV in range(env.num_flights):
                max_future_q.append(np.max(uavs_q_tables[UAV][obs]))
                current_q.append(uavs_q_tables[UAV][obs][random_actions])
                new_q.append((1 - LEARNING_RATE) * current_q + LEARNING_RATE * (rew + DISCOUNT * max_future_q))
                
'''

            time.sleep(0.05)

        # close rendering
        env.close()

    # from shapely.geometry import LineString, MultiPoint, Polygon
    # from shapely.ops import linemerge, unary_union, polygonize, polygonize_full
    # original_fov = env.flights[3].fov
    # lines = original_fov.boundary
    # depth_points = 3
    # width_points = 3
    # position = env.flights[3].position
    # # right FOV line
    # a = LineString([lines.coords[0], lines.coords[1]])
    # # left FOV line
    # c = LineString([lines.coords[0], lines.coords[2]])
    # # top FOV line
    # b = LineString([lines.coords[1], lines.coords[2]])
    # p1 = [a.interpolate(i*(a.length/depth_points))
    #       for i in range(depth_points+1)]
    # p3 = [c.interpolate(i*(c.length/depth_points))
    #       for i in range(depth_points+1)]
    # p2 = [b.interpolate(i*(b.length/width_points))
    #       for i in range(1, width_points)]
    # new_lines = []
    # inner_vertical_lines = []
    # inner_horizontal_lines = []
    # # vertical_lines.append(a)
    # # vertical_lines.append(c)

    # # compute new lines needed to decompose the fov in sub-polygons
    # for i in range(1, len(p1)-1):
    #     inner_horizontal_lines.append(LineString((p1[i], p3[i])))
    # for i in range(len(p2)):
    #     inner_vertical_lines.append(LineString((position, p2[i])))

    # # all_vertical_points is a list of lists
    # # each outer list is a vertical line (from right to left) and each
    # # inner list contains the points of the vertical lines
    # all_vertical_points = []
    # # horizontal_lines.append(b)
    # # we need to find the intersection points of the inner lines
    # all_vertical_points.append(p1)
    # for m, v_line in enumerate(inner_vertical_lines):
    #     vertical_points = []
    #     vertical_points.append(position)
    #     for h_line in inner_horizontal_lines:
    #         vertical_points.append(v_line.intersection(h_line))
    #     vertical_points.append(p2[m])
    #     all_vertical_points.append(vertical_points)
    # all_vertical_points.append(p3)

    # import matplotlib.pyplot as plt

    # # yo = LineString(all_vertical_points[3])
    # for line in all_vertical_points:
    #     yo = LineString(line)
    #     print(i)
    #     # if i == 6:
    #     x, y = yo.xy
    #     plt.plot(x, y)

    # # now build the polygons: (a polygon need 4 points)
    # all_polygons = []
    # for i in range(len(all_vertical_points)-1):
    #     line1 = all_vertical_points[i]
    #     line2 = all_vertical_points[i+1]
    #     for j in range(len(line1)-1):
    #         polygon_points = [line1[j], line1[j+1], line2[j+1], line2[j]]
    #         all_polygons.append(Polygon(polygon_points))

    # colors = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    #           (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0),
    #           (1.0, 0.0, 1.0), (0.5, 0.5, 0.5), (0.3, 0.9, 0.1)]
    # for i, yo in enumerate(all_polygons):
    #     # if i == 6:
    #     x, y = yo.exterior.xy
    #     plt.plot(x, y)

    # # x,y = all_polygons[5].exterior.xy
    # # plt.plot(x,y)

    # # =============================================================================
    # # DA QUI IN POI NON SERVE
    # # =============================================================================
    # new_lines = [*horizontal_lines, *vertical_lines]
    # brolygons = []
    # j = 0
    # for i in range(len(horizontal_lines)-1):
    #     # at first we just take one horizontal line for the small triangle
    #     if i == 0:
    #         all_lines = [*vertical_lines, *horizontal_lines[i:i+1]]
    #         merged_lines = linemerge(all_lines)
    #         borders = unary_union(merged_lines)
    #         # result, dangles, cuts, invalids = polygonize_full([])
    #         # brolygons.append(result)
    #         brolygons.append(polygonize(borders))
    #         # bro = polygonize([*vertical_lines,*horizontal_lines[i:i+1]])
    #         for yo in bro:
    #             print(j)
    #             x, y = yo.exterior.xy
    #             plt.plot(x, y, c=colors[j])
    #             j += 1
    #     all_lines = [*vertical_lines, *horizontal_lines[i:i+2]]
    #     merged_lines = linemerge(all_lines)
    #     borders = unary_union(merged_lines)
    #     # result, dangles, cuts, invalids = polygonize_full([*vertical_lines,*horizontal_lines[i:i+2]])
    #     # brolygons.append(result)
    #     brolygons.append(polygonize(borders))
    #     # bro = polygonize([*vertical_lines,*horizontal_lines[i:i+2]])
    #     # bro = [*vertical_lines,*horizontal_lines[2:3]]

    #     for yo in bro:
    #         print(j)
    #         x, y = yo.exterior.xy
    #         plt.plot(x, y, c=colors[j])
    #         j += 1

    # colors = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    #           (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
    # i = 0
    # for bro in brolygons:
    #     for yo in brolygons[2]:
    #         print(i)
    #         x, y = yo.exterior.xy
    #         plt.plot(x, y, c=colors[i])
    #         i += 1
    # merged_lines = linemerge(new_lines)
    # borders = unary_union(merged_lines)

    # borders = unary_union(new_lines)
    # # yo = list(merged_lines)
    # # x, y = yo[4].xy
    # # plt.plot(x, y)
    # for yo in new_lines:
    #     x, y = yo.xy
    #     plt.plot(x, y)

    # # result, dangles, cuts, invalids = polygonize_full(new_lines)
    # import random
    # # black ,red, green, blue,yellow
    # colors = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    #           (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
    # polygon = polygonize(borders)

    # import matplotlib.pyplot as plt
    # for i, p in enumerate(polygon):
    #     # r = random.random()
    #     # b = random.random()
    #     # g = random.random()
    #     # color = (r, g, b)
    #     print(i)
    #     # if i == 6:
    #     x, y = p.exterior.xy
    #     plt.plot(x, y, c=colors[i])

    # for line in new_lines:
    #     x, y = line.xy
    #     plt.plot(x, y)
