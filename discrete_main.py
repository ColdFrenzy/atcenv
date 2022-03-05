"""
Example
"""

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

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()

        # set done status to false
        done = False

        # execute one episode
        while not done:
            # for i in range(50):
            # the action space of a single agent is a tuple (int,int) where in is
            # a number between {0,1,2,}
            random_actions = []
            for i in range(env.num_flights):
                random_actions.append(
                    (random.randint(0, 2), random.randint(0, 2)))

            # perform step with dummy action
            rew, obs, done, info = env.step(random_actions)
            # print(obs[4])
            env.render()
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
