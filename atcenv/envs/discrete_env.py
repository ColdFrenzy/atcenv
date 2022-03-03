"""
wrapper of the orginal env with discretized action-state space
"""
from ..env import Environment
from typing import Dict, Optional

import gym
import math
import numpy as np
from typing import Dict, List
# from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points, split


WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]


class DiscreteEnvironment(Environment):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 10,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 400,
                 max_episode_len: Optional[int] = 300,
                 min_distance: Optional[float] = 5.,
                 distance_init_buffer: Optional[float] = 5.,
                 max_agent_seen: Optional[int] = 3,
                 **kwargs):
        """
        Initialises the environment

        :param num_flights: numer of flights in the environment
        :param dt: time step (in seconds)
        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param max_episode_len: maximum episode length (in number of steps)
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param distance_init_buffer: distance factor used when initialising the enviroment to avoid flights close to conflict and close to the target
        :param kwargs: other arguments of your custom environment
        :param max_agent_seen: maximum number of closest agents to consider in the partial observation
        """

        super(DiscreteEnvironment, self).__init__(
            num_flights,
            dt,
            max_area,
            min_area,
            max_speed,
            min_speed,
            max_episode_len,
            min_distance,
            distance_init_buffer,
            max_agent_seen,
            **kwargs
        )

    def observation(self, angular_resolution=3, depth_resolution=3) -> List:
        """return discretized observation

        :return all_obs:
        """
        all_obs = super(DiscreteEnvironment, self).observation()
        discretized_state = []
        for i, obs in enumerate(all_obs):
            discretized_obs = np.zeros(
                angular_resolution*depth_resolution, dtype=np.int32)
            # if the obs are all zero just skip
            if np.count_nonzero(obs) == 0:
                discretized_state.append(discretized_obs)
                continue
            else:
                discretized_fov = self.discretize_fov(
                    i, angular_resolution, depth_resolution)
                for i in range(0, len(obs), 2):
                    if obs[i] == 0.0 and obs[i+1] == 0.0:
                        continue
                    # split the original FOV in sub-areas (we need shapely lines)
                    obstacle = Point((obs[i], obs[i+1]))
                    for k, poly in enumerate(discretized_fov):
                        if poly.contains(Point(obstacle)):
                            discretized_obs[k] += 1
                            break
                discretized_state.append(discretized_obs)

        return discretized_state

    def discretize_fov(self, flight_id: int, angular_resolution: int, depth_resolution: int) -> List[Polygon]:
        """return discretized representation of the FoV.
        returns a list of polygons whose union forms the original FoV

        """
        all_polygons = []
        original_fov = self.flights[flight_id].fov
        lines = original_fov.boundary
        depth_points = depth_resolution
        width_points = angular_resolution
        position = self.flights[flight_id].position
        # right FOV line
        a = LineString([lines.coords[0], lines.coords[1]])
        # left FOV line
        c = LineString([lines.coords[0], lines.coords[2]])
        # top FOV line
        b = LineString([lines.coords[1], lines.coords[2]])
        p1 = [a.interpolate(i*(a.length/depth_points))
              for i in range(depth_points+1)]
        p3 = [c.interpolate(i*(c.length/depth_points))
              for i in range(depth_points+1)]
        p2 = [b.interpolate(i*(b.length/width_points))
              for i in range(1, width_points)]
        new_lines = []
        inner_vertical_lines = []
        inner_horizontal_lines = []

        # compute new lines needed to decompose the fov in sub-polygons
        for i in range(1, len(p1)-1):
            inner_horizontal_lines.append(LineString((p1[i], p3[i])))
        for i in range(len(p2)):
            inner_vertical_lines.append(LineString((position, p2[i])))

        # all_vertical_points is a list of lists
        # each outer list is a vertical line (from right to left) and each
        # inner list contains the points of the vertical lines
        all_vertical_points = []
        # horizontal_lines.append(b)
        # we need to find the intersection points of the inner lines
        all_vertical_points.append(p1)
        for m, v_line in enumerate(inner_vertical_lines):
            vertical_points = []
            vertical_points.append(position)
            for h_line in inner_horizontal_lines:
                vertical_points.append(v_line.intersection(h_line))
            vertical_points.append(p2[m])
            all_vertical_points.append(vertical_points)
        all_vertical_points.append(p3)

        for i in range(len(all_vertical_points)-1):
            line1 = all_vertical_points[i]
            line2 = all_vertical_points[i+1]
            for j in range(len(line1)-1):
                polygon_points = [line1[j], line1[j+1], line2[j+1], line2[j]]
                all_polygons.append(Polygon(polygon_points))

        return all_polygons
