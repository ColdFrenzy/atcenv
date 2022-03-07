"""
wrapper of the orginal env with discretized action-state space
"""
from ..env import Environment
from typing import Dict, Optional, List, Tuple

import gym
import math
import numpy as np
# from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points, split
import itertools as it

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
                 angular_resolution: Optional[int] = 2,
                 depth_resolution: Optional[int] = 2,
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
        :param max_agent_seen: maximum number of closest agents to consider in the partial observation
        :param angular_resolution: number of elements into which to divide the FoV opening
        :param depth_resolution:number of elements into which to divide the FoV depth
        :param kwargs: other arguments of your custom environment
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
        self.angular_resolution = angular_resolution
        self.depth_resolution = depth_resolution
        # set of flights that reached the target in the previous timestep
        self.prev_step_done = set()
        # define the action space
        self.yaw_angles = [-5.0, 0.0, 5.0]
        # 5 kt are ~ 10 km/h
        self.accelleration = [-5.0, 0.0, 5.0]
        self.action_list = list(it.product(
            range(len(self.yaw_angles)), range(len(self.accelleration))))
        self.actions = [self.yaw_angles, self.accelleration]
        self.num_actions = len(self.yaw_angles) + len(self.accelleration)
        # (env.max_agent_seen + 1) ^ (angular_resolution * depth_resolution)

    def resolution(self, actions: List) -> None:
        """
        Applies the resolution actions
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed].
        A single action is a tuple of 2 integer in {0,1,2} the first integer
        represents the angular variation in self.yaw_angle and the second
        represents the linear velocity variation in self.accellaration
        :param action: list of resolution actions assigned to each flight
        :return:
        """
        for i, action in enumerate(actions):
            if i in self.done:
                continue
            self.flights[i].track += math.radians(self.yaw_angles[action[0]])
            if self.min_speed <= self.flights[i].airspeed + self.accelleration[action[1]] <= self.max_speed:
                self.flights[i].airspeed += self.accelleration[action[1]]

        # RDC: here you should implement your resolution actions
        ##########################################################
        return None
        ##########################################################

    def step(self, action: List) -> Tuple[List, List, bool, Dict]:
        """
        Performs a simulation step

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """
        # apply resolution actions
        self.resolution(action)

        # update positions
        self.update_positions()

        # update done set
        self.update_done()

        # update conflict set
        self.update_conflicts()

        # compute reward
        rew = self.reward()

        # compute observation
        obs = self.observation()

        # increase steps counter
        self.i += 1

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done = (self.i == self.max_episode_len) or (
            len(self.done) == self.num_flights)

        return rew, obs, done, {}

    def reward(self) -> List:
        """
        Returns the reward assigned to each agent. At the moment agents receive
        +1 if they reach the target and -1 if there is a conflict
        :return: reward assigned to each agent
        """
        total_rewards = []
        for i, flight in enumerate(self.flights):
            if i in self.done:
                total_rewards.append(0.0)
                continue
            # TODO: use an external parameter, not -1 and +1
            conflict_reward = -1.0 if i in self.conflicts else 0.0
            target_reached_reward = + \
                1.0 if (i in self.done and i not in self.prev_step_done) else 0.0
            total_rewards.append(conflict_reward+target_reached_reward)
        # update the prev_step_done
        self.prev_step_done.update(self.done)

        # RDC: here you should implement your reward function
        ##########################################################
        return total_rewards
        ##########################################################

    def observation(self) -> List:
        """return discretized observation
        a single discrete observation is a np.array of dimension
        angular_resolution*depth_resolution and each element of the array may
        have a value between [0, self.max_agent_seen]. Therefore, the state
        space has dimension:
            (self.max_agent_seen+1)^(angular_resolution*depth_resolution)
        :return all_obs:
        """
        all_obs = super(DiscreteEnvironment, self).observation()
        discretized_state = []
        for i, obs in enumerate(all_obs):
            discretized_obs = np.zeros(
                self.angular_resolution*self.depth_resolution, dtype=np.int32)
            # if the obs are all zero just skip
            if np.count_nonzero(obs) == 0:
                discretized_state.append(discretized_obs)
                continue
            else:
                discretized_fov = self.discretize_fov(
                    i, self.angular_resolution, self.depth_resolution)
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

    @property
    def state_space_size(self):
        """return the size of the state space
        """
        # TODO: there are configuration that will never be seen, we can reduce
        # the state space these states will be automatically ignored in the
        # Q-table (for example if we have a FOV with total resolution of 4 and max_agent_seen=3
        # we will never see a state like [3,3,3,3])
        return (self.max_agent_seen +
                1) ** (self.angular_resolution * self.depth_resolution)

    @property
    def action_space_size(self):
        """return the size of the action space
        """
        return len(max(self.yaw_angles, self.accelleration))**2
