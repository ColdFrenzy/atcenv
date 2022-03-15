"""
Environment module
"""
from typing import Dict, Optional


import gym
import math
import numpy as np
from typing import List, Tuple
from atcenv.utils.definitions import Airspace, Flight
import atcenv.utils.units as u
from gym.envs.classic_control import rendering
from shapely.geometry import LineString, MultiPoint
from shapely.ops import nearest_points


WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]


class Environment(gym.Env):
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
                 state_space_shape: Optional[int] = 2,
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
        :param state_space_shape: dimension of the state space
        """
        self.num_flights = num_flights
        self.max_area = max_area * (u.nm ** 2)
        self.min_area = min_area * (u.nm ** 2)
        self.max_speed = max_speed * u.kt
        self.min_speed = min_speed * u.kt
        self.min_distance = min_distance * u.nm
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.max_agent_seen = max_agent_seen
        self.dt = dt

        # =============================================================================
        # ACTIONS
        # =============================================================================
        self.yaw_angles = kwargs["yaw_angles"]
        # 5 kt are ~ 10 km/h
        self.accelleration = kwargs["accelleration"]
        self.action_list = kwargs["action_list"]

        self.state_space_shape = state_space_shape
        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        self.viewer = None
        self.airspace = None
        self.flights = []  # list of flights
        self.conflicts = set()  # set of flights that are in conflict
        # set of flights that reached the target in the previous timestep
        self.prev_step_done = set()
        self.done = set()  # set of flights that reached the target
        self.i = None

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
        if len(actions) == 0:
            return None
        else:
            for i, action in enumerate(actions):
                if i in self.done:
                    continue
                actual_action = self.action_list[action]
                self.flights[i].track += math.radians(
                    self.yaw_angles[actual_action[0]])
                if self.min_speed <= self.flights[i].airspeed + self.accelleration[actual_action[1]] <= self.max_speed:
                    self.flights[i].airspeed += self.accelleration[actual_action[1]]

        # RDC: here you should implement your resolution actions
        ##########################################################
        return None
        ##########################################################

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
        """
        Returns the observation of each agent. A single agent observation is a
        np.array of dimension 2*self.max_agent_seen. It represents the distances
        (dx, dy) with the self.max_agent_seen closest flights.
        An agent should also know the distance and the angle from its destination 
        :return: observation of each agent
        """
        observations = []
        for i, flight in enumerate(self.flights):
            obs = np.zeros(self.state_space_shape, dtype=np.float32)
            if i in self.done:
                observations.append(obs)
                continue
            else:
                # angle between track and bearing
                obs[-2] = flight.drift
                # distance from the target in meters
                obs[-1] = flight.distance
            origin = flight.position
            seen_agents_indices = self.flights_in_fov(i)
            if len(seen_agents_indices) != 0:
                # if we saw less agents than the maximum number, we pick all of them
                if len(seen_agents_indices) <= self.max_agent_seen:
                    for j, seen_agent_idx in enumerate(seen_agents_indices):
                        obs[j * 2:j * 2 + 2] = self.flights[seen_agent_idx].position.x - origin.x, \
                            self.flights[seen_agent_idx].position.y - origin.y
                else:
                    # set of points of all the agents in the fov
                    seen_agents = MultiPoint(
                        [self.flights[seen_agent_idx].position for seen_agent_idx in seen_agents_indices])
                    # take the 3 closest agent
                    for j in range(self.max_agent_seen):
                        nearest_agent = nearest_points(origin, seen_agents)[1]
                        obs[j*2:j*2+2] = nearest_agent.x - \
                            origin.x, nearest_agent.y - origin.y
                        seen_agents.difference(nearest_agent)
            observations.append(obs)

        # RDC: here you should implement your observation function
        ##########################################################
        return np.array(observations)
        ##########################################################

    def flights_in_fov(self, flight_id: int) -> List:
        """
        returns all the agents id in the FoV of the given agent
        :return seen_agents: list of agent ids inside the FoV of the current agent
        """
        seen_agents = []
        flight_fov = self.flights[flight_id].fov
        for i, flight in enumerate(self.flights):
            if i == flight_id:
                continue
            else:
                if flight_fov.contains(flight.position):
                    seen_agents.append(i)

        ##########################################################
        return seen_agents
        ##########################################################

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.conflicts = set()

        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if j not in self.done:
                        distance = self.flights[i].position.distance(
                            self.flights[j].position)
                        if distance < self.min_distance:
                            self.conflicts.update((i, j))

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    self.done.add(i)

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # get current speed components
                dx, dy = f.components

                # get current position
                position = f.position

                # get new position and advance one time step
                f.position._set_coords(
                    position.x + dx * self.dt, position.y + dy * self.dt)

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

    def reset(self) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # create random flights
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance

        idx = 0

        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(
                self.airspace, self.min_speed, self.max_speed, idx, tol)

            # ensure that candidate is not in conflict
            for f in self.flights:
                if candidate.position.distance(f.position) < min_distance:
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)
                idx += 1

        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = set()
        self.done = set()

        # return initial observation
        return self.observation()

    def render(self, mode=None) -> None:
        """
        Renders the environment
        :param mode: rendering mode
        :return:
        """
        if self.viewer is None:
            # initialise viewer
            screen_width, screen_height = 600, 600

            minx, miny, maxx, maxy = self.airspace.polygon.buffer(
                10 * u.nm).bounds
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(minx, maxx, miny, maxy)

            # fill background
            background = rendering.make_polygon([(minx, miny),
                                                 (minx, maxy),
                                                 (maxx, maxy),
                                                 (maxx, miny)],
                                                filled=True)
            background.set_color(*BLACK)
            self.viewer.add_geom(background)

            # display airspace
            sector = rendering.make_polygon(
                self.airspace.polygon.boundary.coords, filled=False)
            sector.set_linewidth(1)
            sector.set_color(*WHITE)
            self.viewer.add_geom(sector)

        # add current positions
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            if i in self.conflicts:
                color = RED
            else:
                color = BLUE

            circle = rendering.make_circle(radius=self.min_distance / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                             f.position.y)))
            circle.set_color(*BLUE)

            plan = LineString([f.position, f.target])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([f.position, f.prediction])
            self.viewer.draw_polyline(
                prediction.coords, linewidth=4, color=color)

            self.viewer.add_onetime(circle)

            # add fovs
            fov_points = list(zip(*f.fov.exterior.coords.xy))[:-1]
            fov = rendering.make_polygon(fov_points, filled=True)
            # fov = rendering.make_polygon([(fov_points[0].x, fov_points[0].y),
            #                               (fov_points[1].x, fov_points[1].y),
            #                               (fov_points[2].x, fov_points[2].y),
            #                               ],
            #                              filled=True)
            fov.set_color(*YELLOW)
            self.viewer.add_onetime(fov)

        self.viewer.render()

    def close(self) -> None:
        """
        Closes the viewer
        :return:
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def action_space_dim(self):
        """return the size of the action space of a single agent
        """
        return len(self.action_list)
