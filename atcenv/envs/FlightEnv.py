"""
Environment module
"""
from copy import copy, deepcopy
from typing import DefaultDict, Dict, List

import numpy as np
import pygame
import itertools as it
from pygame import gfxdraw
from collections import defaultdict
from ray.rllib import MultiAgentEnv
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import nearest_points

from atcenv.common.wind_utils import abs_compass
from atcenv.definitions import *

WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]
ORANGE = [255, 165, 0]


def min_max_normalizer(val, min_, max_):
    return (val - min_) / (max_ - min_)


def normalizer(maxx, minx, maxy, miny, screen_w, screen_h):
    def inner(x, y, use_int=False):
        x = np.asarray(x)
        y = np.asarray(y)

        x = min_max_normalizer(x, minx, maxx) * screen_w
        y = min_max_normalizer(y, miny, maxy) * screen_h
        if use_int:
            x = np.int64((np.floor(x)))
            y = np.int64((np.floor(y)))
        return x, y

    return inner


class FlightEnv(MultiAgentEnv):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 10,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 400,
                 accelleration: Optional[List[float]] = [-5.0, 0.0, 5.0],
                 yaw_angles: Optional[List[float]] = [-5.0, 0.0, 5.0],
                 max_episode_len: Optional[int] = 300,
                 min_distance: Optional[float] = 5.,
                 distance_init_buffer: Optional[float] = 5.,
                 max_agent_seen: Optional[int] = 3,
                 wind_speed: Optional[float] = 0,
                 wind_dir: Optional[str] = 'NW3',
                 reward_as_dict: Optional[bool] = False,
                 screen_size=600,
                 stop_when_outside=True,
                 max_distance_from_target=10,
                 reward_dict:Dict={},
                 **kwargs):
        """
        Initialise the environment.

        :param num_flights: numer of flights in the environment
        :param dt: time step (in seconds)
        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param accelleration: available accelleration actions in m/s^2
        :param yaw_angles: available yaw angles actions in deg,
        :param max_episode_len: maximum episode length (in number of steps)
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param distance_init_buffer: distance factor used when initialising the enviroment to avoid flights close to conflict and close to the target
        :param kwargs: other arguments of your custom environment
        :param max_agent_seen: maximum number of closest agents to consider in the partial observation
        :param wind_speed: wind speed (in kt)
        :param wind_dir: cardinal direction of the wind 
        :param reward_as_dict: if True, the reward is returned as a dict of the individual reward components. Useful for debug
        :param screen_size: size of the screen 
        :param stop_when_outside: if True, agent is done when outside the Airspace and when its distance from target is > max_distance_from_target
        :param max_distance_from_target: only considered when stop_when_outside is True
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
        self.wind_speed = wind_speed * u.kt
        self.wind_dir = wind_dir
        self.reward_as_dict = reward_as_dict
        self.stop_when_outside = stop_when_outside
        self.max_distance_from_target = max_distance_from_target * u.nm

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        # Rendering
        self.screen = None
        self.surf = None
        self.isopen = True
        self.screen_size = screen_size

        self.airspace = None
        self.flights = {}  # list of flights
        self.conflicts = set()  # set of flights that are in conflict
        self.done = {}  # set of flights that reached the target
        self.i = None
        self.wind_components = self.wind_effect()

        # =============================================================================
        # REWARDS
        # =============================================================================

        rew_dict = dict(
            collision_weight=-0.1,
            dist_weight=0.0,  # - 1.0
            target_reached_w=+100.0,
            distance_from_optimal_trajectory_w=0.0,  # - 0.01
            drift_penalty_w=-0.1,
            changed_angle_penalty_w=0.0,
        )


        for k in rew_dict.keys():
            if k not in reward_dict.keys():
                reward_dict[k] = rew_dict[k]

        # print("#" * 20)
        # print("Params in flight env")
        # print("#" * 20)
        # print(reward_dict)
        self.reward_dict = reward_dict

        # =============================================================================
        # ACTIONS
        # =============================================================================
        self.yaw_angles = [math.radians(angle) for angle in yaw_angles]
        self.accelleration = [acc * u.kt for acc in accelleration]
        # action_space contains all combinations of angles and accelleration
        # action_space = [(-5.0, -5.0), (-5.0, 0.0), (-5.0, 5.0), ..., (5.0, 5.0)]
        self.action_list = list(it.product(
            self.yaw_angles, self.accelleration))
        # Distance between the furthest points in the airspace
        # max distance is the true max_distance between the Airspace bounding box
        self.max_distance = None
        # max screen distance is used for scaling the rendering
        self.max_screen_distance = None

    def resolution(self, actions: Dict) -> None:
        """
        Applies the resolution actions
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed]
        :param actions: dict of resolution actions assigned to each flight
        :return:
        """
        ##########################################################
        # RDC: here you should implement your resolution actions
        ##########################################################
        self.changed_angle_penalty = {k: 0 for k in self.flights.keys()}
        for f_id, action in actions.items():
            f = self.flights[f_id]
            actual_action = self.action_list[action]
            if actual_action[0] != 0.0:
                self.changed_angle_penalty[f_id] = 1.0
            f.track += actual_action[0]
            # always keep the track between 0 and 2pi
            if f.track < 0:
                f.track = (f.track + u.circle) % u.circle
            if f.track >= u.circle:
                f.track = f.track % u.circle
            # change airspeed only if the new airspeed is the [self.min_speed, self.max_speed] range
            # this is not needed since invalid accellerations is already masked out
            if self.min_speed <= f.airspeed + actual_action[1] * self.dt <= self.max_speed:
                f.airspeed += actual_action[1]
            assert self.min_speed <= f.airspeed <= self.max_speed, f"The speed value is invalid {f.airspeed}. It should be between [{self.min_speed},{self.max_speed}]"

    def reward(self) -> Dict:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        """

        ##########################################################
        # RDC: here you should implement your reward function
        ##########################################################

        def speed_penalty(f, weight):
            """
            each flight's speed should be as close as possible to optimal speed

            """
            cur_speed = f.airspeed
            optimal_speed = f.optimal_airspeed

            diff_speed = cur_speed - optimal_speed
            perc = 0
            # flight is going slower than optimal
            if diff_speed < 0:
                max_diff = optimal_speed - self.min_speed
                diff_speed = abs(diff_speed)
                perc = diff_speed / max_diff
            elif diff_speed > 0:
                max_diff = self.max_speed - optimal_speed
                perc = diff_speed / max_diff

            return perc * weight

        def target_dist(f) -> float:
            """
            Return the normalized distance between the flight and its target
            """
            dist = f.distance
            dist /= self.max_distance

            # During exploration it may happen that the Flight goes outside the Airspace
            if dist >= 1:
                dist = 1
            # assert 0 <= dist <= 1, f"Distance in not normalized, got '{dist}'"
            return dist

        def target_reached(f) -> bool:
            """
            Check if the flight has reached the target
            """

            dist = f.distance

            if dist < self.tol:
                return True
            return False

        # WEIGHTS OF THE REWARDS
        collision_weight = self.reward_dict['collision_weight']
        dist_weight = self.reward_dict['dist_weight'] # - 1.0
        target_reached_w =self.reward_dict['target_reached_w']
        distance_from_optimal_trajectory_w = self.reward_dict['distance_from_optimal_trajectory_w']# - 0.01
        drift_penalty_w = self.reward_dict['drift_penalty_w']
        changed_angle_penalty_w = self.reward_dict['changed_angle_penalty_w']

        if self.reward_as_dict:
            rews = {k: defaultdict(float) for k in self.flights.keys()}
        else:
            rews = {k: 0 for k in self.flights.keys()}

        for f_id, flight in self.flights.items():
            if self.reward_as_dict:
                rews[f_id]["distance_from_target_rew"] += target_dist(
                    flight) * dist_weight
                rews[f_id]["distance_from_traj_rew"] += flight.distance_from_optimal_trajectory * \
                                                        distance_from_optimal_trajectory_w
                rews[f_id]["angle_changed_rew"] += self.changed_angle_penalty[f_id] * \
                                                   changed_angle_penalty_w
                drift_rew = min_max_normalizer(flight.drift,
                                               0, 2 * math.pi)
                rews[f_id]["drift_rew"] += (drift_rew * drift_penalty_w) if drift_rew >= 0 else (
                        drift_rew * -drift_penalty_w)
                if target_reached(flight):
                    rews[f_id]["target_reached_rew"] += target_reached_w

            else:
                rews[f_id] += target_dist(flight) * dist_weight
                rews[f_id] += flight.distance_from_optimal_trajectory * \
                              distance_from_optimal_trajectory_w
                rews[f_id] += self.changed_angle_penalty[f_id] * \
                              changed_angle_penalty_w
                drift_rew = min_max_normalizer(flight.drift,
                                               0, 2 * math.pi)
                rews[f_id] += (drift_rew * drift_penalty_w) if drift_rew >= 0 else (
                        drift_rew * -drift_penalty_w)
                if target_reached(flight):
                    rews[f_id] += target_reached_w

        # collision penalty
        for c in self.conflicts:
            if self.reward_as_dict:
                rews[c]["collision_rew"] += collision_weight
            else:
                rews[c] += collision_weight

        return rews

    def get_mask(self, flight_id) -> Dict:
        """Return a mask which mask out the invalid actions involving the accelleration"""
        action_masked = []
        for i, elem in enumerate(self.action_list):
            if not (self.min_speed <= self.flights[flight_id].airspeed + elem[1] * self.dt <= self.max_speed):
                action_masked.append(i)
        return action_masked

    def observation(self) -> Dict:
        """
        Returns the observation of each agent. A single agent observation is a
        np.array of dimension 2*self.max_agent_seen. It represents the distances
        (dx, dy) with the self.max_agent_seen closest flights
        :return: observation of each agent
            Each agents' observation is made up of:
                - velocity : float [0,1] : normalized velocity [min_speed, max_speed]
                - bearing : float [0,1] : normalized bearing [0, 2*pi]
                - agents_in_fov : ndarray of shape (2*max_seen_agents) range [0,1] : a vector of normalized polar
                    coordinates of agents in fov, first max_seen_agents are the angles and the last the distances
                    if there are no flights in the fov, the values are set to -1
        """

        def polar_distance(f: Flight):
            """
            Computes the normalized polar distance between a flight and all the other flights in the current field of view

            @params : flight :

            @returns:
                angles:
                dist: normalized distance (1 is max fov depth)
            """
            # left_angle = np.full(self.max_agent_seen, -1., dtype=np.float32)
            # right_angle = np.full(self.max_agent_seen, -1., dtype=np.float32)
            # angles = np.full(self.max_agent_seen, -1., dtype=np.float32)
            # dists = np.full(self.max_agent_seen, -1., dtype=np.float32)

            left_angle = np.zeros(self.max_agent_seen, dtype=np.float32)
            right_angle = np.zeros(self.max_agent_seen, dtype=np.float32)
            angles = np.zeros(self.max_agent_seen, dtype=np.float32)
            dists = np.zeros(self.max_agent_seen, dtype=np.float32)

            origin = f.position
            seen_agents_indices = self.flights_in_fov(i)
            if len(seen_agents_indices) != 0:
                # if we saw less agents than the maximum number, we pick all of them
                if len(seen_agents_indices) <= self.max_agent_seen:
                    for j, seen_agent_idx in enumerate(seen_agents_indices):
                        x_dist = self.flights[seen_agent_idx].position.x - origin.x
                        y_dist = self.flights[seen_agent_idx].position.y - origin.y
                        # angle computed w.r.t. the north
                        angle = (np.arctan2(x_dist, y_dist) +
                                 u.circle) % u.circle
                        fov_start_angle = (
                                                  (f.track - f.fov_angle / 2) + u.circle) % u.circle
                        if angle < fov_start_angle:
                            angle += u.circle
                        angle = angle - fov_start_angle
                        # angle is normalized between 0 and 1, at zero we are at the beginning
                        # of the fov (clockwise) at 1 we are at the end of the fov
                        angles[j] = min_max_normalizer(angle, 0, f.fov_angle)
                        # with the following if, we map the FOV angle such that it is 0 on the extremes
                        # and 0.5 in the middle
                        if angles[j] >= 0.5:
                            right_angle[j] = 1.0 - angles[j]
                        else:
                            left_angle[j] = angles[j]
                        right_angle[j] += 1
                        left_angle[j] += 1
                        assert 0.0 <= angles[
                            j] <= 1.0, f"The FoV angle is invalid: {angles[j]}. The angle should be between [0,1]"
                        # dists[j] = math.dist([self.flights[seen_agent_idx].position.x,
                        #                       self.flights[seen_agent_idx].position.y],
                        #                      [origin.x, origin.y]) / f.fov_depth
                        dists[j] = (math.dist([self.flights[seen_agent_idx].position.x,
                                              self.flights[seen_agent_idx].position.y],
                                              [origin.x, origin.y]) / f.fov_depth) + 1.0

                else:
                    # set of points of all the agents in the fov
                    seen_agents = MultiPoint(
                        [self.flights[seen_agent_idx].position for seen_agent_idx in seen_agents_indices])
                    # take the 3 closest agent
                    for j in range(self.max_agent_seen):
                        nearest_agent = nearest_points(origin, seen_agents)[1]
                        x_dist = nearest_agent.x - origin.x
                        y_dist = nearest_agent.y - origin.y
                        # angle between flight seen and current flight computed w.r.t. the north
                        angle = (np.arctan2(x_dist, y_dist) +
                                 u.circle) % u.circle
                        # starting angle of the current flight fov w.r.t. north
                        fov_start_angle = (
                                                  (f.track - f.fov_angle / 2) + u.circle) % u.circle

                        if angle < fov_start_angle:
                            angle += u.circle
                        angle = angle - fov_start_angle
                        # angle is normalized between 0 and 1, at zero we are at the beginning
                        # of the fov (clockwise) at 1 we are at the end of the
                        angles[j] = min_max_normalizer(angle, 0, f.fov_angle)
                        if angles[j] >= 0.5:
                            right_angle[j] = 1.0 - angles[j]
                        else:
                            left_angle[j] = angles[j]

                        right_angle[j] += 1
                        left_angle[j] += 1
                        assert 0.0 <= angles[
                            j] <= 1.0, f"The FoV angle is invalid: {angles[j]}. The angle should be between [0,1]"
                        # dists[j] = math.dist([nearest_agent.x,
                        #                       nearest_agent.y],
                        #                      [origin.x, origin.y]) / f.fov_depth
                        dists[j] = (math.dist([nearest_agent.x,
                                              nearest_agent.y],
                                              [origin.x, origin.y]) / f.fov_depth) + 1.0
                        seen_agents.difference(nearest_agent)
            return left_angle, right_angle, dists

        observations = {}

        for i, flight in self.flights.items():
            observations[i] = {}
            # compute observations and normalizations
            # speed normalized between min and max speed
            v = min_max_normalizer(
                flight.airspeed, self.min_speed, self.max_speed)
            # bearing angle between [-pi,pi]
            b = min_max_normalizer(flight.bearing, 0, 2 * math.pi)
            d = min_max_normalizer(
                flight.distance, 0, self.max_distance)
            if d > 1.0:
                d = 1.0
            left_angle, right_angle, dists = polar_distance(flight)
            obs = np.concatenate([left_angle, right_angle, dists])

            assert 0 <= v <= 1, f"Airspeed is not in range [0,1]. Got '{v}'"
            assert 0 <= b <= 1, f"Bearing is not in range [0,1]. Got '{b}'"

            observations[i]['velocity'] = np.asarray([v])
            observations[i]['bearing'] = np.asarray([b])
            observations[i]['agents_in_fov'] = obs
            observations[i]['distance_from_target'] = np.asarray([d])
            observations[i]['action_mask'] = np.ones(len(self.action_list))
            observations[i]['action_mask'][self.get_mask(i)] = 0.0

        return observations

    def flights_in_fov(self, flight_id: int) -> List:
        """
        returns all the agents id in the FoV of the given agent
        :return seen_agents: list of agent ids inside the FoV of the current agent
        """
        seen_agents = []
        flight_fov = self.flights[flight_id].fov
        for i, flight in self.flights.items():
            if i == flight_id or self.done[i]:
                continue
            else:
                if flight_fov.contains(flight.position):
                    seen_agents.append(i)

        ##########################################################
        return seen_agents
        ##########################################################

    def wind_effect(self) -> Tuple[float, float]:
        """
        Wind X and Y compoents
        :return: tuple containing wind speed
        """

        assert self.wind_dir in abs_compass, 'Invalid wind direction seleted! Select a cardinal direction from' + str(
            [key for key, val in abs_compass.items()])

        # Angle beween the North and the wind direction:
        alpha = math.radians(abs_compass[self.wind_dir])

        # Wind speed coordinates w.r.t. North Wind (it corresponds also to the wind speed coordinates w.r.t. agent position)
        # These correspond also to the wind speed absolute components w.r.t. agent position (since we are assuming wind origin right in (0,0):
        wsx = self.wind_speed * math.sin(alpha)
        wsy = self.wind_speed * math.cos(alpha)

        return (wsx, wsy)

    def track_prediction(self, flight: Flight, dt: Optional[float] = 120) -> Point:
        """
        Predicts the future actual (track) position after dt seconds (wind effect included)

        :param: flight: current flight
        :param: dt: prediction look-ahead time (in seconds)
        :return: current flight actual (track) position (wind effected included)
        """
        dx, dy = flight.components
        dx += self.wind_components[0]
        dy += self.wind_components[1]
        return Point(flight.position.x + dx * dt, flight.position.y + dy * dt)

    def wind_direction(self, flight: Flight, dt: Optional[float] = 120) -> Point:
        """
        Wind direction w.r.t. the current flight

        :param: flight: current flight
        :param: dt: prediction look-ahead time (in seconds)
        :return: wind applied on the current flight
        """
        dx = self.wind_components[0]
        dy = self.wind_components[1]
        return Point(flight.position.x + dx * dt, flight.position.y + dy * dt)

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.conflicts = set()
        # use list for less comparison
        flight_list = list(self.flights.items())

        for idx in range(self.num_flights - 1):

            fi_id, fi = flight_list[idx]

            if self.done[fi_id]:
                # skip done id
                continue

            for jdx in range(idx + 1, self.num_flights):

                fj_id, fj = flight_list[jdx]

                if self.done[fj_id]:
                    # skip done id
                    continue

                distance = fi.position.distance(fj.position)
                if distance < self.min_distance:
                    self.conflicts.update((fi_id, fj_id))

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        for i, f in self.flights.items():
            if not self.done[i]:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    self.done[i] = True
                # we also stop it when the Flight is outside the Airspace and it's distant from the target
                elif self.stop_when_outside:
                    if distance > self.max_distance_from_target and not self.airspace.polygon.contains(f.position):
                        self.done[i] = True

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """
        for i, f in self.flights.items():
            if not self.done[i]:
                # get current speed components
                dx, dy = f.components
                # get the current wind components
                ws_x, ws_y = self.wind_components

                # add wind components (if any)
                dx += ws_x
                dy += ws_y

                # get current position
                position = f.position

                # get new position and advance one time step
                f.position._set_coords(
                    position.x + dx * self.dt, position.y + dy * self.dt)

    def step(self, actions: dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Performs a simulation step

        :param actions: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """
        # apply resolution actions
        self.resolution(actions)

        # update positions
        self.update_positions()

        # update conflict set
        self.update_conflicts()

        # update done set
        self.update_done()

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

        all_done = self.i == self.max_episode_len
        if not all_done:
            all_done = all([v for k, v in self.done.items() if k != "__all__"])

        self.done["__all__"] = all_done

        done = copy(self.done)

        return rew, obs, done, {}

    def reset(self, random=True, return_init=False, config: dict = None) -> Tuple[Dict, Optional[Dict]]:
        """
        Resets the environment and returns initial observation
        :param random: if to initialize the environment randomly or statically
        :param return_init: return the initialization params
        :param config: dict with the following values 
            {airspace_bounds: [Points],
            flights: {0: {"position": Point,
                           "target": Point,
                           "airspeed": float
                           }
                      1: ...
                      }
            }
        :return: initial observation
        :return: actual configuration of the random environment if return_init is true
        """
        if random == False:
            assert config is not None, "If the reset is not random, you need to specify a configuration"

        if random:
            # create random airspace
            self.airspace = Airspace.random(self.min_area, self.max_area)
            self.flights = {}
            # create random flights
            idx = 0
            tol = self.distance_init_buffer * self.tol
            min_distance = self.distance_init_buffer * self.min_distance
            while len(self.flights) < self.num_flights:
                valid = True
                candidate = Flight.random(
                    self.airspace, self.min_speed, self.max_speed, idx, tol)

                # ensure that candidate is not in conflict
                for f in self.flights.values():
                    if candidate.position.distance(f.position) < min_distance:
                        valid = False
                        break
                if valid:
                    self.flights[idx] = candidate
                    idx += 1
            if return_init:
                env_config = {}
                env_config["airspace_bounds"] = deepcopy(
                    self.airspace.polygon.exterior.coords)
                env_config["flights"] = {}
                for f_id in self.flights:
                    env_config["flights"][f_id] = {}
                    env_config["flights"][f_id]["position"] = deepcopy(
                        self.flights[f_id].position)
                    env_config["flights"][f_id]["target"] = deepcopy(
                        self.flights[f_id].target)
                    env_config["flights"][f_id]["airspeed"] = deepcopy(
                        self.flights[f_id].airspeed)
        else:
            self.airspace = Airspace.fixed(config["airspace_bounds"])
            # create fixed_flights
            self.flights = {}
            for f_id in config["flights"].keys():
                self.flights[f_id] = Flight.fixed(config["flights"][f_id]["position"], config["flights"]
                [f_id]["target"], config["flights"][f_id]["airspeed"], self.airspace, f_id)

        # max distance inside the polygon
        minx, miny, maxx, maxy = self.airspace.polygon.bounds
        self.max_distance = Point(minx, miny).distance(Point(maxx, maxy))
        # max distance inside the screen
        minx, miny, maxx, maxy = self.airspace.polygon.buffer(
            10 * u.nm).bounds
        self.max_screen_distance = Point(
            minx, miny).distance(Point(maxx, maxy))

        if abs(maxx - minx) >= abs(maxy - miny):
            self.scaler = normalizer(
                maxx, minx, maxx, minx, self.screen_size, self.screen_size
            )
        else:
            self.scaler = normalizer(
                maxy, miny, maxy, miny, self.screen_size, self.screen_size
            )

        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = set()
        self.done = {flight_id: False for flight_id in self.flights.keys()}
        self.done["__all__"] = False

        # return initial observation
        if return_init:
            return self.observation(), env_config
        else:
            return self.observation()

    def render(self, mode=None) -> Optional[np.ndarray]:
        """
        Renders the environment
        :param mode: rendering mode
        :return:
        """

        # initialise screen
        if self.screen is None:
            pygame.init()
            if mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_size, self.screen_size))

        # init empty background
        self.surf = pygame.Surface((self.screen_size, self.screen_size))
        self.surf.fill(WHITE)

        for i, f in self.flights.items():
            if self.done[i]:
                continue
            # add fovs first to avoid drawing on other elements
            xy = self.scaler(*f.fov.exterior.coords.xy)
            xy = np.stack(xy, axis=-1)
            pygame.draw.polygon(self.surf, points=xy, color=ORANGE, width=0)

        # display airspace
        xy = self.scaler(*self.airspace.polygon.boundary.coords.xy)
        xy = np.stack(xy, axis=-1)
        gfxdraw.aapolygon(self.surf, xy, BLACK)

        # estimate radius
        radius = self.min_distance / 2.
        radius /= self.max_screen_distance
        radius *= self.screen_size

        # add current positions
        for i, f in self.flights.items():
            if self.done[i]:
                continue

            if i in self.conflicts:
                color = RED
            else:
                color = BLUE

            # add drone area
            x, y = self.scaler(f.position.x, f.position.y, use_int=True)
            pygame.draw.circle(self.surf, color=color, center=(
                x, y), radius=radius, width=1)
            font = pygame.font.SysFont('arial', 50)
            text = font.render(str(i), True, (0, 0, 0))
            text = pygame.transform.flip(text, False, True)
            self.surf.blit(text, (x, y))

            # add plan
            plan = LineString([f.position, f.target])
            xy = self.scaler(*plan.coords.xy)
            x, y = np.stack(xy, axis=-1)
            pygame.draw.line(self.surf, start_pos=x,
                             end_pos=y, color=color, width=1)

            # add track prediction
            track_prediction = LineString(
                [f.position, self.track_prediction(flight=f)])
            xy = self.scaler(*track_prediction.coords.xy)
            x, y = np.stack(xy, axis=-1)
            pygame.draw.line(self.surf, start_pos=x,
                             end_pos=y, color=color, width=2)

            # add heading
            heading = LineString([f.position, f.heading_prediction])
            xy = self.scaler(*heading.coords.xy)
            x, y = np.stack(xy, axis=-1)
            pygame.draw.line(self.surf, start_pos=x,
                             end_pos=y, color=color, width=2)

            # add wind
            wind_comp = self.wind_direction(flight=f)
            wind = LineString([f.position, wind_comp])
            xy = self.scaler(*wind.coords.xy)
            x, y = np.stack(xy, axis=-1)
            pygame.draw.line(self.surf, start_pos=x,
                             end_pos=y, color=GREEN, width=2)

        self.surf = pygame.transform.flip(self.surf, False,  True)

        if mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            ret = np.transpose(
                np.asarray(
                    pygame.surfarray.pixels3d(self.surf)
                ), axes=(1, 0, 2)
            )
            return ret

    def close(self) -> None:
        """
        Closes the viewer
        :return:
        """
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
