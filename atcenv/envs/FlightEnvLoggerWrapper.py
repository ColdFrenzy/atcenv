from typing import Dict, Tuple, Optional

import numpy as np
from .CurriculumFlightEnv import CurriculumFlightEnv
from .FlightEnv import FlightEnv


class CurriculumFlightEnvLoggerWrapper(CurriculumFlightEnv):
    """Compute additional information to log about the environment."""

    def __init__(self, env_context=None, **kwargs):
        """
        Init used for ray support
        """
        super().__init__(env_context, **kwargs)

        self.logging_actions = []
        self.logging_obs = {}
        self.logging_env = {}
        self.logging_video = []

    def reset(self, **kwargs) -> Dict:
        self.logging_actions = dict(
            accel=[],
            track=[]
        )
        self.logging_obs = dict(
            non_zero=[]

        )
        self.logging_env = dict(
            reached_target=[]
        )

        self.logging_video = []

        return super(CurriculumFlightEnvLoggerWrapper, self).reset(**kwargs)

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        obs, rew, done, info = super(
            CurriculumFlightEnvLoggerWrapper, self).step(actions)

        # log mean actions
        accel = [self.action_list[x][1]
                 for x in actions.values()]
        track = [self.action_list[x][0]
                 for x in actions.values()]

        accel = np.asarray(accel).mean()
        track = np.bincount(track).argmax()
        self.logging_actions['accel'].append(accel)
        self.logging_actions['track'].append(track)

        # log non zero observations
        # TODO: unconmment
        # non_zero_obs = sum([np.count_nonzero(x['agents_in_fov'])
        #                    for x in obs.values()])
        # self.logging_obs['non_zero'].append(non_zero_obs)

        return obs, rew, done, info


class FlightEnvLoggerWrapper(FlightEnv):
    """Compute additional information to log about the environment."""

    def __init__(self, env_context=None, **kwargs):
        """
        Init used for ray support
        """
        super().__init__(**kwargs)

        self.logging_actions = []
        self.logging_obs = {}
        self.logging_env = {}
        self.logging_video = []

    def reset(self, **kwargs) -> Dict:
        self.logging_actions = dict(
            accel=[],
            track=[]
        )
        self.logging_obs = dict(
            non_zero=[]

        )
        self.logging_env = dict(
            reached_target=[]
        )

        self.logging_video = []

        return super(FlightEnvLoggerWrapper, self).reset(**kwargs)

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        obs, rew, done, info = super(
            FlightEnvLoggerWrapper, self).step(actions)

        # log mean actions
        accel = [self.action_list[x][1]
                 for x in actions.values()]
        track = [self.action_list[x][0]
                 for x in actions.values()]

        accel = np.asarray(accel).mean()
        track = np.bincount(track).argmax()
        self.logging_actions['accel'].append(accel)
        self.logging_actions['track'].append(track)

        return obs, rew, done, info
