from typing import Dict, Tuple, Optional

import numpy as np

from atcenv.envs.RayWrapper import RayWrapper


class LoggerWrapper(RayWrapper):

    def __init__(self, env_context, **kwargs):
        """
        Init used for ray support
        """
        super().__init__(env_context,**kwargs)

        self.logging_actions = []
        self.logging_obs = {}
        self.logging_env = {}
        self.logging_video = []

    def reset(self) -> Dict:
        self.logging_actions = dict(
            accel=[],
            track=[]
        )
        self.logging_obs = dict(
            non_zero=[]

        )
        self.logging_env = dict(
            reached_target=[],
            steps=0,
        )

        self.logging_video = []

        return super(LoggerWrapper, self).reset()

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        obs, rew, done, info = super(LoggerWrapper, self).step(actions)

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
        non_zero_obs = sum([np.count_nonzero(x['agents_in_fov'])
                           for x in obs.values()])
        self.logging_obs['non_zero'].append(non_zero_obs)

        self.logging_env['steps']+=1

        return obs, rew, done, info
