from typing import Dict, Tuple

import numpy as np

from atcenv.envs.RayWrapper import RayWrapper


class LoggerWrapper(RayWrapper):

    def __init__(self, env_context, **kwargs):
        """
        Init used for ray support
        """
        super().__init__(env_context)

        self.logging_actions = []
        self.logging_obs = {}
        self.logging_env = {}

    def reset(self) -> Dict:
        self.logging_actions = dict(
            accel=[],
            track=[]
        )
        self.logging_obs = dict(
            non_zero=[]

        )
        self.logging_env=dict(
            reached_target=[]
        )
        return super(LoggerWrapper, self).reset()

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        obs, rew, done, info = super(LoggerWrapper, self).step(actions)

        # log mean actions
        accel = [x["accel"] for x in actions.values()]
        track = [x["track"] for x in actions.values()]

        accel = np.asarray(accel).mean()
        track = np.asarray(track).mean()
        self.logging_actions['accel'].append(accel)
        self.logging_actions['track'].append(track)

        # log non zero observations
        non_zero_obs = sum([np.count_nonzero(x['agents_in_fov']) for x in obs.values()])
        self.logging_obs['non_zero'].append(non_zero_obs)

        #leg on env stats
        self.logging_env['reached_target'].append(len([v for k, v in done.items() if v and k!="__all__"]))

        return obs, rew, done, info


