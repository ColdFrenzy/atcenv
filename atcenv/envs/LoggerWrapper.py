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

    def reset(self) -> Dict:
        self.logging_actions = []
        self.logging_obs = dict(
            non_zero=[]

        )
        return super(LoggerWrapper, self).reset()

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        obs, rew, done, info = super(LoggerWrapper, self).step(actions)

        # log mean actions
        actions = [x[0] for x in actions.values()]
        actions=np.asarray(actions).mean()
        self.logging_actions.append(actions)

        # log non zero observations
        non_zero_obs = sum([np.count_nonzero(x) for x in obs.values()])
        self.logging_obs['non_zero'].append(non_zero_obs)

        return obs, rew, done, info
