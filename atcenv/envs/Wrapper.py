from typing import Dict, Tuple
import gym
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
        self.action_space = gym.spaces.Dict({
            "track": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "accel": gym.spaces.Box(low=0, high=1, shape=(1,)),
        })

    def reset(self) -> Dict:
        self.logging_actions = dict(
            accel=[],
            track=[]
        )
        self.logging_obs = dict(
            non_zero=[]

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

        return obs, rew, done, info