from typing import Dict, List, Tuple

import gym

from atcenv.envs.FlightEnv import FlightEnv


class RayWrapper(FlightEnv):

    def __init__(self, env_context, **kwargs):
        """
        Init used for ray support
        """
        super().__init__(**env_context)

        # define spaces
        self.observation_space = gym.spaces.Box(low=0, high=self.max_area, shape=(2 * self.max_agent_seen,))
        self.action_space = gym.spaces.Box(low=self.min_speed, high=self.max_speed, shape=(1,))

        self.done_ids = []

    def step(self, action: List) -> Tuple[Dict, Dict, Dict, Dict]:
        rew, obs, done, info = super(RayWrapper, self).step(action)

        # rllib doesn't want any input from previously done agents, so filter them out
        for done_id in self.done_ids:
            rew.pop(done_id)
            obs.pop(done_id)
            done.pop(done_id)

        # get new dones to add
        done_keys = [k for k, v in done.items() if v and k != "__all__"]
        self.done_ids += done_keys
        self.done_ids = list(set(self.done_ids))

        return obs, rew, done, info
