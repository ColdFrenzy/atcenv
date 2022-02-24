from typing import Dict, Tuple

import gym

from atcenv.envs.FlightEnv import FlightEnv


class RayWrapper(FlightEnv):

    def __init__(self, env_context, **kwargs):
        """
        Init used for ray support
        """
        super().__init__(**env_context)

        # define spaces
        # todo: check if low/high is correct
        self.observation_space = gym.spaces.Box(low=-self.max_area, high=self.max_area,
                                                shape=(2 * self.max_agent_seen,))
        self.action_space = gym.spaces.Box(low=self.min_speed, high=self.max_speed, shape=(1,))

        self.done_ids = []

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        # ray returns an ndarray of one action, the env wants a float, so take first action
        actions = {k: v[0] for k, v in actions.items()}

        rew, obs, done, info = super(RayWrapper, self).step(actions)

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

    def reset(self) -> Dict:
        # empty the done list
        self.done_ids = []
        return super(RayWrapper, self).reset()
