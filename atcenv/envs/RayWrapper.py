from typing import Dict, Tuple

import gym

from atcenv.envs.FlightEnv import FlightEnv
from atcenv.envs.CurriculumFlightEnv import CurriculumFlightEnv


class RayWrapper(CurriculumFlightEnv):

    def __init__(self, env_context, **kwargs):
        """
        Init used for ray support
        """
        super(RayWrapper, self).__init__(env_context, **kwargs)
        self.num_flights

        # if 'env_config' in env_context.keys():
        #     num_flights = env_context['env_config']['num_flights']
        # else:
        #     num_flights = env_context['num_flights']

        # self._agent_ids = [idx for idx in range(num_flights)]
        self._agent_ids = [idx for idx in range(self.num_flights)]
        # define spaces
        self.observation_space = gym.spaces.Dict({
            "velocity": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "bearing": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "agents_in_fov": gym.spaces.Box(low=-1, high=1, shape=(2 * self.max_agent_seen,)),
            "distance_from_target": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "action_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(len(self.action_list),)),
        })
        self.action_space = gym.spaces.Discrete(len(self.action_list))

        self.done_ids = []

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:

        rew, obs, done, info = super(RayWrapper, self).step(actions)
        # todo:
        #   - rewards:
        #       - discostamento da traiettoria (?)

        # rllib doesn't want any input from previously done agents, so filter them out
        # TODO: check if this is correct
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
        obs = super(RayWrapper, self).reset()
        self._agent_ids = [idx for idx in range(self.num_flights)]
        return obs
