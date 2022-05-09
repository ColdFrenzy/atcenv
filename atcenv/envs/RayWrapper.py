from typing import Dict, Tuple

from atcenv.envs.FlightEnv import FlightEnv
from atcenv.envs.CurriculumFlightEnv import CurriculumFlightEnv


class RayWrapper(CurriculumFlightEnv):
    """Wrapper for the ray[rllib] library.

    It removes the info about done agents
    """

    def __init__(self, env_context, **kwargs):
        """
        Init used for ray support
        """
        super(RayWrapper, self).__init__(env_context, **kwargs)
        self.done_ids = []
        self._agent_ids = [idx for idx in range(self.num_flights)]

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:

        rew, obs, done, info = super(RayWrapper, self).step(actions)

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
