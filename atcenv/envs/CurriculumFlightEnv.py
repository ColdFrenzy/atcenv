import gym
import random

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override
from atcenv.envs.FlightEnv import FlightEnv
from ray.rllib import MultiAgentEnv


class CurriculumFlightEnv(MultiAgentEnv, TaskSettableEnv):
    """Curriculum learning capable env.
    This simply wraps a FlightEnv env and makes it harder with each
    task"""
    LEVELS = [{"min_area": 50 * 50, "max_area": 100*100, "num_flights": 3},
              {"min_area": 80 * 80, "max_area": 120*120, "num_flights": 4},
              {"min_area": 100 * 100, "max_area": 140*140, "num_flights": 5},
              {"min_area": 110 * 110, "max_area": 160*160, "num_flights": 6},
              {"min_area": 115 * 115, "max_area": 170*170, "num_flights": 7},
              {"min_area": 120 * 120, "max_area": 180*180, "num_flights": 8},
              {"min_area": 125 * 125, "max_area": 200*200, "num_flights": 9},
              {"min_area": 125 * 125, "max_area": 200*200, "num_flights": 10}
              ]

    def __init__(self, config: EnvContext):
        self.cur_level = config.get("start_level", 1)
        self.flight_env = None
        self._make_flight_env()  # create the flightenv
        self.switch_env = False
        # Variables needed from the wrappers
        self.action_list = self.flight_env.action_list
        self.max_agent_seen = self.flight_env.max_agent_seen

    def reset(self):
        if self.switch_env:
            self.switch_env = False
            self._make_flight_env()
        return self.flight_env.reset()

    def step(self, action):
        r, s, d, i = self.flight_env.step(action)
        # Make rewards scale with the level exponentially:
        # Level 1: x1
        # Level 2: x10
        # Level 3: x100, etc..
        for f_id in r.keys():
            r[f_id] *= 10 ** (self.cur_level - 1)

        return r, s, d, i

    @ override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(1, len(self.LEVELS)) for _ in range(n_tasks)]

    @ override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    @ override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        self.cur_level = task
        self.switch_env = True

    def _make_flight_env(self):
        self.flight_env = FlightEnv(**self.LEVELS[self.cur_level-1])
