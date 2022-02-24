from typing import Dict, Optional

import numpy as np
from ray.rllib import BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID


class MyCallbacks(DefaultCallbacks):

    def __init__(self):
        super(MyCallbacks, self).__init__()

        self.num_conflicts=0

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: Episode,
                        **kwargs) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """

        env = base_env.envs[0]
        self.num_conflicts += len(env.conflicts)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy], episode: Episode,
                       **kwargs) -> None:
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """

        env=base_env.envs[0]

        flights=list(env.flights.values())

        speed_diff=[abs(f.optimal_airspeed-f.airspeed) for f in flights]

        episode.custom_metrics["num_conflicts"] = self.num_conflicts/2
        episode.custom_metrics["speed_diff"] = np.asarray(speed_diff).mean()
        episode.custom_metrics["actions"] = np.asarray(env.logging_actions).mean()
        episode.custom_metrics["non_zero_obs"] = np.asarray(env.logging_obs['non_zero']).mean()
        self.num_conflicts=0


