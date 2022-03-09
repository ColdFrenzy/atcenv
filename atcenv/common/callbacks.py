import os
import pickle
from os import listdir
from os.path import isfile, join
from typing import Dict, Optional

import imageio
import numpy as np
import wandb
from ray.rllib import BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID
from ray.tune.integration.wandb import WandbLoggerCallback, _clean_log

class MyCallbacks(DefaultCallbacks):

    def __init__(self):
        super(MyCallbacks, self).__init__()

        self.num_conflicts = 0

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

        env = base_env.envs[0]

        flights = list(env.flights.values())

        speed_diff = [abs(f.optimal_airspeed - f.airspeed) for f in flights]

        episode.custom_metrics["num_conflicts"] = self.num_conflicts / 2
        episode.custom_metrics["speed_diff"] = float(np.asarray(speed_diff).mean())
        episode.custom_metrics["actions_accel"] = np.asarray(env.logging_actions['accel']).mean()
        #episode.custom_metrics["actions_track"] = float(np.bincount(env.logging_actions['track']).argmax())
        episode.custom_metrics["non_zero_obs"] = float(np.asarray(env.logging_obs['non_zero']).mean())
        episode.hist_data["actions_track"] = env.logging_actions['track']

        # get all the agents that reached the target
        done_ids = [v for k, v in env.done.items() if v and k != "__all__"]
        done_ids = len(done_ids) / len(env.flights)
        episode.custom_metrics["reached_target"] = done_ids

        self.num_conflicts = 0




class MediaWandbLogger(WandbLoggerCallback):

    def __init__(self, vide_dir, **kwargs):
        super().__init__(**kwargs)
        self.video_dir = vide_dir

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if trial not in self._trial_processes:
            self.log_trial_start(trial)

        result = _clean_log(result)

        ##############################
        #   Medias
        ###############################

        # # get all the media files in the dir and unlink
        files = [join(self.video_dir, f) for f in listdir(self.video_dir) if isfile(join(self.video_dir, f))]

        media=[x for x in files if "mp4" in x]
        media=sorted(media)[-2]
        files.pop(files.index(media))

        # get the most recent one and log it
        result["evaluation"]['episode_media'] = {
            "behaviour": wandb.Video(media, format="mp4")}

        # empty video dir
        [os.unlink(x) for x in files]


        ##############################
        #   histograms
        ###############################

        action_track=result['hist_stats']['actions_track']
        action_track= [x for sub in action_track for x in sub]
        result['hist_stats']['actions_track']=wandb.Histogram(action_track)

        self._trial_queues[trial].put(result)
