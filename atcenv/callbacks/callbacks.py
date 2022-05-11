import os
from os import listdir
from os.path import isfile, join
from typing import Dict, Optional

import numpy as np
import wandb


class CurriculumCallbacks:

    def __init__(self):
        super(CurriculumCallbacks, self).__init__()

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
        self.num_conflicts += len(env.flight_env.conflicts)

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
        flights = list(env.flight_env.flights.values())
        level = env.cur_level

        speed_diff = [abs(f.optimal_airspeed - f.airspeed) for f in flights]

        episode.custom_metrics["num_conflicts"] = self.num_conflicts / 2
        episode.custom_metrics["speed_diff"] = float(
            np.asarray(speed_diff).mean())
        episode.custom_metrics["actions_accel"] = np.asarray(
            env.logging_actions['accel']).mean()
        episode.custom_metrics["actions_track"] = float(
            np.bincount(env.logging_actions['track']).argmax())
        episode.custom_metrics["non_zero_obs"] = float(
            np.asarray(env.logging_obs['non_zero']).mean())
        episode.custom_metrics["current_difficulty_level"] = level
        #episode.hist_data["actions_track"] = env.logging_actions['track']

        # get all the agents that reached the target
        done_ids = [v for k, v in env.flight_env.done.items()
                    if v and k != "__all__"]
        done_ids = len(done_ids) / len(env.flight_env.flights)
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
        files = [join(self.video_dir, f) for f in listdir(
            self.video_dir) if isfile(join(self.video_dir, f))]

        media = [x for x in files if "mp4" in x]
        media = sorted(media)[-2]
        files.pop(files.index(media))

        # get the most recent one and log it
        result["evaluation"]['episode_media'] = {
            "behaviour": wandb.Video(media, format="mp4")}

        # empty video dir
        # check if file is used somewhere else, otherwise close it
        try:
            [os.unlink(x) for x in files if x]
        except:
            pass
        self._trial_queues[trial].put(result)

# import argparse
# import os
# import torch.nn as nn
# from typing import Any, Dict, Optional, Union, List

# import wandb


# class WandbLogger:
#     def __init__(
#         self,
#         out_dir: str,
#         debug: bool,
#         hyperparams: Dict,
#         opts: Union[argparse.ArgumentParser, Dict, str, None] = {},
#         project: Optional[str] = None,
#         run_id: Optional[str] = None,
#         **kwargs,
#     ):
#         # This callback logs to wandb the interaction as they are stored in the leader process.
#         # When interactions are not aggregated in a multigpu run, each process will store
#         # its own Dict[str, Any] object in logs. For now, we leave to the user handling this case by
#         # subclassing WandbLogger and implementing a custom logic since we do not know a priori
#         # what type of data are to be logged.
#         self.opts = opts

#         # create wandb dir if not existing
#         if not os.path.isdir(out_dir):
#             os.mkdir(out_dir)

#         wandb.init(
#             # name of the wandb project
#             project=project,
#             id=run_id,
#             dir=out_dir,
#             # name of the wandb account
#             entity="francesco_diag",
#             # hyperparameters
#             config=hyperparams,
#             mode="disabled" if debug else "online",
#             **kwargs,
#         )
#         wandb.config.update(opts)

#     @staticmethod
#     def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
#         wandb.log(metrics, commit=commit, **kwargs)

#     def wandb_close(self):
#         """close method.

#         it ends the current wandb run
#         """
#         wandb.finish()


# class CustomWandbCallback(WandbLogger):
#     def __init__(
#             self,
#             train_log_step: int,
#             val_log_step: int,
#             models: List[nn.Module],
#             horizon: int,
#             **kwargs,
#     ):
#         """
#         Logs env model training onto wandb

#         :param train_log_step: number of training steps between each log
#         :param val_log_step: number of validation steps between each log
#         :param model: neural networks model to log
#         :param horizon: length of one episode
#         """

#         super(CustomWandbCallback, self).__init__(**kwargs)

#         for idx, mod in enumerate(models):
#             wandb.watch(mod, log_freq=1000, log_graph=True, idx=idx, log="all")

#         self.train_log_step = train_log_step if train_log_step > 0 else 2
#         self.val_log_step = val_log_step if val_log_step > 0 else 2
#         self.horizon = horizon
#         self.epoch = 0

#     def on_epoch_end(self, logs: Dict[str, Any], rollout):
#         """what to do after training on a batch

#         :param logs: dictionary of values to log on wandb
#         :param rollout: rollout of experience used for training
#         """

#         logs["epoch"] = self.epoch

#         actions = rollout.actions[:rollout.step].squeeze().cpu().numpy()
#         rewards = rollout.rewards[:rollout.step].squeeze().cpu().numpy()

#         # grids = write_infos(states, rollout, self.params.action_meanings)
#         # logs["behaviour"] = wandb.Video(states, fps=16, format="gif")
#         # logs["behaviour_info"] = wandb.Video(grids, fps=10, format="gif")
#         logs["hist/actions"] = actions
#         logs["hist/rewards"] = rewards
#         logs["mean_reward"] = rewards.mean()

#         self.log_to_wandb(logs, commit=True)
#         self.epoch += 1

#     def on_episode_end(self, logs):
#         """what to do at the end of an episode.

#         :params logs: dictionary of values to log on wandb
#         """
#         self.log_to_wandb(logs, commit=True)


# def delete_run(run_to_remove: str):
#     """delete_run method.

#     Parameters
#     ----------
#     run_to_remove : str
#         "<entity>/<project>/<run_id>"

#     Returns
#     -------
#     None.

#     """
#     api = wandb.Api()
#     run = api.run(run_to_remove)
#     run.delete()
