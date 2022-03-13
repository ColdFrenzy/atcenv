import argparse
import os
import torch.nn as nn
from typing import Any, Dict, Optional, Union, List

import wandb


class WandbLogger:
    def __init__(
        self,
        out_dir: str,
        debug: bool,
        hyperparams: Dict,
        opts: Union[argparse.ArgumentParser, Dict, str, None] = {},
        project: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        # This callback logs to wandb the interaction as they are stored in the leader process.
        # When interactions are not aggregated in a multigpu run, each process will store
        # its own Dict[str, Any] object in logs. For now, we leave to the user handling this case by
        # subclassing WandbLogger and implementing a custom logic since we do not know a priori
        # what type of data are to be logged.
        self.opts = opts

        # create wandb dir if not existing
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        wandb.init(
            # name of the wandb project
            project=project,
            id=run_id,
            dir=out_dir,
            # name of the wandb account
            entity="francesco_diag",
            # hyperparameters
            config=hyperparams,
            mode="disabled" if debug else "online",
            **kwargs,
        )
        wandb.config.update(opts)

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def wandb_close(self):
        """close method.

        it ends the current wandb run
        """
        wandb.finish()


class CustomWandbCallback(WandbLogger):
    def __init__(
            self,
            train_log_step: int,
            val_log_step: int,
            models: List[nn.Module],
            horizon: int,
            action_meaning: Dict[str, str],
            **kwargs,
    ):
        """
        Logs env model training onto wandb

        :param train_log_step: number of training steps between each log
        :param val_log_step: number of validation steps between each log
        :param model: neural networks model to log
        :param horizon: length of one episode
        """

        super(CustomWandbCallback, self).__init__(**kwargs)

        for idx, mod in enumerate(models):
            wandb.watch(mod, log_freq=1000, log_graph=True, idx=idx, log="all")

        self.train_log_step = train_log_step if train_log_step > 0 else 2
        self.val_log_step = val_log_step if val_log_step > 0 else 2
        self.horizon = horizon
        self.action_meaning = action_meaning
        self.epoch = 0

    def on_batch_end(self, logs: Dict[str, Any], batch_id: int, rollout):
        """what to do after training on a batch

        :param logs: dictionary of values to log on wandb
        :param batch_id: 
        :param rollout: rollout of experience used for training
        """

        logs["epoch"] = batch_id

        if batch_id % self.log_behavior_step == 0:

            actions = rollout.actions[:rollout.step].squeeze().cpu().numpy()
            rewards = rollout.rewards[:rollout.step].squeeze().cpu().numpy()

            # grids = write_infos(states, rollout, self.params.action_meanings)
            # logs["behaviour"] = wandb.Video(states, fps=16, format="gif")
            # logs["behaviour_info"] = wandb.Video(grids, fps=10, format="gif")
            logs["hist/actions"] = actions
            logs["hist/rewards"] = rewards
            logs["mean_reward"] = rewards.mean()
            logs["episode_length"] = rollout.step

        self.log_to_wandb(logs, commit=True)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any], model_path: str):
        self.epoch += 1


def delete_run(run_to_remove: str):
    """delete_run method.

    Parameters
    ----------
    run_to_remove : str
        "<entity>/<project>/<run_id>"

    Returns
    -------
    None.

    """
    api = wandb.Api()
    run = api.run(run_to_remove)
    run.delete()
