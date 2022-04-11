import wandb
import argparse
import os
from typing import Any, Dict, Optional, Union

import wandb


class WandbCallbacks():
    def __init__(
        self,
        project: Optional[str] = None,
        run_id: Optional[str] = None,
        mode: Optional[str] = None,
        group: Optional[str] = None,
        video_dir: Optional[str] = None,
        **kwargs,
    ):

        wandb.init(
            project=project,
            id=run_id,
            mode=mode,
            # group=group,
            **kwargs,
        )
        self.video_dir = video_dir

    def log_media(self, result: Dict):

        ##############################
        #   Medias
        ###############################

        # # get all the media files in the dir and unlink
        files = [os.path.join(self.video_dir, f) for f in os.listdir(
            self.video_dir) if os.path.isfile(os.path.join(self.video_dir, f))]

        media = [x for x in files if "mp4" in x]
        custom_policy = sorted(media)[0]
        standard = sorted(media)[1]
        # files.pop(files.index(custom_policy))
        # files.pop(files.index(standard))
        # get the most recent one and log it
        result["media"] = {
            "Custom_Policy": wandb.Video(custom_policy, format="mp4"),
            "Standard": wandb.Video(standard, format="mp4")}

        new_result = return_results(result)

        wandb.log(new_result)

        # empty video dir
        # check if file is used somewhere else, otherwise close it
        # try:
        [os.unlink(x) for x in files if x]
        # except:
        #     pass

    def watch_model(self, model):
        wandb.watch(model, log_freq=1, log_graph=True, log="all")

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def wandb_close(self):
        """close method.

        it ends the current wandb run
        """
        wandb.finish()


def return_results(results):
    """Utility function, it returns the result dict in such a way that
    it's loggable by wandb
    """
    new_results = {}
    for elem in results.keys():
        # training configuration. It's already in the file rllib_configs
        if elem == "config" or isinstance(results[elem], bool) or isinstance(results[elem], str):
            pass
        elif isinstance(results[elem], dict):
            for inner_elem in results[elem].keys():
                if isinstance(results[elem][inner_elem], bool) or isinstance(results[elem][inner_elem], str):
                    pass
                elif isinstance(results[elem][inner_elem], list):
                    new_results[elem + "/" +
                                inner_elem] = wandb.Histogram(results[elem][inner_elem])
                else:
                    new_results[elem + "/" +
                                inner_elem] = results[elem][inner_elem]
        else:
            new_results[elem] = results[elem]

    return new_results
