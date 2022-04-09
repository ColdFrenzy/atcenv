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
        media = sorted(media)[-2]
        files.pop(files.index(media))

        # get the most recent one and log it
        result["media"] = {
            "behaviour": wandb.Video(media, format="mp4")}

        # empty video dir
        # check if file is used somewhere else, otherwise close it
        try:
            [os.unlink(x) for x in files if x]
        except:
            pass

        wandb.log(result["media"])

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
