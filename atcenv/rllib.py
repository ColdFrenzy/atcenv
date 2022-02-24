import ray as ray
import torch
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLoggerCallback

from atcenv.common.callbacks import MyCallbacks
from atcenv.common.utils import parse_args
from atcenv.envs import get_env_cls

args = parse_args()

ray.init(local_mode=True if args.debug else False,
         num_gpus=0 if args.debug else 1,
         num_cpus=0 if args.debug else 6,
         )
env_config = vars(args.env)

env_cls= get_env_cls()

tmp = env_cls(env_config)

config = {
    "env": env_cls,
    "env_config": env_config,  # config to pass to env class
    "framework": "torch",
    "num_workers": 0 if args.debug else 3,
    "num_gpus": 0 if args.debug else 1,
    "callbacks": MyCallbacks,
    "multiagent": {
        "policies": {
            "default": (None, tmp.observation_space,
                        tmp.action_space, {}),

        },
        "policy_mapping_fn": lambda x: "default",
    },
    # Evaluate once per training iteration.
    "evaluation_interval": 1,
    # Run evaluation on (at least) two episodes
    "evaluation_duration": 2,
    # ... using one evaluation worker (setting this to 0 will cause
    # evaluation to run on the local evaluation worker, blocking
    # training until evaluation is done).
    "evaluation_num_workers": 1,
    # Special evaluation config. Keys specified here will override
    # the same keys in the main config, but only for evaluation.
    "evaluation_config": {
        # Store videos in this relative directory here inside
        # the default output dir (~/ray_results/...).
        # Alternatively, you can specify an absolute path.
        # Set to True for using the default output dir (~/ray_results/...).
        # Set to False for not recording anything.
        # "record_env": "videos",
        # "record_env": "/Users/xyz/my_videos/",
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
        "render_env": False,
    },
}

wandb = WandbLoggerCallback(
    project="atcenv"
)

callbakcs = []

if not args.debug:
    callbakcs.append(wandb)

tune.run(
    ppo.PPOTrainer,
    config=config,
    name="ppo_trainer",
    callbacks=callbakcs,
)
