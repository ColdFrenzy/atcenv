import ray as ray
from ray.rllib.agents import ppo

from atcenv.common.utils import parse_args
from atcenv.envs.RayWrapper import RayWrapper

ray.init(local_mode=False)

args = parse_args()
env_config= vars(args.env)

tmp= RayWrapper(env_config)



trainer = ppo.PPOTrainer(env=RayWrapper, config={
    "env_config": env_config,  # config to pass to env class
    "framework": "torch",
    "num_workers": 3,
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
        "record_env": "videos",
        # "record_env": "/Users/xyz/my_videos/",
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
        "render_env": True,
    },
})



while True:
    print(trainer.train())
