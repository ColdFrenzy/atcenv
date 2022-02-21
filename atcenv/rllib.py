import ray as ray
from ray.rllib.agents import ppo

from atcenv.common.utils import parse_args
from atcenv.envs.RayWrapper import RayWrapper

ray.init(local_mode=True)

args = parse_args()
env_config= vars(args.env)

tmp= RayWrapper(env_config)



trainer = ppo.PPOTrainer(env=RayWrapper, config={
    "env_config": env_config,  # config to pass to env class
    "framework": "torch",
    "num_workers": 0,
    "multiagent": {
        "policies": {
            "default": (None, tmp.observation_space,
                        tmp.action_space, {}),

        },
        "policy_mapping_fn": lambda x: "default",
    },
})



while True:
    print(trainer.train())
