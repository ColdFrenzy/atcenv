import ray as ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLoggerCallback

from atcenv.common.callbacks import MyCallbacks, MediaWandbLogger
from atcenv.common.rllib_configs import multi_agent_configs, eval_configs, resources_configs, ppo_configs, model_configs
from atcenv.common.utils import parse_args
from atcenv.envs import get_env_cls

if __name__ == '__main__':

    args = parse_args()

    ##########################
    #   Init ray with degub options
    ##########################

    ray.init(local_mode=True if args.debug else False,
             num_gpus=0 if args.debug else 1,
             num_cpus=0 if args.debug else 6,
             log_to_driver=False,
             )
    env_cls = get_env_cls()

    config = {
        "env": env_cls,
        "framework": "torch",
        "callbacks": MyCallbacks,

    }

    ##########################
    #   Update config dict
    ##########################
    env_config = dict(env_config=vars(args.env))
    tmp = env_cls(env_config)

    ma_configs = multi_agent_configs(args, tmp.observation_space, tmp.action_space)
    e_configs = eval_configs(args)
    r_configs = resources_configs(args)
    p_configs = ppo_configs(args)
    m_configs= model_configs(args)

    config.update(ma_configs)
    config.update(e_configs)
    config.update(r_configs)
    config.update(env_config)
    config.update(p_configs)
    config.update(m_configs)

    ##########################
    #   Define tune loggers
    ##########################
    callbakcs = []

    wandb = MediaWandbLogger(
        vide_dir=e_configs['evaluation_config']['record_env'],
        project="atcenv",
        monitor_gym=True,
        mode="offline" if args.debug else "online",
        resume=True,
    )


    callbakcs.append(wandb)

    ##########################
    #   Start Training
    ##########################

    tune.run(
        ppo.PPOTrainer,
        config=config,
        name="ppo_trainer",
        callbacks=callbakcs,
        keep_checkpoints_num=5,

        # a very useful trick! this will resume from the last run specified by
        # sync_config (if one exists), otherwise it will start a new tuning run
        resume=False,

    )
