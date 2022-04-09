import os
import ray as ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLoggerCallback


from atcenv.models.action_mask_model import FlightActionMaskModel, FlightActionMaskRNNModel
from atcenv.common.callbacks import MyCallbacks, CurriculumCallbacks, MediaWandbLogger
from atcenv.common.rllib_configs import multi_agent_configs, eval_configs, ppo_configs, model_configs, resources_configs
from atcenv.common.utils import parse_args, curriculum_fn
from atcenv.envs import get_env_cls
from ray.tune import CLIReporter

if __name__ == '__main__':
    args = parse_args()

    ##########################
    #   Init ray with degub options
    ##########################
    r_configs = resources_configs(args)
    ray.shutdown()
    ray.init(local_mode=True if args.debug else False,
             num_gpus=r_configs["num_gpus"],
             num_cpus=r_configs["num_cpus"],
             log_to_driver=args.debug,
             )
    env_cls = get_env_cls()

    config = {
        "env": env_cls,
        "framework": "torch",
        "callbacks": CurriculumCallbacks,
        "env_task_fn": curriculum_fn,
    }

    ##########################
    #   Update config dict
    ##########################
    env_config = dict(env_config=vars(args.env))
    tmp = env_cls(env_config)

    ma_configs = multi_agent_configs(
        args, tmp.observation_space, tmp.action_space)
    e_configs = eval_configs(args)
    p_configs = ppo_configs(args)
    m_configs = model_configs(args)

    config.update(ma_configs)
    config.update(e_configs)
    # config.update(r_configs)
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
    # check the hardware resources needed.
    # print(ppo.PPOTrainer.default_resource_request(config=config)._bundles)

    tune.run(
        ppo.PPOTrainer,
        config=config,
        name="ppo_trainer",
        callbacks=callbakcs,
        keep_checkpoints_num=5,

        # a very useful trick! this will resume from the last run specified by
        # sync_config (if one exists), otherwise it will start a new tuning run
        resume=args.resume,
        progress_reporter=CLIReporter()
    )
