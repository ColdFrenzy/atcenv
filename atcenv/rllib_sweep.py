import os
import ray as ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLoggerCallback

from atcenv.common.callbacks import MyCallbacks, MediaWandbLogger
from atcenv.common.rllib_configs import multi_agent_configs, eval_configs, resources_configs, ppo_configs, model_configs
from atcenv.common.utils import parse_args
from atcenv.envs import get_env_cls
from ray.tune import CLIReporter

if __name__ == '__main__':

    args = parse_args()

    ##########################
    #   Init ray with degub options
    ##########################

    ray.shutdown()
    ray.init(local_mode=True if args.debug else False,
             num_gpus=0 if args.debug else args.num_gpus,
             num_cpus=2 if args.debug else args.num_cpus,
             log_to_driver=False,  # args.debug,
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

    ma_configs = multi_agent_configs(
        args, tmp.observation_space, tmp.action_space)
    e_configs = eval_configs(args)
    r_configs = resources_configs(args)
    p_configs = ppo_configs(args)
    m_configs = model_configs(args)

    config.update(ma_configs)
    config.update(e_configs)
    config.update(r_configs)
    config.update(env_config)
    config.update(p_configs)
    config.update(m_configs)

    ########################
    # SWEEP PARAMS
    ########################

    #PPO
    config['lr'] = tune.grid_search([5e-3, 5e-4, 5e-5])
    config['vf_clip_param'] = tune.grid_search([100, 10, 1])
    config['batch_mode'] = tune.grid_search(["truncate_episodes", "complete_episodes"])

    # Model LSTM
    config['model']['use_lstm'] = tune.grid_search([True, False])
    config['model']['lstm_use_prev_action'] = tune.grid_search([True, False])
    config['model']['lstm_use_prev_reward'] = tune.grid_search([True, False])

    # Model Attention
    config['model']['use_attention'] = tune.grid_search([True, False])
    config['model']['attention_num_heads'] = tune.grid_search([1,4,8])
    config['model']['attention_num_transformer_units'] = tune.grid_search([1, 2])
    config['model']['use_attention'] = tune.grid_search([True, False])

    config['epochs']=100


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
