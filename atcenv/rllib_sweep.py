import ray as ray
from hyperopt import hp
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from atcenv.common.callbacks import MyCallbacks, MediaWandbLogger
from atcenv.common.rllib_configs import multi_agent_configs, eval_configs, resources_configs, ppo_configs, model_configs
from atcenv.common.utils import parse_args
from atcenv.envs import get_env_cls

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
    # Step 1: Specify the search space

    ppo = False
    lstm = False
    attention = False
    fc = False
    hyperopt_space = {}

    if ppo:
        hyperopt_space.update(
            {
                # PPO
                "lr": hp.uniform("netG_lr", 1e-5, 1e-2),
                "netD_lr": hp.quniform("vf_clip_param", 1, 100, 10),
                "batch_mode": hp.choice("batch_mode", ["truncate_episodes", "complete_episodes"])
            }
        )
    #########
    # MODEL
    #########
    if lstm and not attention:
        hyperopt_space.update(
            {
                "use_lstm": hp.choice("use_lstm", [True]),
                "lstm_use_prev_action": hp.choice("lstm_use_prev_action", [True, False]),
                "lstm_use_prev_reward": hp.choice("lstm_use_prev_reward", [True, False]),
            }
        )

    if attention and not lstm:
        hyperopt_space.update(
            {
                "use_attention": hp.choice("use_attention", [True]),
                "attention_num_heads": hp.quniform("attention_num_heads", 1, 8, 1),
                "attention_num_transformer_units": hp.quniform("attention_num_transformer_units", 1, 1, 1),

            }
        )

    if fc:
        hyperopt_space.update(
            {
                "fcnet_hiddens": hp.choice("fcnet_hiddens", [[64], [64, 32], [64, 16, 32], [128, 64, 16, 32]]),

            }
        )

    hyperopt_alg = HyperOptSearch(space=hyperopt_space, metric="custom_metrics/num_conflicts_mean", mode="min")
    # hyperopt_alg = ConcurrencyLimiter(hyperopt_alg, max_concurrent=2)
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
    #  Early stopping
    ##########################
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='custom_metrics/num_conflicts_mean',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1)

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
        progress_reporter=CLIReporter(),
        scheduler=asha_scheduler,
    )
