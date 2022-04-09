import ray as ray
import wandb
from ray import tune
from ray.rllib.agents import ppo
from ray.tune import CLIReporter
from atcenv.envs.CurriculumFlightEnv import CurriculumFlightEnv

from atcenv.common.wandb_callbacks import WandbCallbacks
from atcenv.models.action_mask_model import FlightActionMaskModel, FlightActionMaskRNNModel
from atcenv.common.callbacks import CurriculumCallbacks, MediaWandbLogger
from atcenv.common.rllib_configs import multi_agent_configs, eval_configs, ppo_configs, model_configs, resources_configs
from atcenv.common.utils import parse_args, curriculum_fn
from atcenv.common.custom_eval import FlightCustomEval
from atcenv.envs import get_env_cls
from ray.rllib.agents.ppo import PPOTrainer


if __name__ == "__main__":
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
        # "env_task_fn": curriculum_fn,
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
    config.update(env_config)
    config.update(p_configs)
    config.update(m_configs)

    ##########################
    #   Define wandb callbacks
    ##########################

    wdb_callback = WandbCallbacks(
        video_dir=e_configs['evaluation_config']['record_env'],
        project="atcenv",
        # group="PPOTrainer",
        mode="offline" if args.debug else "online",
        resume=False,
    )

    ##########################
    #   Define custom training function
    ##########################

    trainer_obj = PPOTrainer(config=config)
    num_epochs = 1000
    next_level = False
    wandb_watch = False
    # START TRAINING
    for epoch in range(num_epochs):
        result = trainer_obj.train()
        default_policy = trainer_obj.get_policy("default")
        if not wandb_watch:
            wdb_callback.watch_model(default_policy.model)
            wandb_watch = True

        cur_level = ray.get(trainer_obj.workers.remote_workers(
        )[0].foreach_env.remote(lambda env: env.get_task()))
        if len(cur_level) > 0:
            cur_level = cur_level[0]
        env = CurriculumFlightEnv(
            config["evaluation_config"]["env_config"], cur_level=cur_level)
        eval_result, next_level = FlightCustomEval(
            env, default_policy, config["evaluation_config"]["record_env"])
        result.update(eval_result)
        wdb_callback.log_media(result)
        if next_level:
            for worker in trainer_obj.workers.remote_workers():
                worker.foreach_env.remote(
                    lambda env: env.set_task(cur_level+1))
