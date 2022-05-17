from logging import Logger
import ray as ray
import os
from regex import W
import torch
import random
from atcenv.envs.FlightEnv import FlightEnv
from atcenv.models.action_mask_model import FlightActionMaskModel, FlightActionMaskRNNModel
from atcenv.common.wandb_callbacks import WandbCallbacks
from atcenv.common.callbacks import QmixCallbacks
from atcenv.common.qmix_configs import multi_agent_configs, eval_configs, qmix_configs, model_configs, resources_configs
from atcenv.common.utils import parse_args
from atcenv.common.custom_eval import flight_custom_eval, flight_custom_eval_no_video
from gym.spaces import Tuple
from ray.rllib.agents.qmix import QMixTrainer
from ray.tune import register_env
from atcenv.envs.FlightEnvLoggerWrapper import FlightEnvLoggerWrapper

random.seed(7)

if __name__ == "__main__":
    env_cls = FlightEnvLoggerWrapper
    args = parse_args(env_cls)
    CUR_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    WEIGHTS_DIR = os.path.join(CUR_DIR, "weights")
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "qmix")
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    IMPORTANT_WEIGHTS = os.path.join(CUR_DIR, "important_weights")
    if not os.path.exists(IMPORTANT_WEIGHTS):
        os.makedirs(IMPORTANT_WEIGHTS)
    IMPORTANT_WEIGHTS = os.path.join(IMPORTANT_WEIGHTS, "qmix")
    if not os.path.exists(IMPORTANT_WEIGHTS):
        os.makedirs(IMPORTANT_WEIGHTS)
    LOG_DIR = os.path.join(CUR_DIR, "log_dir")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(LOG_DIR, "qmix")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
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
    ##########################
    #   Update config dict
    ##########################
    env_config = dict(env_config=vars(args.env))
    env_config["env_config"]["qmix_obs"] = True
    tmp = env_cls(**vars(args.env))
    grouping = {
        "group_1": [i for i in range(tmp.num_flights)],
    }
    action_space = Tuple([tmp.action_space for _ in range(tmp.num_flights)])
    obs_space = Tuple([tmp.observation_space for _ in range(tmp.num_flights)])
    register_env("grouped_flightenv",
                 lambda cfg: FlightEnvLoggerWrapper(cfg).with_agent_groups(
                     grouping, obs_space=obs_space, act_space=action_space
                 ))
    config = {
        "env": "grouped_flightenv",
        "framework": "torch",
        "callbacks": QmixCallbacks,
        # "env_task_fn": curriculum_fn,
    }

    ma_configs = multi_agent_configs(
        args, obs_space, action_space)
    e_configs = eval_configs(args)
    qmix_configs = qmix_configs(args)
    m_configs = model_configs(args)
    config.update(e_configs)
    config.update(ma_configs)
    config.update(env_config)
    config.update(qmix_configs)
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
        name="QMIX-target_reached_rew-collision-penalty"
    )

    ##########################
    #   Define custom training function
    ##########################

    trainer_obj = QMixTrainer(config=config)
    num_epochs = 1000
    wandb_watch = False
    # START TRAINING
    for epoch in range(0, num_epochs):
        print(f"Epoch {epoch} of {num_epochs}")
        result = trainer_obj.train()
        default_policy = trainer_obj.get_policy("default")
        if not wandb_watch:
            wdb_callback.watch_model(default_policy.model)
            wandb_watch = True

        ##################################################
        # SAVE MEDIA
        ##################################################
        if epoch % args.media_checkpoints_freq == 0 and epoch != 0:
            env = FlightEnvLoggerWrapper(
                **config["evaluation_config"]["env_config"], reward_as_dict=True)
            LOG_FILE = os.path.join(LOG_DIR, f"atc_challenge_{epoch}.log")
            eval_result, next_level = flight_custom_eval(
                env, default_policy, config["evaluation_config"]["record_env"], LOG_FILE,qmix=True)

            result.update(eval_result)
            wdb_callback.log_media(result)
        else:
            env = FlightEnvLoggerWrapper(
                **config["evaluation_config"]["env_config"])
            eval_result, next_level = flight_custom_eval_no_video(
                env, default_policy, config["evaluation_duration"],qmix=True)
            result.update(eval_result)
            wdb_callback.log(result)
        ##################################################
        # SAVE CHECKPOINTS
        ##################################################
        if epoch % args.checkpoint_freq == 0 and epoch != 0:
            print("Saving checkpoints...")
            new_weight_file = os.path.join(
                WEIGHTS_DIR, f"model_weights_{epoch}.pt")
            torch.save(default_policy.model.state_dict(), new_weight_file)
            weights_file = [os.path.join(WEIGHTS_DIR, w)
                            for w in os.listdir(WEIGHTS_DIR)]
            print("Done")
            if len(weights_file) > args.keep_checkpoints_num:
                print("Removing old checkpoints...")
                oldest_file = min(weights_file, key=os.path.getctime)
                os.remove(os.path.join(WEIGHTS_DIR, oldest_file))
                print("Done")
