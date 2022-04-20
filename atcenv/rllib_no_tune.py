import ray as ray
import os
from regex import W
import torch
from atcenv.envs.CurriculumFlightEnv import CurriculumFlightEnv

from atcenv.common.wandb_callbacks import WandbCallbacks
from atcenv.models.action_mask_model import FlightActionMaskModel, FlightActionMaskRNNModel
from atcenv.common.callbacks import CurriculumCallbacks, MediaWandbLogger
from atcenv.common.rllib_configs import multi_agent_configs, eval_configs, ppo_configs, model_configs, resources_configs
from atcenv.common.utils import parse_args, curriculum_fn
from atcenv.common.custom_eval import flight_custom_eval, flight_custom_eval_no_video
from atcenv.envs import get_env_cls
from ray.rllib.agents.ppo import PPOTrainer


if __name__ == "__main__":
    args = parse_args()
    CUR_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    WEIGHTS_DIR = os.path.join(CUR_DIR, "weights")
    IMPORTANT_WEIGHTS = os.path.join(CUR_DIR, "important_weights")
    LOG_DIR = os.path.join(CUR_DIR, "log_dir")
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    if not os.path.exists(IMPORTANT_WEIGHTS):
        os.makedirs(IMPORTANT_WEIGHTS)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    ##########################
    #   Init ray with degub options
    ##########################

    # for reproducibilty
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    wandb_watch = False
    # START TRAINING
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch} of {num_epochs}")
        result = trainer_obj.train()
        default_policy = trainer_obj.get_policy("default")
        if not wandb_watch:
            wdb_callback.watch_model(default_policy.model)
            wandb_watch = True

        cur_level = ray.get(trainer_obj.workers.remote_workers(
        )[0].foreach_env.remote(lambda env: env.get_task()))
        if len(cur_level) > 0:
            cur_level = cur_level[0]
        print(f"evaluating environment with cur_level: {cur_level}")

        ##################################################
        # SAVE MEDIA
        ##################################################
        if epoch % args.media_checkpoints_freq == 0:
            env = CurriculumFlightEnv(
                **config["evaluation_config"]["env_config"], cur_level=cur_level, reward_as_dict=True)
            LOG_FILE = os.path.join(LOG_DIR, f"atc_challenge_{epoch}.log")
            eval_result, next_level = flight_custom_eval(
                env, default_policy, config["evaluation_config"]["record_env"], LOG_FILE)

            result.update(eval_result)
            wdb_callback.log_media(result)
        else:
            env = CurriculumFlightEnv(
                **config["evaluation_config"]["env_config"], cur_level=cur_level)
            eval_result, next_level = flight_custom_eval_no_video(
                env, default_policy, config["evaluation_duration"])
            result.update(eval_result)
            wdb_callback.log(result)
        ##################################################
        # SAVE CHECKPOINTS
        ##################################################
        if epoch % args.checkpoint_freq == 0:
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
        if next_level:
            # save weights used to switch between levels to the important weights dir
            NEW_IMPORTANT_WEIGHTS_DIR = os.path.join(
                IMPORTANT_WEIGHTS, f"important_weights_{cur_level}.pt")
            if not os.path.exists(NEW_IMPORTANT_WEIGHTS_DIR):
                os.makedirs(NEW_IMPORTANT_WEIGHTS_DIR)
            new_weight_file = os.path.join(
                NEW_IMPORTANT_WEIGHTS_DIR, f"model_weights_{cur_level}.pt")
            print(f"Goal reached, moving to the next level: {cur_level+1}")
            print(f"Goal reached, moving to the next level: {cur_level+1}")
            for worker in trainer_obj.workers.remote_workers():
                worker.foreach_env.remote(
                    lambda env: env.set_task(cur_level+1))
