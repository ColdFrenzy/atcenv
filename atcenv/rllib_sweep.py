import ray as ray
import os
import wandb
from copy import copy
import torch
import inspect
from atcenv.envs.CurriculumFlightEnv import CurriculumFlightEnv

from atcenv.common.wandb_callbacks import WandbCallbacks
from atcenv.models.action_mask_model import FlightActionMaskModel, FlightActionMaskRNNModel
from atcenv.common.callbacks import CurriculumCallbacks, MediaWandbLogger
from atcenv.common.independent_PPO_configs import multi_agent_configs, eval_configs, ppo_configs, model_configs, resources_configs
from atcenv.common.utils import parse_args, curriculum_fn
from atcenv.common.custom_eval import flight_custom_eval, flight_custom_eval_no_video
from atcenv.envs import get_env_cls
from ray.rllib.agents.ppo import PPOTrainer

wandb.login()
hyperparams_defaults = dict(
    # TODO: add dropout to the models
    dropout=0.2,
    gae_lambda=0.9,
    hidden_layer_1_size=128,
    hidden_layer_2_size=16,
    hidden_layer_3_size=32,
    learn_rate=0.01,
    sgd_iterations=10,
    clip_param=0.3,
    batch_size=16,
    epochs=100,
)


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.num_cpus = 6
            self.num_gpus = 0
            self.num_workers = 5
            self.checkpoint_freq = 5
            self.media_checkpoints_freq = 5
            self.keep_checkpoints_num = 5
            self.debug = False
            self.env = CurriculumFlightEnv
            self.cur_dir = os.path.abspath(os.path.join(__file__, os.pardir))
    args = Args()
    CUR_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    WEIGHTS_DIR = os.path.join(CUR_DIR, "weights")
    IMPORTANT_WEIGHTS = os.path.join(CUR_DIR, "important_weights")
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    if not os.path.exists(IMPORTANT_WEIGHTS):
        os.makedirs(IMPORTANT_WEIGHTS)
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
    }

    ##########################
    #   Update config dict
    ##########################
    env_args = inspect.getfullargspec(args.env.__init__).args
    # remove self
    env_args.pop(0)
    env_val = inspect.getfullargspec(args.env.__init__)[3]
    env_config = {'env_config': {}}
    for i, elem in enumerate(env_args):
        env_config['env_config'][elem] = env_val[i]
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
        mode="offline" if args.debug else "online",
        config=hyperparams_defaults,
        project="atc-challenge-sweep"
    )
    # Access all hyperparameter values through wandb.config
    sweep_config = wandb.config
    run_name = wandb.run.name

    ##########################
    #   Set ray parameters to the one chosen by wandb
    ##########################
    config["model"]["custom_model_config"]["shared_fc_layers"] = (
        [sweep_config.hidden_layer_1_size, sweep_config.hidden_layer_2_size,
            sweep_config.hidden_layer_3_size],
    )
    config["lambda"] = sweep_config.gae_lambda
    config["sgd_minibatch_size"] = sweep_config.batch_size
    config["num_sgd_iter"] = sweep_config.sgd_iterations
    config["lr"] = sweep_config.learn_rate
    config["clip_param"] = sweep_config.clip_param

    ##########################
    #   Define custom training function
    ##########################
    trainer_obj = PPOTrainer(config=config)
    num_epochs = sweep_config.epochs
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
        env = CurriculumFlightEnv(
            **config["evaluation_config"]["env_config"], cur_level=cur_level)
        ##################################################
        # SAVE MEDIA
        ##################################################
        if epoch % args.media_checkpoints_freq == 0:
            eval_result, next_level = flight_custom_eval(
                env, default_policy, config["evaluation_config"]["record_env"])
            result.update(eval_result)
            wdb_callback.log_media(result)
        else:
            eval_result, next_level = flight_custom_eval_no_video(
                env, default_policy, config["evaluation_duration"])
            result.update(eval_result)
            wdb_callback.log(result)
        ##################################################
        # SAVE CHECKPOINTS
        ##################################################
        if epoch % args.checkpoint_freq == 0:
            print("Saving checkpoints...")
            NEW_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, run_name)
            if not os.path.exists(NEW_WEIGHTS_DIR):
                os.makedirs(NEW_WEIGHTS_DIR)
            new_weight_file = os.path.join(
                NEW_WEIGHTS_DIR, f"model_weights_{epoch}.pt")
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
                IMPORTANT_WEIGHTS, run_name)
            if not os.path.exists(NEW_IMPORTANT_WEIGHTS_DIR):
                os.makedirs(NEW_IMPORTANT_WEIGHTS_DIR)
            new_weight_file = os.path.join(
                NEW_IMPORTANT_WEIGHTS_DIR, f"model_weights_{cur_level}.pt")
            print(f"Goal reached, moving to the next level: {cur_level+1}")
            for worker in trainer_obj.workers.remote_workers():
                worker.foreach_env.remote(
                    lambda env: env.set_task(cur_level+1))
