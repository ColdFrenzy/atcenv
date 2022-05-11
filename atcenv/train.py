import torch
import wandb
import random
from tqdm import tqdm
from atcenv.models.action_mask_model import FlightActionMaskModel
from atcenv.configs.Params import Params
from atcenv.envs.FlightEnv import FlightEnv
from atcenv.agents.PPO_Agent import PPO_Agent
from atcenv.trainers.Trainer import Trainer
from atcenv.agents.RolloutStorage import RolloutStorage
from atcenv.behaviour_policies.MultinomialPolicy import MultinomialPolicy
from atcenv.evaluation.custom_eval import flight_custom_eval, flight_custom_eval_no_video
# from atcenv.callbacks.wandb_callbacks import WandbCallbacks
# from atcenv.callbacks.callbacks import CurriculumCallbacks

random.seed(7)

params = Params()
# =============================================================================
# ENV
# =============================================================================
env_configs = params.get_env_config()
env = FlightEnv(**env_configs)
obs_space = env.observation_space
obs_shape = 0
for obs in obs_space.spaces.values():
    obs_shape += obs.shape[0]
action_space = env.action_space
num_actions = action_space.n
# =============================================================================
# MODEL
# =============================================================================
model_configs = params.get_model_configs()
model = FlightActionMaskModel(
    obs_space=obs_space, action_space=action_space, **model_configs)
behaviour_policy = MultinomialPolicy(model)
# =============================================================================
# ROLLOUT STORAGE
# =============================================================================
rollout_configs = params.get_rollout_configs()
rollout = RolloutStorage(
    obs_shape=obs_shape, num_actions=num_actions, **rollout_configs)
rollout.to(params.device)
# =============================================================================
# AGENT
# =============================================================================
agent_configs = params.get_ppo_configs()
agents = [PPO_Agent(**agent_configs, policy=behaviour_policy)
          for i in range(params.num_flights)]
# =============================================================================
# CALLBACKS
# =============================================================================
# TODO: fix all the callbacks
callback_configs = params.get_callback_configs()
# wdb_callback = WandbCallbacks(
#     video_dir=callback_configs["video_dir"]
#     project="atcenv",
#     # group="PPOTrainer",
#     mode="offline" if args.debug else "online",
#     resume=False,
#     name="target_reached_rew1-drift_penalty0.1-no_drift_obs-no_agent_in_fov-no_entropy-no_kl"
# )
# callbacks = CustomWandbCallback(
#     **callback_configs, models=[model]) if params.use_wandb else None

# =============================================================================
# TRAINER
# =============================================================================
trainer_configs = params.get_trainer_configs()
trainer = Trainer(agents, rollout, env, callbacks=None, **trainer_configs)

# params.epochs
for epoch in tqdm(range(params.epochs), desc="Training..."):
    print("\nCollecting Trajectories...")
    trainer.collect_trajectories()
    print("Updating Models...")
    action_loss, value_loss, entropy, logs = trainer.train()
    if epoch % params.val_log_step == 0 and params.val_log_step != 0:
        print("Evaluating Model...")
        trainer.evaluate()

    # # TODO
    # print(f"Epoch {epoch} of {num_epochs}")
    # result = trainer_obj.train()
    # default_policy = trainer_obj.get_policy("default")
    # if not wandb_watch:
    #     wdb_callback.watch_model(default_policy.model)
    #     wandb_watch = True

    # cur_level = ray.get(trainer_obj.workers.remote_workers(
    # )[0].foreach_env.remote(lambda env: env.get_task()))
    # if len(cur_level) > 0:
    #     cur_level = cur_level[0]
    # print(f"evaluating environment with cur_level: {cur_level}")

    # ##################################################
    # # SAVE MEDIA
    # ##################################################
    # if epoch % args.media_checkpoints_freq == 0 and epoch != 0:
    #     env = FlightEnvLoggerWrapper(
    #         **config["evaluation_config"]["env_config"], cur_level=cur_level, reward_as_dict=True)
    #     LOG_FILE = os.path.join(LOG_DIR, f"atc_challenge_{epoch}.log")
    #     eval_result, next_level = flight_custom_eval(
    #         env, default_policy, config["evaluation_config"]["record_env"], LOG_FILE)

    #     result.update(eval_result)
    #     wdb_callback.log_media(result)
    # else:
    #     env = FlightEnvLoggerWrapper(
    #         **config["evaluation_config"]["env_config"], cur_level=cur_level)
    #     eval_result, next_level = flight_custom_eval_no_video(
    #         env, default_policy, config["evaluation_duration"])
    #     result.update(eval_result)
    #     wdb_callback.log(result)
    # ##################################################
    # # SAVE CHECKPOINTS
    # ##################################################
    # if epoch % args.checkpoint_freq == 0 and epoch != 0:
    #     print("Saving checkpoints...")
    #     new_weight_file = os.path.join(
    #         WEIGHTS_DIR, f"model_weights_{epoch}.pt")
    #     torch.save(default_policy.model.state_dict(), new_weight_file)
    #     weights_file = [os.path.join(WEIGHTS_DIR, w)
    #                     for w in os.listdir(WEIGHTS_DIR)]
    #     print("Done")
    #     if len(weights_file) > args.keep_checkpoints_num:
    #         print("Removing old checkpoints...")
    #         oldest_file = min(weights_file, key=os.path.getctime)
    #         os.remove(os.path.join(WEIGHTS_DIR, oldest_file))
    #         print("Done")
    # if next_level:
    #     # save weights used to switch between levels to the important weights dir
    #     NEW_IMPORTANT_WEIGHTS_DIR = os.path.join(
    #         IMPORTANT_WEIGHTS, f"important_weights_{cur_level}.pt")
    #     if not os.path.exists(NEW_IMPORTANT_WEIGHTS_DIR):
    #         os.makedirs(NEW_IMPORTANT_WEIGHTS_DIR)
    #     new_weight_file = os.path.join(
    #         NEW_IMPORTANT_WEIGHTS_DIR, f"model_weights_{cur_level}.pt")
    #     print(f"Goal reached, moving to the next level: {cur_level+1}")
    #     print(f"Goal reached, moving to the next level: {cur_level+1}")
    #     for worker in trainer_obj.workers.remote_workers():
    #         worker.foreach_env.remote(
    #             lambda env: env.set_task(cur_level+1))
