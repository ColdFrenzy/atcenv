import torch
from tqdm import tqdm
from atcenv.models.Model import FCActorCriticModel
from atcenv.configs.Params import Params
from atcenv.envs.env import Environment
from atcenv.agents.PPO_Agent import PPO_Agent
from atcenv.trainers.Trainer import Trainer
from atcenv.agents.RolloutStorage import RolloutStorage
from atcenv.behaviour_policies.MultinomialPolicy import MultinomialPolicy
from atcenv.callbacks.callbacks import CustomWandbCallback

params = Params()
# =============================================================================
# MODEL
# =============================================================================
model_configs = params.get_model_configs()
model = FCActorCriticModel(**model_configs)
behaviour_policy = MultinomialPolicy(model)
# =============================================================================
# ENV
# =============================================================================
env_configs = params.get_env_config()
env = Environment(**env_configs)
# =============================================================================
# ROLLOUT STORAGE
# =============================================================================
rollout_configs = params.get_rollout_configs()
rollout = RolloutStorage(**rollout_configs)
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
callback_configs = params.get_callback_configs()
callbacks = CustomWandbCallback(
    **callback_configs, models=[model]) if params.use_wandb else None

# =============================================================================
# TRAINER
# =============================================================================
trainer_configs = params.get_trainer_configs()
trainer = Trainer(agents, rollout, env, callbacks=callbacks, **trainer_configs)

# params.epochs
for epoch in tqdm(range(params.epochs), desc="Training..."):
    print("\nCollecting Trajectories...")
    trainer.collect_trajectories()
    print("Updating Models...")
    action_loss, value_loss, entropy, logs = trainer.train()
    if epoch % params.val_log_step == 0 and params.val_log_step != 0:
        print("Evaluating Model...")
        trainer.evaluate()
