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
    **callback_configs, model=model) if params.use_wandb else None

# =============================================================================
# TRAINER
# =============================================================================
trainer_configs = params.get_trainer_configs()
trainer = Trainer(agents, rollout, env, **trainer_configs)

# params.epochs
for epoch in tqdm(range(2), desc="Training..."):
    print("\ncollecting trajectories")
    trainer.collect_trajectories()
    print("updating models")
    action_loss, value_loss, entropy, logs = trainer.train()

    # if params.use_wandb:
    #     trainer.logger.on_batch_end(
    #         logs=logs, batch_id=epoch, rollout=rollout)
