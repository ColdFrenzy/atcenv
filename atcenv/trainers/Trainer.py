import math
from typing import Dict

import torch


class Trainer:
    def __init__(self, agents, rollout, env, num_agents, num_episodes, device,
                 num_steps, learning_epochs):
        """__init__ method.

        Parameters
        ----------
        agents : List[atcenv.agents]
            a list of agents defined in the atcenv/agents directory
        rollout: RolloutStorage
            rollout object to feed with experiences
        env : atcenv.envs
        num_agents: int
            number of the agents in the environment
        num_episodes: int
            number of episodes to collect for each trajectories
        device: torch.device("cuda") or torch.device("cpu")
        num_steps: int
            max number of steps per episodes
        learning_epochs: int
            number of learning iterations that the algorithm does on the same
            batch of trajectories
        Returns
        -------
        None.

        """
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.learning_epochs = learning_epochs

        self.device = device
        self.agents = agents
        self.env = env
        self.rollout = rollout

    def collect_trajectories(self):
        """Update rollout with new experiences"""
        [model.eval_mode() for model in self.agents]
        # restart rollout index
        self.rollout.after_update()

        action_list = [None for _ in range(self.num_agents)]
        values_list = [None for _ in range(self.num_agents)]
        action_log_list = [None for _ in range(self.num_agents)]
        done_list = [None for _ in range(self.num_agents)]

        for episode in range(self.num_episodes):
            # print(f"episode {episode}/{self.num_episodes}...")
            observation = torch.tensor(self.env.reset())
            self.rollout.states[episode *
                                self.num_steps] = observation

            for step in range(self.num_steps):
                # print(f"step {step}/{self.num_steps}")
                obs = observation.to(self.device)

                for agent_id in range(self.num_agents):
                    with torch.no_grad():
                        value, action, action_log_prob = \
                            self.agents[agent_id].act(obs[agent_id])

                    action_list[agent_id] = action
                    values_list[agent_id] = value
                    action_log_list[agent_id] = action_log_prob

                rewards, observation, done, infos = self.env.step(
                    action_list)
                done_list = [
                    0 if i in self.env.done else 1 for i in range(self.num_agents)]

                actions = torch.Tensor(action_list).unsqueeze(-1)
                action_log_probs = torch.cat(
                    [elem.unsqueeze(0) for elem in action_log_list], 0)
                masks = torch.Tensor(done_list).unsqueeze(-1)
                rewards = torch.Tensor(rewards).unsqueeze(-1)
                values = torch.Tensor(values_list).unsqueeze(-1)
                observation = torch.tensor(observation)

                self.rollout.insert(
                    state=observation,
                    action=actions,
                    action_log_probs=action_log_probs,
                    value_preds=values,
                    reward=rewards,
                    mask=masks
                )

                if done:
                    observation = self.env.reset()

    def train(self) -> [torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict]]:
        [model.train_mode() for model in self.agents]

        logs = {str(ag): None for ag in range(self.num_agents)}

        action_losses = [0 for ag in range(self.num_agents)]
        value_losses = [0 for ag in range(self.num_agents)]
        entropies = [0 for ag in range(self.num_agents)]

        with torch.no_grad():
            next_values_list = []
            for agent_id in range(self.num_agents):
                action_logits, next_values = \
                    self.agents[agent_id].model(
                        self.rollout.states[-1, agent_id].unsqueeze(dim=0))
                next_values_list.append(next_values)
        next_value = torch.cat(next_values_list, dim=0).to(self.device)

        self.rollout.compute_returns(next_value)
        advantages = self.rollout.returns - self.rollout.value_preds
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for epoch in range(self.learning_epochs):
            data_generator = self.rollout.recurrent_generator(advantages)

            for sample in data_generator:
                states_batch, actions_batch, logs_probs_batch, \
                    values_batch, return_batch, masks_batch, adv_targ = sample

                for agent_index in range(self.num_agents):
                    agent_actions = actions_batch[:, agent_index]
                    agent_adv_targ = adv_targ[:, agent_index]
                    agent_log_probs = logs_probs_batch[:, agent_index, :]
                    agent_returns = return_batch[:, agent_index]
                    agent_values = values_batch[:, agent_index]

                    with torch.enable_grad():
                        action_loss, value_loss, entropy, log = self.agents[agent_index].ppo_step(
                            states_batch[:, agent_index], agent_actions, agent_log_probs, agent_values,
                            agent_returns, agent_adv_targ, masks_batch
                        )

                    logs[agent_id] = log

                    action_losses[agent_id] += float(action_loss)
                    value_losses[agent_id] += float(value_loss)
                    entropies[agent_id] += float(entropy)

        num_updates = self.learning_epochs * \
            int(math.ceil(self.rollout.rewards.size(0) / self.rollout.minibatch))

        action_losses = sum(action_losses) / num_updates
        value_losses = sum(value_losses) / num_updates
        entropies = sum(entropies) / num_updates

        return action_losses, value_losses, entropies, logs

    def checkpoint(self):
        pass
        # torch.save({
        #     'epoch': EPOCH,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': LOSS,
        # }, PATH)

    def restore_training(self):
        pass


# if __name__ == '__main__':

#     trainer = Trainer(ModelFree, PPO_Agent, env, params)
#     for epoch in tqdm(range(params.epochs, desc="Training...")):
#         rollout = trainer.collect_trajectories()
#         action_loss, value_loss, entropy, logs = trainer.train(rollout)

#         if params.use_wandb:
#             logs = preprocess_logs(
#                 [value_loss, action_loss, entropy, logs], trainer)
#             trainer.logger.on_batch_end(
#                 logs=logs, batch_id=epoch, rollout=rollout)

#         rollout.after_update()
