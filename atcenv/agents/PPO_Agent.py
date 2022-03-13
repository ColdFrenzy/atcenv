import torch
from torch import optim, nn


class PPO_Agent:
    def __init__(self, policy, device, lr, eps,
                 clip_param, clip_value_loss, max_grad_norm,
                 entropy_coef, value_loss_coef):
        """PPO agent initialization
        Parameters
        ----------
        policy : BehaviourPolicy
            defined in atcenv.behaviour_policies
        device : torch.device("cuda") or torch.device("cpu")
        lr : float
            learning rate
        eps : float
            epsilon parameter for the optimizer
        clip_param : float
            clip parameter for the action loss
        clip_value_loss : float
            clip parameter for the value loss
        max_grad_norm : int
            clipping parameter for the gradient
        entropy_coef : float
            entropy coefficient
        value_loss_coef : float
            value loss coefficient

        Returns
        -------
        None.

        """
        self.policy = policy
        self.model = policy.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=eps)

        self.clip_param = clip_param
        self.clip_value_loss = clip_value_loss
        self.max_grad_norm = max_grad_norm

        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def get_modules(self):
        return self.model.get_modules()

    def eval_mode(self):
        """set the model in evaluation mode."""
        self.model.eval()

    def train_mode(self):
        """set the model in training mode."""
        self.model.train()

    def act(self, inputs):
        """select action through the behaviour_policy."""
        return self.policy.act(inputs)

    def ppo_step(self, states, actions, log_probs, values, returns, adv_targ, masks):
        """optimization step using the PPO loss.

        """
        def mean_fn(tensor): return float(tensor.mean())

        logs = dict(
            ratio=[], surr1=[], surr2=[], returns=[],
            adv_targ=[], perc_surr1=[], perc_surr2=[],
            curr_log_probs=[], old_log_probs=[]
        )

        curr_values, curr_log_probs, entropy = self.model.evaluate_actions(
            states)

        logs["curr_log_probs"].append(mean_fn(curr_log_probs))
        logs["old_log_probs"].append(mean_fn(log_probs))
        logs["returns"].append(mean_fn(returns))
        logs["adv_targ"].append(mean_fn(values))

        single_log_prob = log_probs.gather(-1, actions)
        single_curr_log_prob = curr_log_probs.gather(-1, actions)

        ratio = torch.exp(single_curr_log_prob - single_log_prob)
        surr1 = ratio * adv_targ
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        action_loss = -torch.min(surr1, surr2).mean()
        logs["ratio"].append(mean_fn(ratio))
        logs["surr1"].append(mean_fn(surr1))
        logs["surr2"].append(mean_fn(surr2))
        logs["perc_surr1"].append(mean_fn((surr1 <= surr2).float()))
        logs["perc_surr2"].append(mean_fn((surr1 < surr2).float()))

        if self.clip_value_loss:
            value_pred_clipped = values + (curr_values - values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (curr_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = (
                0.5 * torch.max(value_losses, value_losses_clipped).mean()
            )
        else:
            value_loss = 0.5 * (returns - curr_values).pow(2).mean()

        self.optimizer.zero_grad()

        value_loss *= self.value_loss_coef
        entropy *= self.entropy_coef
        loss = (
            value_loss
            + action_loss
            - entropy
        )
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return action_loss, value_loss, entropy, logs
