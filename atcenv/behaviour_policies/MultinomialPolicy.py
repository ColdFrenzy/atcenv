import torch.nn as nn
import torch.nn.functional as F


class MultinomialPolicy:
    """Wrapper for a neural network model. It select the action based on
    model output probability, following a multinomial distribution
    """

    def __init__(self, model):
        """__init__ method.

        Parameters
        ----------
        model : nn.Module
            neural network model used for training
        Returns
        -------
        None.

        """
        self.model = model

    def act(self, inputs, deterministic=False):
        """select action using a multinomial distribution if deterministic=False.

        Parameters
        ----------
        inputs : torch.Tensor
            [batch_size, input_shape]
        deterministic : bool, optional
            if to use a deterministic policy or multinomial

        """
        # normalize the input outside
        action_logit, value = self.model(inputs)
        action_probs = F.softmax(action_logit, dim=-1)

        if deterministic:
            action = action_probs.max(-1)[1]
        else:
            action = action_probs.multinomial(1)

        log_actions_prob = F.log_softmax(action_logit, dim=-1).squeeze()

        value = float(value)
        action = int(action)

        return value, action, log_actions_prob
