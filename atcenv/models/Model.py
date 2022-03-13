from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


class FCModel(nn.Module):
    def __init__(self, input_shape, fc_layers, activ=nn.Tanh):
        """Simple Fully connected model

        Parameters
        ----------
        input_shape : Tuple
        fc_layers : List[int]
            dimensions of the fully connected layers
        activ: Torch activation
            activation function
        Returns
        -------
        None.

        """
        super(FCModel, self).__init__()
        model = OrderedDict()
        next_inp = input_shape
        for i, layers in enumerate(fc_layers):
            model["fc_"+str(i)] = nn.Linear(next_inp, layers)
            model["fc_"+str(i)+"_activ"] = activ()
            next_inp = layers

        self.model = nn.Sequential(model)

    def forward(self, inputs):
        """forward step
        """
        return self.model(inputs)


class FCActorCriticModel(nn.Module):
    """FCActorCriticModel class.

    An actor critic model
    """

    def __init__(self, input_shape, action_shape, share_weights, shared_fc_layers,
                 fc_layers, use_recurrent, **kwargs):
        super(FCActorCriticModel, self).__init__()

        self.name = "FCModel"
        self.input_shape = input_shape
        self.act_shape = action_shape
        self.share_weights = share_weights
        if len(shared_fc_layers) == 1:
            shared_fc_layers = (shared_fc_layers[0], shared_fc_layers[0])
        if len(fc_layers) == 1:
            fc_layers = (fc_layers[0], fc_layers[0])

        if self.share_weights:
            self.fc_base = FCModel(input_shape, shared_fc_layers[0])
        else:
            self.fc_base_critic = FCModel(input_shape, shared_fc_layers[0])

            self.fc_base_actor = FCModel(input_shape, shared_fc_layers[1])

        fc_layers = fc_layers
        # =============================================================================
        # CRITIC SUBNETS
        # =============================================================================
        next_inp = shared_fc_layers[0][-1]
        critic_subnet = OrderedDict()
        for i, fc in enumerate(fc_layers[0]):
            critic_subnet["critic_fc_" + str(i)] = nn.Linear(next_inp, fc)
            critic_subnet["critic_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc
        critic_subnet["critic_out"] = nn.Linear(next_inp, 1)

        # =============================================================================
        # ACTOR SUBNETS
        # =============================================================================
        if self.share_weights:
            next_inp = shared_fc_layers[0][-1]
        else:
            next_inp = shared_fc_layers[1][-1]
        actor_subnet = OrderedDict()
        for i, fc in enumerate(fc_layers[1]):
            actor_subnet["actor_fc_" + str(i)] = nn.Linear(next_inp, fc)
            actor_subnet["actor_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc

        actor_subnet["actor_out"] = nn.Linear(next_inp, self.act_shape)

        self.actor = nn.Sequential(actor_subnet)
        self.critic = nn.Sequential(critic_subnet)

    def evaluate_actions(self, inputs: torch.Tensor):
        """evaluate_actions method.

        compute the actions logit, value and actions probability by passing
        the actual states to the network. Then computes
        the entropy of the actions
        Parameters
        ----------
        inputs : PyTorch Array
            a 4 dimensional tensor [batch_size, channels, width, height]

        Returns
        -------
        action_logit : Torch.Tensor [batch_size, num_actions]
            output of the ModelFree network before passing it to the softmax
        action_log_probs : torch.Tensor [batch_size, num_actions]
            log_probs of all the actions
        probs : Torch.Tensor [batch_size, num_actions]
            probability of actions given by the network
        value : Torch.Tensor [batch_size,1]
            value of the state given by the network
        entropy : Torch.Tensor [batch_size,1]
            value of the entropy given by the action with index equal to action_indx.
        """

        action_logit, value = self.forward(inputs)
        action_probs = F.softmax(action_logit, dim=1)
        action_log_probs = F.log_softmax(action_logit, dim=1)
        entropy = -(action_probs * action_log_probs).sum(1).mean()

        return value, action_log_probs, entropy

    def to(self, device):
        if self.share_weights:
            self.fc_base.to(device)
        else:
            self.fc_base_critic.to(device)
            self.fc_base_actor.to(device)

        self.critic.to(device)
        self.actor.to(device)

        return self

    def forward(self, inputs):
        """forward method.

        Return the logit and the values of the model.
        Parameters
        ----------
        inputs : torch.Tensor
            [batch_size, num_channels, width, height]

        Returns
        -------
        logit : torch.Tensor
            [batch_size, num_actions]
        value : torch.Tensor
            [batch_size, value]

        """
        if self.share_weights:
            x = self.fc_base.forward(inputs)
            return self.actor(x), self.critic(x)
        else:
            x = self.fc_base_critic.forward(inputs)
            value = self.critic(x)

            x = self.fc_base_actor.forward(inputs)
            action_logits = self.actor(x)

            return action_logits, value
