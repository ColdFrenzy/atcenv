import ray
from collections import OrderedDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.utils import try_import_torch
from ray.rllib.models.modelv2 import restore_original_dimensions

torch, nn = try_import_torch()


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
        # super(FCModel, self).__init__()
        nn.Module.__init__(self)
        model = OrderedDict()
        next_inp = input_shape
        for i, layers in enumerate(fc_layers):
            model["fc_"+str(i)] = nn.Linear(next_inp, layers)
            model["fc_"+str(i)+"_activ"] = activ()
            next_inp = layers

        self.fc_base_model = nn.Sequential(model)

    def forward(self, inputs):
        """forward step
        """
        return self.fc_base_model(inputs)


class FlightActionMaskModel(TorchModelV2, nn.Module):
    """Parametric action model that handles the dot product and masking.
    If you invert the inheritance it doesn't work.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        *args,
        **kwargs
    ):

        # super(FlightActionMaskModel, self).__init__(
        #     obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=model_config, name=name  # ,*args, **kwargs
        # )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)

        self.share_weights = kwargs["share_weights"]
        shared_fc_layers = kwargs["shared_fc_layers"]
        fc_layers = kwargs["fc_layers"]
        self.name = name
        action_mask_len = obs_space.original_space["action_mask"].shape[0]
        self.input_shape = obs_space.shape[0] - action_mask_len
        self.act_shape = action_space.n
        if len(shared_fc_layers) == 1:
            shared_fc_layers = (shared_fc_layers[0], shared_fc_layers[0])
        if len(fc_layers) == 1:
            fc_layers = (fc_layers[0], fc_layers[0])

        if self.share_weights:
            self.fc_base = FCModel(self.input_shape, shared_fc_layers[0])
        else:
            self.fc_base_critic = FCModel(
                self.input_shape, shared_fc_layers[0])

            self.fc_base_actor = FCModel(self.input_shape, shared_fc_layers[1])

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

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    # @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Args:
            input_dict (dict): dictionary of input tensors, including "obs",
                "obs_flat", "prev_action", "prev_reward", "is_training",
                "eps_id", "agent_id", "infos", and "t".
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths
        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, num_outputs], and the new RNN state.
        """
        obs = restore_original_dimensions(
            input_dict["obs"], self.obs_space, "torch")
        action_mask = obs["action_mask"]
        obs_state = torch.cat(
            [obs[elem] for elem in obs.keys() if elem != "action_mask"], dim=1)

        if self.share_weights:
            x = self.fc_base.forward(obs_state)
            self._features = x.reshape(x.shape[0], -1)
            action_logits = self.actor(x)
        else:
            self._last_flat_in = obs_state.reshape(obs_state.shape[0], -1)

            x = self.fc_base_actor.forward(obs_state)
            action_logits = self.actor(x)

        # inf_mask return a 0 value if the action is valid and a big negative
        # value if it is invalid. Example:
        # [0, 0, -3.4+38, 0, -3.4+38, 0, 0]
        inf_mask = torch.max(torch.log(action_mask),
                             torch.full(action_mask.shape, torch.finfo(torch.float32).min))
        # The new logits have an extremely low value for invalid actions, that
        # is then cut to zero during the softmax computation
        new_action_logits = action_logits + inf_mask

        return new_action_logits, state

    # @override(TorchModelV2)
    def value_function(self):
        # return self.base_model.value_function()
        if self.share_weights:
            return self.critic(self._features).squeeze(1)
        else:
            return self.critic(self.fc_base_critic.forward(self._last_flat_in)).squeeze(1)
