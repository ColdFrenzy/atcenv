import ray
from collections import OrderedDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN

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


class FlightActionMaskRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 lstm_state_size=64,
                 *args,
                 **kwargs):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)

        self.lstm_state_size = lstm_state_size
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
            # (batch_size, seq_len, features) with batch_first=True
            # (seq_len, batch_size, features) with batch_first=False
            self.lstm = nn.LSTM(
                shared_fc_layers[0][-1], self.lstm_state_size, batch_first=True)
        else:
            self.fc_base_critic = FCModel(
                self.input_shape, shared_fc_layers[0])
            self.critic_lstm = nn.LSTM(
                shared_fc_layers[0][-1], self.lstm_state_size, batch_first=True)
            self.fc_base_actor = FCModel(self.input_shape, shared_fc_layers[1])
            self.actor_lstm = nn.LSTM(
                shared_fc_layers[1][-1], self.lstm_state_size, batch_first=True)
        fc_layers = fc_layers

        # =============================================================================
        # CRITIC SUBNETS
        # =============================================================================
        next_inp = self.lstm_state_size
        critic_subnet = OrderedDict()
        for i, fc in enumerate(fc_layers[0]):
            critic_subnet["critic_fc_" + str(i)] = nn.Linear(next_inp, fc)
            critic_subnet["critic_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc
        critic_subnet["critic_out"] = nn.Linear(next_inp, 1)
        critic_subnet["critic_out_activ"] = nn.Tanh()

        # =============================================================================
        # ACTOR SUBNETS
        # =============================================================================
        next_inp = self.lstm_state_size
        actor_subnet = OrderedDict()
        for i, fc in enumerate(fc_layers[1]):
            actor_subnet["actor_fc_" + str(i)] = nn.Linear(next_inp, fc)
            actor_subnet["actor_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc

        actor_subnet["actor_out"] = nn.Linear(next_inp, self.act_shape)
        actor_subnet["actor_out_activ"] = nn.Tanh()

        self.actor_subnet = nn.Sequential(actor_subnet)
        self.critic_subnet = nn.Sequential(critic_subnet)

        # Holds the current "base" output (before logits layer).
        self._features = None
        self._values = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        # View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        if self.share_weights:
            h = [
                self.fc_base.fc_base_model[-2].weight.new(1,
                                                          self.lstm_state_size).zero_().squeeze(0),
                self.fc_base.fc_base_model[-2].weight.new(1,
                                                          self.lstm_state_size).zero_().squeeze(0)
            ]
        else:
            h = [
                self.fc_base_critic.fc_base_model[-2].weight.new(1,
                                                                 self.lstm_state_size).zero_().squeeze(0),
                self.fc_base_critic.fc_base_model[-2].weight.new(1,
                                                                 self.lstm_state_size).zero_().squeeze(0),
                self.fc_base_actor.fc_base_model[-2].weight.new(
                    1, self.lstm_state_size).zero_().squeeze(0),
                self.fc_base_actor.fc_base_model[-2].weight.new(
                    1, self.lstm_state_size).zero_().squeeze(0)
            ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        # return self.base_model.value_function()
        # if self.share_weights:
        #     return torch.reshape(self.critic_subnet(self._features), [-1])
        # else:
        value = torch.reshape(self.critic_subnet(self._values), [-1])
        return value

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        obs = restore_original_dimensions(
            inputs, self.obs_space, "torch")
        action_mask = obs["action_mask"]
        obs_state = torch.cat(
            [obs[elem] for elem in obs.keys() if elem != "action_mask"], dim=2)

        if self.share_weights:
            x = self.fc_base.forward(obs_state)
            # self._features = x.reshape(x.shape[0], -1)
            self._features, [h, c] = self.lstm(
                x, [torch.unsqueeze(state[0], 0),
                    torch.unsqueeze(state[1], 0)])
            action_logits = self.actor_subnet(self._features)
            # inf_mask return a 0 value if the action is valid and a big negative
            # value if it is invalid. Example:
            # [0, 0, -3.4+38, 0, -3.4+38, 0, 0]
            inf_mask = torch.max(torch.log(action_mask),
                                 torch.full(action_mask.shape, torch.finfo(torch.float32).min))
            # The new logits have an extremely low value for invalid actions, that
            # is then cut to zero during the softmax computation
            new_action_logits = action_logits + inf_mask
            return new_action_logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]
        else:
            self._features, [act_h, act_c] = self.actor_lstm(
                self.fc_base_actor(obs_state), [torch.unsqueeze(state[2], 0),
                                                torch.unsqueeze(state[3], 0)])
            self._values, [crit_h, crit_c] = self.critic_lstm(
                self.fc_base_critic(obs_state), [torch.unsqueeze(state[0], 0),
                                                 torch.unsqueeze(state[1], 0)])

            action_logits = self.actor_subnet(self._features)
            # inf_mask return a 0 value if the action is valid and a big negative
            # value if it is invalid. Example:
            # [0, 0, -3.4+38, 0, -3.4+38, 0, 0]
            inf_mask = torch.max(torch.log(action_mask),
                                 torch.full(action_mask.shape, torch.finfo(torch.float32).min))
            # The new logits have an extremely low value for invalid actions, that
            # is then cut to zero during the softmax computation
            new_action_logits = action_logits + inf_mask

            return new_action_logits, [torch.squeeze(crit_h, 0), torch.squeeze(crit_c, 0), torch.squeeze(act_h, 0), torch.squeeze(act_c, 0)]
