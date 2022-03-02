def model_configs(args):
    configs = {
        "model": {

            # === Built-in options ===
            # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
            # These are used if no custom model is specified and the input space is 1D.
            # Number of hidden layers to be used.
            "fcnet_hiddens": [256, 256],
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            "fcnet_activation": "tanh",

            # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
            # These are used if no custom model is specified and the input space is 2D.
            # Filter config: List of [out_channels, kernel, stride] for each filter.
            # Example:
            # Use None for making RLlib try to find a default filter setup given the
            # observation space.
            "conv_filters": None,
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            "conv_activation": "relu",

            # Some default models support a final FC stack of n Dense layers with given
            # activation:
            # - Complex observation spaces: Image components are fed through
            #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
            #   everything is concated and pushed through this final FC stack.
            # - VisionNets (CNNs), e.g. after the CNN stack, there may be
            #   additional Dense layers.
            # - FullyConnectedNetworks will have this additional FCStack as well
            # (that's why it's empty by default).
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": "relu",

            # For DiagGaussian action distributions, make the second half of the model
            # outputs floating bias variables instead of state-dependent. This only
            # has an effect is using the default fully connected net.
            "free_log_std": False,
            # Whether to skip the final linear layer used to resize the hidden layer
            # outputs to size `num_outputs`. If True, then the last hidden layer
            # should already match num_outputs.
            "no_final_linear": False,
            # Whether layers should be shared for the value function.
            "vf_share_layers": True,

            # == LSTM ==
            # Whether to wrap the model with an LSTM.
            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 20,
            # Size of the LSTM cell.
            "lstm_cell_size": 256,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": False,
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": False,
            # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            "_time_major": False,

            # == Attention Nets (experimental: torch-version is untested) ==
            # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
            # wrapper Model around the default Model.
            "use_attention": False,
            # The number of transformer units within GTrXL.
            # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
            # b) a position-wise MLP.
            "attention_num_transformer_units": 1,
            # The input and output size of each transformer unit.
            "attention_dim": 64,
            # The number of attention heads within the MultiHeadAttention units.
            "attention_num_heads": 1,
            # The dim of a single head (within the MultiHeadAttention units).
            "attention_head_dim": 32,
            # The memory sizes for inference and training.
            "attention_memory_inference": 50,
            "attention_memory_training": 50,
            # The output dim of the position-wise MLP.
            "attention_position_wise_mlp_dim": 32,
            # The initial bias values for the 2 GRU gates within a transformer unit.
            "attention_init_gru_gate_bias": 2.0,
            # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
            "attention_use_n_prev_actions": 0,
            # Whether to feed r_{t-n:t-1} to GTrXL.
            "attention_use_n_prev_rewards": 0,

            # == Atari ==
            # Set to True to enable 4x stacking behavior.
            "framestack": True,
            # Final resized frame dimension
            "dim": 84,
            # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
            "grayscale": False,
            # (deprecated) Changes frame to range from [-1, 1] if true
            "zero_mean": True,

            # === Options for custom models ===
            # Name of a custom model to use
            "custom_model": None,
            # Extra options to pass to the custom classes. These will be available to
            # the Model's constructor in the model_config field. Also, they will be
            # attempted to be passed as **kwargs to ModelV2 models. For an example,
            # see rllib/models/[tf|torch]/attention_net.py.
            "custom_model_config": {},
            # Name of a custom action distribution to use.
            "custom_action_dist": None,
            # Custom preprocessors are deprecated. Please use a wrapper class around
            # your environment instead to preprocess observations.
            "custom_preprocessor": None,

        }}

    return configs


def ppo_configs(args):
    configs = {
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 256,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-4,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        "model": {
            # Share layers for value function. If you set this to True, it's
            # important to tune vf_loss_coeff.
            "vf_share_layers": False,
        },
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 100.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",

    }

    return configs


def resources_configs(args):
    configs = {
        "num_workers": 0 if args.debug else 4,
        "num_gpus": 0 if args.debug else 1,
    }

    return configs


def eval_configs(args):
    configs = {
        # Evaluate once per training iteration.
        "evaluation_interval": 1,
        # Run evaluation on (at least) two episodes
        "evaluation_duration": 2,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Store videos in this relative directory here inside
            # the default output dir (~/ray_results/...).
            # Alternatively, you can specify an absolute path.
            # Set to True for using the default output dir (~/ray_results/...).
            # Set to False for not recording anything.
            # "record_env": "videos",
            # "record_env": "/Users/xyz/my_videos/",
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": False,
        },
    }

    return configs


def multi_agent_configs(args, obs_space, action_space):
    configs = {
        "multiagent": {
            "policies": {
                "default": (None, obs_space, action_space, {}),

            },
            "policy_mapping_fn": lambda x: "default",
        },
    }

    return configs
