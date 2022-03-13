import argparse
import inspect
import os
import uuid
import numpy as np
import itertools as it

import torch


class Params:
    unique_id = str(uuid.uuid1())[:8]

    # =============================================================================
    # DIRECTORIES
    # =============================================================================
    WORKING_DIR = os.getcwd().split("atcenv")[0]
    WORKING_DIR = os.path.join(WORKING_DIR, "atcenv", "atcenv")
    LOG_DIR = os.path.join(WORKING_DIR, "log_dir")
    EVAL_DIR = os.path.join(LOG_DIR, "eval")
    WANDB_DIR = os.path.join(LOG_DIR, "wandb")
    TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
    MODEL_LOG_DIR = os.path.join(LOG_DIR, "model_log")
    MODEL_LOGGER_FILE = os.path.join(
        MODEL_LOG_DIR, "model_log.log")

    # =============================================================================
    # TRAINING
    # =============================================================================
    debug = False
    visible = False
    device = torch.device("cuda")
    epochs = 3000
    # number of learning iterations that the algorithm does on the same batch
    # of trajectories (trajectories are shuffled at each iteration)
    learning_epochs = 3
    # number of elements on which the algorithm performs a learning step
    minibatch = 32  # 64
    # number of episodes to collect in each rollout
    num_episodes = 2  # 4
    batch_size = 4
    framework = "torch"
    epochs = 10

    # =============================================================================
    # WANDB LOGS
    # =============================================================================
    use_wandb = False
    train_log_steps = 5
    val_log_step = 5
    project_name = "atc_challenge"
    opts = {},

    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    num_flights = 10
    dt = 5.
    max_area = 200. * 200.
    min_area = 125. * 125.
    max_speed = 500.
    min_speed = 400
    max_episode_len = 50  # 300
    min_distance = 5.
    distance_init_buffer = 5.
    max_agent_seen = 3
    yaw_angles = [-5.0, 0.0, 5.0]
    accelleration = [-5.0, 0.0, 5.0]
    action_list = list(it.product(
        range(len(yaw_angles)), range(len(accelleration))))

    # =============================================================================
    # MULTIAGENT
    # =============================================================================
    param_sharing = False

    # =============================================================================
    #  OPTIMIZER
    # =============================================================================
    lr = 3e-4
    alpha = 0.99
    max_grad_norm = 5
    eps = 1e-5

    # =============================================================================
    # ALGO PARAMETERS
    # =============================================================================
    gamma = 0.998
    gae_lambda = 0.95
    ppo_clip_param = 0.1
    clip_value_loss = False
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    vf_clip_param = 10.0
    # Loss
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set actor and critic networks share weights (from rllib ppo
    # implementation)
    value_loss_coef = 1  # [0.5, 1]
    entropy_coef = 0.01  # [0.5, 0.1]

    # =============================================================================
    # MODEL PARAMETERS
    # =============================================================================
    # if to use the same architecture for the actor-critic
    share_weights = False
    # fc_layers is a tuple of 1 or 2 elements. If len(fc_layers)==1 we use
    # the same structure for both actor and critic network, otherwise the first
    # element are the layers of the critic and the second are the layers of
    # the actor. if share_weights=True we use the first element as shared architecture.
    shared_fc_layers = ([128, 64, 32],)
    fc_layers = ([], [])
    # use recurrent neural network
    use_recurrent = False
    # size of the hidden layers. Used if use_residual = True
    hidden_size = 64

    # =============================================================================
    # EVALUATION
    # =============================================================================
    log_step = 500
    checkpoint_freq = 50
    restore = True
    resume_training = False
    max_checkpoint_keep = 10

    def __init__(self):
        self.__initialize_dirs()
        self.__parse_args()

    def __initialize_dirs(self):
        """
        Initialize all the directories  listed above
        :return:
        """
        variables = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        for var in variables:
            if var.lower().endswith("dir"):
                path = getattr(self, var)
                if not os.path.exists(path):
                    print(f"Mkdir {path}")
                    os.makedirs(path)
            # change values based on argparse

    def __parse_args(self):
        """
        Use argparse to change the default values in the param class
        """

        att = self.__get_attributes()

        """Create the parser to capture CLI arguments."""
        parser = argparse.ArgumentParser()

        # for every attribute add an arg instance
        for k, v in att.items():
            if isinstance(v, bool):
                parser.add_argument(
                    "-" + k.lower(),
                    action="store_true",
                    default=v,
                )
            else:
                parser.add_argument(
                    "--" + k.lower(),
                    type=type(v),
                    default=v,
                )

        args, unk = parser.parse_known_args()
        for k, v in vars(args).items():
            self.__setattr__(k, v)

    def __get_attributes(self):
        """
        Get a dictionary for every attribute that does not have "filter_str" in it
        :return:
        """

        # get every attribute
        attributes = inspect.getmembers(self)
        # filter based on double underscore
        filter_str = "__"
        attributes = [elem for elem in attributes if filter_str not in elem[0]]
        # convert to dict
        attributes = dict(attributes)

        return attributes

    def get_rollout_configs(self):
        rollout_configs = dict(
            num_steps=self.max_episode_len*self.num_episodes,
            obs_shape=self.max_agent_seen*2,
            num_actions=len(self.action_list),
            num_agents=self.num_flights,
            minibatch=self.minibatch,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        return rollout_configs

    def get_ppo_configs(self):
        ppo_configs = dict(
            lr=self.lr,
            eps=self.eps,
            entropy_coef=self.entropy_coef,
            value_loss_coef=self.value_loss_coef,
            clip_param=self.ppo_clip_param,
            clip_value_loss=self.clip_value_loss,
            max_grad_norm=self.max_grad_norm,
            device=self.device
        )
        return ppo_configs

    def get_env_config(self):
        env_config = dict(
            num_flights=self.num_flights,
            dt=self.dt,
            max_area=self.max_area,
            min_area=self.min_area,
            max_speed=self.max_speed,
            min_speed=self.min_speed,
            max_episode_len=self.max_episode_len,
            min_distance=self.min_distance,
            distance_init_buffer=self.distance_init_buffer,
            max_agent_seen=self.max_agent_seen,
            yaw_angles=self.yaw_angles,
            accelleration=self.accelleration,
            action_list=self.action_list
        )
        return env_config

    def get_model_configs(self):
        model_configs = dict(
            input_shape=self.max_agent_seen*2,
            action_shape=len(self.action_list),
            share_weights=self.share_weights,
            shared_fc_layers=self.shared_fc_layers,
            fc_layers=self.fc_layers,
            use_recurrent=self.use_recurrent,
            hidden_size=self.hidden_size,
        )
        return model_configs

    def get_callback_configs(self):
        callback_configs = dict(
            train_log_steps=self.train_log_steps,
            val_log_step=self.val_log_step,
            project=self.project_name,
            opts=self.opts,
            horizon=self.max_episode_len,
            out_dir=self.WANDB_DIR,
            debug=self.debug,
            hyperparams=self.__dict__
        )
        return callback_configs

    def get_trainer_configs(self):
        trainer_configs = dict(
            num_agents=self.num_flights,
            num_steps=self.max_episode_len,
            num_episodes=self.num_episodes,
            device=self.device,
            learning_epochs=self.learning_epochs
        )
        return trainer_configs
