import time
import copy
import numpy as np
from ray.rllib.utils import try_import_torch
from atcenv.common.custom_gym_monitor import CustomGymMonitor
import random
torch, nn = try_import_torch()


def FlightCustomEval(env, policy_to_evaluate, video_dir):
    """custom evaluation function. In this function we execute 2 policies on the
    same copy of the environment to compare their results.
    Args:
        env: env to evaluate
        policy_to_evaluate: policy to evaluate
        standard_policy: standard policy used for comparison
    Returns:
        metrics (dict): evaluation metrics dict.
        next_level (bool): true if the policy is good enough to skip to the next level
    """

    metrics = {}
    # the random seed is used to create the same copy of the environment
    seed_indx = random.randint(0, 10000)
    env = CustomGymMonitor(
        env=env,
        directory=video_dir,
        video_callable=lambda x: True,
        force=False)
    random.seed(seed_indx)
    obs = env.reset()
    done = {"__all__": False}
    counter = 0
    num_collisions = 0
    # no move policy
    actions = {flight_id: 4 for flight_id in env.flight_env.flights.keys()}
    while not done["__all__"]:
        rew, obs, done, info = env.step(actions)
        num_collisions += len(env.flight_env.conflicts)
        counter += 1
    print(f"No Move Policy collisions: {num_collisions}")
    done = {"__all__": False}
    num_collisions2 = 0
    # reset the seed and the environment
    random.seed(seed_indx)
    obs = env.reset()
    model = policy_to_evaluate.model
    h = {flight_id: model.get_initial_state()
         for flight_id in env.flight_env.flights.keys()}
    seq_len = torch.tensor([1.])
    actions = {flight_id: None for flight_id in env.flight_env.flights.keys()}

    with torch.no_grad():
        while not done["__all__"]:
            # add both the batch and the time dim to the observation returned by the env
            for flight_id in env.flight_env.flights.keys():
                for elem in obs[flight_id].keys():
                    obs[flight_id][elem] = torch.from_numpy(
                        obs[flight_id][elem]).float().unsqueeze(0).unsqueeze(0)
                for elem in range(len(h[flight_id])):
                    if len(h[flight_id][elem].shape) < 2:
                        h[flight_id][elem] = h[flight_id][elem].unsqueeze(0)
            for flight_id in env.flight_env.flights.keys():
                actions[flight_id], h[flight_id] = model.forward_rnn(
                    obs[flight_id], h[flight_id], seq_len)
                actions[flight_id] = torch.argmax(actions[flight_id])
            rew, obs, done, info = env.step(actions)
            num_collisions2 += len(env.flight_env.conflicts)
    print(f"Default Policy collisions: {num_collisions2}")
    env.close()
    # the closest to zero, the better it is
    metrics["num_of_collisions_with_no_actions"] = num_collisions
    metrics["num_of_collisions_with_model"] = num_collisions2
    metrics["num_of_collisions_ratio"] = num_collisions2 / \
        num_collisions if num_collisions != 0 else 0.0
    # the closest to one, the better it is. Since the custom
    metrics["lenght_of_episode_with_no_actions"] = counter
    metrics["lenght_of_episode_with_model"] = env.i
    metrics["lenght_episode_ratio"] = counter / env.i if env.i != 0 else 0.0
    next_level = False
    # If the episode ends before the max_episode_len, it means that the level
    # was succesfully completed
    if env.flight_env.i < env.flight_env.max_episode_len:
        next_level = True

    return metrics, next_level
