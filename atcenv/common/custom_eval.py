import numpy as np
from collections import defaultdict
from ray.rllib.utils import try_import_torch
from atcenv.common.custom_gym_monitor import CustomGymMonitor
torch, nn = try_import_torch()


def flight_custom_eval(env, policy_to_evaluate, video_dir):
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
    env = CustomGymMonitor(
        env=env,
        directory=video_dir,
        video_callable=lambda x: True,
        force=False)
    obs, env_config = env.reset(random=True, return_init=True)
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
    obs = env.reset(random=False, config=env_config)
    model = policy_to_evaluate.model
    if model.name == "FlightActionMaskRNNModel":
        h = {flight_id: model.get_initial_state()
             for flight_id in env.flight_env.flights.keys()}
        seq_len = torch.tensor([1.])
    actions = {flight_id: None for flight_id in env.flight_env.flights.keys()}

    with torch.no_grad():
        while not done["__all__"]:
            # add both the batch and the time dim to the observation returned by the env
            if model.name == "FlightActionMaskRNNModel":
                for flight_id in env.flight_env.flights.keys():
                    for elem in obs[flight_id].keys():
                        obs[flight_id][elem] = torch.from_numpy(
                            obs[flight_id][elem]).float().unsqueeze(0).unsqueeze(0)
                    for elem in range(len(h[flight_id])):
                        if len(h[flight_id][elem].shape) < 2:
                            h[flight_id][elem] = h[flight_id][elem].unsqueeze(
                                0)
                for flight_id in env.flight_env.flights.keys():
                    actions[flight_id], h[flight_id] = model.forward_rnn(
                        obs[flight_id], h[flight_id], seq_len)
                    actions[flight_id] = torch.argmax(actions[flight_id])
            elif model.name == "FlightActionMaskModel":
                for flight_id in env.flight_env.flights.keys():
                    for elem in obs[flight_id].keys():
                        obs[flight_id][elem] = torch.from_numpy(
                            obs[flight_id][elem]).float().unsqueeze(0)
                for flight_id in env.flight_env.flights.keys():
                    actions[flight_id], _ = model.forward(
                        obs[flight_id], [], [])
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
    # if env.flight_env.i < env.flight_env.max_episode_len:
    #     next_level = True

    return metrics, next_level


def flight_custom_eval_no_video(env, policy_to_evaluate, num_episode=1):
    """custom evaluation function. In this function we execute 2 policies on the
    same copy of the environment to compare their results. This is the same as the
    FlightCustomEval with the only difference that we don't record video
    Args:
        env: env to evaluate
        policy_to_evaluate: policy to evaluate
        num_episodes: num of episode to evaluate
    Returns:
        metrics (dict): evaluation metrics dict.
        next_level (bool): true if the policy is good enough to skip to the next level
    """
    assert num_episode > 0, "Select at least 1 episode for the evaluation"
    metrics = defaultdict(lambda: 0)
    mean_reward = 0.0
    level_completed = 0
    for episode in range(num_episode):
        obs, env_config = env.reset(random=True, return_init=True)
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
        obs = env.reset(random=False, config=env_config)
        model = policy_to_evaluate.model
        if model.name == "FlightActionMaskRNNModel":
            h = {flight_id: model.get_initial_state()
                 for flight_id in env.flight_env.flights.keys()}
            seq_len = torch.tensor([1.])
        actions = {flight_id: None for flight_id in env.flight_env.flights.keys()}
        episode_rew = 0
        with torch.no_grad():
            while not done["__all__"]:
                # add both the batch and the time dim to the observation returned by the env
                if model.name == "FlightActionMaskRNNModel":
                    for flight_id in env.flight_env.flights.keys():
                        for elem in obs[flight_id].keys():
                            obs[flight_id][elem] = torch.from_numpy(
                                obs[flight_id][elem]).float().unsqueeze(0).unsqueeze(0)
                        for elem in range(len(h[flight_id])):
                            if len(h[flight_id][elem].shape) < 2:
                                h[flight_id][elem] = h[flight_id][elem].unsqueeze(
                                    0)
                    for flight_id in env.flight_env.flights.keys():
                        actions[flight_id], h[flight_id] = model.forward_rnn(
                            obs[flight_id], h[flight_id], seq_len)
                        actions[flight_id] = torch.argmax(actions[flight_id])
                elif model.name == "FlightActionMaskModel":
                    for flight_id in env.flight_env.flights.keys():
                        for elem in obs[flight_id].keys():
                            obs[flight_id][elem] = torch.from_numpy(
                                obs[flight_id][elem]).float().unsqueeze(0)
                    for flight_id in env.flight_env.flights.keys():
                        actions[flight_id], _ = model.forward(
                            obs[flight_id], [], [])
                        actions[flight_id] = torch.argmax(actions[flight_id])
                rew, obs, done, info = env.step(actions)
                for flight_id in rew.keys():
                    episode_rew += rew[flight_id]
                num_collisions2 += len(env.flight_env.conflicts)
        mean_reward += episode_rew
        print(f"Default Policy collisions: {num_collisions2}")
        # the closest to zero, the better it is
        metrics["num_of_collisions_with_no_actions"] += num_collisions
        metrics["num_of_collisions_with_model"] += num_collisions2
        metrics["num_of_collisions_ratio"] += num_collisions2 / \
            num_collisions if num_collisions != 0 else 0.0
        # the closest to one, the better it is. Since the custom
        metrics["lenght_of_episode_with_no_actions"] += counter
        metrics["lenght_of_episode_with_model"] += env.i
        metrics["lenght_episode_ratio"] += counter / \
            env.i if env.i != 0 else 0.0
        # If the episode ends before the max_episode_len, it means that the level
        # was succesfully completed
        level_completed += 1 if env.flight_env.i < env.flight_env.max_episode_len else 0
    for elem in metrics.keys():
        metrics[elem] = metrics[elem]/num_episode
    metrics["mean_reward"] = mean_reward/num_episode
    metrics = dict(metrics)
    next_level = False
    if level_completed == num_episode:
        next_level = True

    return metrics, next_level
