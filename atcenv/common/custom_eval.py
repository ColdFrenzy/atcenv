from atcenv.envs.FlightEnv import FlightEnv
import gym
from ray.rllib.utils import try_import_torch
from copy import deepcopy
torch, nn = try_import_torch()


def CurriculumCustomEval(trainer, eval_workers):
    """custom evaluation function. In this function we execute 2 policies on the
    same copy of the environment to compare their results.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    metrics = {}
    # reset the env and make a clone of it
    video_dir = trainer.config["evaluation_config"]["record_env"]
    env = FlightEnv()
    if video_dir:
        env = gym.wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda x: True,
            force=True)
    obs1 = env.reset()
    obs2 = deepcopy(obs1)
    cloned_env = deepcopy(env)
    model = trainer.get_policy("default").model
    no_move_policy = {}
    for flight_id in cloned_env.flights.keys():
        # don't accellerate and don't change angle
        no_move_policy[flight_id] = 4
    done = {"__all__": False}
    counter = 0
    num_collisions = 0
    while not done["__all__"]:
        # perform step with dummy action
        rew, obs, done, info = cloned_env.step(no_move_policy)
        num_collisions += len(cloned_env.conflicts)
        counter += 1
        # cloned_env.render(mode="human")
        # time.sleep(0.05)
    # cloned_env.close()
    # print(
    #     f"Number of collisions in this episode using the no_move_policy: {num_collisions}")
    done = {"__all__": False}
    num_collisions2 = 0
    h = {flight_id: model.get_initial_state()
         for flight_id in env.flights.keys()}
    seq_len = torch.tensor([1.])
    actions = {flight_id: None for flight_id in env.flights.keys()}

    with torch.no_grad():
        while not done["__all__"]:
            # add both the batch and the time dim to the observation returned by the env
            for flight_id in env.flights.keys():
                for elem in obs2[flight_id].keys():
                    obs2[flight_id][elem] = torch.from_numpy(
                        obs2[flight_id][elem]).float().unsqueeze(0).unsqueeze(0)
                for elem in range(len(h[flight_id])):
                    if len(h[flight_id][elem].shape) < 2:
                        h[flight_id][elem] = h[flight_id][elem].unsqueeze(0)
            for flight_id in env.flights.keys():
                actions[flight_id], h[flight_id] = model.forward_rnn(
                    obs2[flight_id], h[flight_id], seq_len)
                actions[flight_id] = torch.argmax(actions[flight_id])
            rew, obs2, done, info = env.step(actions)
            print(actions)
            num_collisions2 += len(env.conflicts)
        #     env.render(mode="human")
        #     time.sleep(0.05)
        # env.close()
        # print(
        #     f"Number of collisions in this episode using the default policy:  {num_collisions}")
    env._close_video_recorder()
    # the closest to zero, the better it is
    metrics["num_of_collisions_with_no_actions"] = num_collisions
    metrics["num_of_collisions_with_model"] = num_collisions2
    metrics["num_of_collisions_ratio"] = num_collisions2 / \
        num_collisions if num_collisions != 0 else 0.0
    # the closest to one, the better it is. Since the custom
    metrics["lenght_of_episode_with_no_actions"] = counter
    metrics["lenght_of_episode_with_model"] = env.i
    metrics["lenght_episode_ratio"] = counter / env.i if env.i != 0 else 0.0

    return metrics
