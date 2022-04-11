"""
Example
"""
from typing import OrderedDict
from collections import defaultdict
from tqdm import tqdm
import time
from atcenv.common.utils import parse_args
import random
import math

from atcenv.envs import FlightEnv, RayWrapper

# random.seed(42)
random.seed(11)


def random_action(flight_ids, action_space):

    def inner():

        actions = {}
        for f_id in flight_ids:
            actions[f_id] = action_space.sample()
        return actions

    return inner


if __name__ == "__main__":

    # parse arguments
    args = parse_args()
    # custom values for a run
    args.env["reward_as_dict"] = True
    kwargs = vars(args.env)
    config = kwargs.pop("config")
    # init environment
    env = RayWrapper(config, **kwargs)
    env.flight_env.screen_size = 600
    obs = env.reset()

    random_policy = random_action(
        env.flight_env.flights.keys(), env.action_space)
    no_move_policy = {}
    move_left_policy = {}
    move_right_policy = {}
    for flight_id in env.flight_env.flights.keys():
        # don't accellerate and don't change angle
        no_move_policy[flight_id] = 4
        move_left_policy[flight_id] = 1
        move_right_policy[flight_id] = 8
    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()
        rews = {agent: defaultdict(float) for agent in env._agent_ids}

        # set done status to false
        done = {"__all__": False}

        # execute one episode
        counter = 0
        # while not done["__all__"]:
        # 15000., 15000.
        env.flight_env.flights[0].position._set_coords(-100000000., 15000.)
        env.flight_env.flights[0].track = math.pi/2
        env.flight_env.flights[1].position._set_coords(80000., 15000.)
        env.flight_env.flights[1].track = env.flight_env.flights[1].bearing
        env.flight_env.flights[2].position._set_coords(-1000000, 0.0)
        for i in range(50):
            # perform step with dummy action
            # obs, rew, done, info = env.step(random_policy())
            obs, rew, done, info = env.step(move_left_policy)
            # rews[0]["distance_from_target_rew"] += rew[0]["distance_from_target_rew"]
            # rews[0]["distance_from_traj_rew"] += rew[0]["distance_from_traj_rew"]
            # # rews[0]["angle_changed_rew"] += rew[0]["angle_changed_rew"]
            # rews[0]["target_reached_rew"] += rew[0]["target_reached_rew"]
            # rews[0]["drift_rew"] += rew[0]["drift_rew"]
            # print(f"Episode {e}, step {counter} rewards for agent 0:")
            # print(f"distance from target: {rew[0]['distance_from_target_rew']}\naccelleration reward: {rew[0]['accelleration_rew']}\ndistance from trajectory: {rew[0]['distance_from_traj_rew']}\nangle changed reward: {rew[0]['angle_changed_rew']}\ntarget reached reward: {rew[0]['target_reached_rew']}\n")
            print("step {i}")
            for agent_id in obs.keys():
                print(f"flight {agent_id} rewards: ")
                print(f"distance from target: {rew[agent_id]['distance_from_target_rew']} " +
                      f"drift penalty: {rew[agent_id]['drift_rew']} " +
                      f"target reached rew: {rew[agent_id]['target_reached_rew']}"
                      )
                print(obs[agent_id]["agents_in_fov"])
            # print(f"Episode {e}, step {counter} rewards: {rew}")
            # print(f"Episode {e}, step {counter} done info: {done}")
            counter += 1
            env.render(mode="human")
            # rews += sum(rew.values())/len(rew.keys())
            time.sleep(0.05)
            # input("Press Enter")

    env.close()
