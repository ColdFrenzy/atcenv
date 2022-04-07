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
    args["episodes"] = 1
    args.env["min_area"] = 50 * 50
    args.env["max_area"] = 100*100
    args.env["num_flights"] = 3
    args.env["reward_as_dict"] = True
    # init environment
    env = RayWrapper(vars(args.env))
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
        env.flight_env.flights[0].position._set_coords(0., 0.)
        env.flight_env.flights[1].position._set_coords(15000., 15000.)
        env.flight_env.flights[1].track = math.pi/2
        env.flight_env.flights[2].position._set_coords(80000., 15000.)
        env.flight_env.flights[2].track = -math.pi/2
        for i in range(100):
            # perform step with dummy action
            # obs, rew, done, info = env.step(random_policy())
            obs, rew, done, info = env.step(no_move_policy)
            # rews[0]["distance_from_target_rew"] += rew[0]["distance_from_target_rew"]
            # rews[0]["distance_from_traj_rew"] += rew[0]["distance_from_traj_rew"]
            # rews[0]["angle_changed_rew"] += rew[0]["angle_changed_rew"]
            # rews[0]["target_reached_rew"] += rew[0]["target_reached_rew"]
            # print(f"Episode {e}, step {counter} rewards for agent 0:")
            # print(f"distance from target: {rew[0]['distance_from_target_rew']}\naccelleration reward: {rew[0]['accelleration_rew']}\ndistance from trajectory: {rew[0]['distance_from_traj_rew']}\nangle changed reward: {rew[0]['angle_changed_rew']}\ntarget reached reward: {rew[0]['target_reached_rew']}\n")
            print("step {i}")
            for agent_id in obs.keys():
                print(obs[agent_id]["agents_in_fov"])
            # print(f"Episode {e}, step {counter} rewards: {rew}")
            # print(f"Episode {e}, step {counter} done info: {done}")
            counter += 1
            env.render(mode="human")
            # rews += sum(rew.values())/len(rew.keys())
            time.sleep(0.05)
            # input("Press Enter")

        # close rendering
        # print(rews/100)
    env.close()
