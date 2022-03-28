"""
Example
"""
from typing import OrderedDict
from collections import defaultdict
from tqdm import tqdm
import time
from atcenv.common.utils import parse_args
import random

from atcenv.envs import FlightEnv, RayWrapper

random.seed(42)


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
    args["episodes"] = 5
    args.env["min_area"] = 50 * 50
    args.env["max_area"] = 100*100
    args.env["num_flights"] = 3
    args.env["reward_as_dict"] = True
    # init environment
    env = RayWrapper(vars(args.env))
    env.screen_size = 600
    obs = env.reset()

    random_policy = random_action(env.flights.keys(), env.action_space)

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()
        rews = {agent: defaultdict(float) for agent in env._agent_ids}

        # set done status to false
        done = {"__all__": False}

        # execute one episode
        counter = 0
        while not done["__all__"]:
            # for i in range(100):
            # perform step with dummy action
            obs, rew, done, info = env.step(random_policy())
            rews[0]["distance_from_target_rew"] += rew[0]["distance_from_target_rew"]
            rews[0]["accelleration_rew"] += rew[0]["accelleration_rew"]
            rews[0]["distance_from_traj_rew"] += rew[0]["distance_from_traj_rew"]
            rews[0]["angle_changed_rew"] += rew[0]["angle_changed_rew"]
            rews[0]["target_reached_rew"] += rew[0]["target_reached_rew"]
            print(f"Episode {e}, step {counter} rewards for agent 0:")
            print(f"distance from target: {rew[0]['distance_from_target_rew']}\naccelleration reward: {rew[0]['accelleration_rew']}\ndistance from trajectory: {rew[0]['distance_from_traj_rew']}\nangle changed reward: {rew[0]['angle_changed_rew']}\ntarget reached reward: {rew[0]['target_reached_rew']}\n")
            counter += 1
            env.render(mode="human")
            # rews += sum(rew.values())/len(rew.keys())
            time.sleep(0.05)

        # close rendering
        print(rews/100)
    env.close()
