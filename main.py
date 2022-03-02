"""
Example
"""
from atcenv.common.utils import parse_args
import random

from atcenv.envs import FlightEnv

random.seed(42)
import time
from tqdm import tqdm


def random_action(flight_ids):

    def inner():

        actions={}
        for f_id in flight_ids:
            actions[f_id]=dict(
                accel=random.random(),
                track=random.random(),
            )
        return actions

    return inner

if __name__ == "__main__":


    # parse arguments
    args = parse_args()

    # init environment
    env = FlightEnv(**vars(args.env))
    obs = env.reset()

    random_policy=random_action(env.flights.keys())

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()
        rews=0

        # set done status to false
        done = False

        # execute one episode
        # while not done:
        for i in range(100):
            # perform step with dummy action
            rew, obs, done, info = env.step(random_policy())
            env.render()
            rews+=sum(rew.values())/len(rew.keys())
            #time.sleep(0.01)

        # close rendering
        print(rews/100)
        env.close()
