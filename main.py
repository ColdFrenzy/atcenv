"""
Example
"""
from atcenv.common.utils import parse_args
import random

from atcenv.envs import FlightEnv

random.seed(42)
import time
from tqdm import tqdm

if __name__ == "__main__":


    # parse arguments
    args = parse_args()

    # init environment
    env = FlightEnv(**vars(args.env))

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()

        # set done status to false
        done = False

        # execute one episode
        # while not done:
        for i in range(100):
            # perform step with dummy action
            rew, obs, done, info = env.step({})
            env.render()
            time.sleep(0.05)

        # close rendering
        env.close()
