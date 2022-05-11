"""
Example
"""

if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv.envs.FlightEnv import FlightEnv
    from atcenv.configs.Params import Params
    import time
    from tqdm import tqdm

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('-debug', action="store_true", default=False)
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    params = Params()

    # init environment
    env = FlightEnv(**params.get_env_config())

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()
        env.render("human")
        # set done status to false
        done = {"__all__": False}
        counter = 0
        actions = {flight_id: 4 for flight_id in env.flights.keys()}
        while not done["__all__"]:
            obs, rew, done, info = env.step(actions)
            counter += 1
            env.render("human")
            time.sleep(0.05)

        # close rendering
        env.close()
