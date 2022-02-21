from jsonargparse import ArgumentParser

from jsonargparse import ActionConfigFile

from atcenv.envs import FlightEnv


def parse_args():
    """

    """
    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(FlightEnv, 'env')

    # parse arguments
    args = parser.parse_args()

    return args
