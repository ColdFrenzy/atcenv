import os
import numpy as np
from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser

from atcenv.envs import FlightEnv
from atcenv.envs import CurriculumFlightEnv


def parse_args():
    """

    """
    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('-debug', action="store_true")
    parser.add_argument('-cur_dir', default=os.getcwd())
    # parser.add_class_arguments(FlightEnv, 'env')
    parser.add_class_arguments(CurriculumFlightEnv, 'env')

    # parse arguments
    args = parser.parse_args()

    assert args.num_workers <= args.num_cpus, f"The number of workers must be less equal to the cpu count. Got {args.num_workers}>{args.num_cpus}"

    return args


def curriculum_fn(
        train_results: dict, task_settable_env: "TaskSettableEnv", env_ctx: "EnvContext") -> "TaskType":
    """Function returning a possibly new task to set `task_settable_env` to.
    Args:
        train_results (dict): The train results returned by Trainer.train().
        task_settable_env (TaskSettableEnv): A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx (EnvContext): The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.
    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """

    # If all episodes in the evaluation were able to end before the max_episode_len
    # we move to the next task
    new_task = task_settable_env.get_task()
    in_eval = task_settable_env.in_eval
    max_level = task_settable_env.max_level
    if in_eval:
        print("Evaluating results...")
        if all(ep_len < env_ctx["max_episode_len"] for ep_len in train_results["evaluation"]["hist_stats"]["episode_lengths"]):
            if (new_task + 1) < max_level:
                new_task += 1
            print(
                f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
                f"\nR={train_results['episode_reward_mean']}"
                f"\nSetting env to task={new_task}"
            )

    return new_task
