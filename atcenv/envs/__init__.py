from .FlightEnv import FlightEnv, Flight
from .LoggerWrapper import LoggerWrapper
from .RayWrapper import RayWrapper
from .CurriculumFlightEnv import CurriculumFlightEnv

__all__ = [
    "RayWrapper",
    "LoggerWrapper",
    "CurriculumFlightEnv",
    "Flight",
    "FlightEnv"
]


def get_env_cls():
    return LoggerWrapper
