"""
Definitions module
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

from shapely.geometry import Point, Polygon, LineString

import atcenv.common.units as u


@dataclass
class Airspace:
    """
    Airspace class
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float):
        """
        Creates a random airspace sector with min_area < area <= max_area

        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :return: random airspace
        """
        R = math.sqrt(max_area / math.pi)

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        p = [random_point_in_circle(R) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon=polygon)

    @classmethod
    def fixed(cls, points):
        """
        Creates a fixed airspace sector given its points
        """

        polygon = Polygon(points).convex_hull

        return cls(polygon=polygon)


@dataclass
class Flight:
    """
    Flight class
    """
    position: Point
    target: Point
    optimal_airspeed: float
    flight_id: int
    fov_depth: float = 60*u.nm
    fov_angle: float = math.pi / 2

    airspeed: float = field(init=False)
    track: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed
        :return:
        """
        # Since the track is initialized as the bearing, the track angle is also clockwise w.r.t. north
        self.track = self.bearing
        self.airspeed = self.optimal_airspeed
        self.optimal_trajectory = LineString(
            [(self.position.x, self.position.y), (self.target.x, self.target.y)])

    @ property
    def bearing(self) -> float:
        """
        Bearing from current position to target [0, 2PI]
        (clockwise angular distance between the north and the line connecting the Flight
        current position and its destination)
        :return:
        """
        # relative distance between target and flight
        # w.r.t the flight's frame
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        # compass=bussola
        # north-clockwise convention (x and y swapped):
        # https://en.wikipedia.org/wiki/Atan2#East-counterclockwise,_north-clockwise_and_south-clockwise_conventions,_etc.
        compass = math.atan2(dx, dy)
        # map atan between [0,2pi]
        # https://stackoverflow.com/questions/1311049/how-to-map-atan2-to-degrees-0-360
        return (compass + u.circle) % u.circle

    @ property
    def distance_from_optimal_trajectory(self) -> float:
        """
        Compute the distance from the optimal trajectory
        """
        return self.position.distance(self.optimal_trajectory)

    @ property
    def heading_prediction(self, dt: Optional[float] = 120) -> Point:
        """
        Predicts the future position after dt seconds related to the heading direction (wind effect not included)
        :param dt: prediction look-ahead time (in seconds)
        """
        dx, dy = self.components
        return Point(self.position.x + dx * dt, self.position.y + dy * dt)

    # Implementare l'HEADING per la visualizzazione --> !!!!!!!!!!!!!!!!!!!!!!!!

    @ property
    def fov(self) -> Polygon:
        """
        Returns the field of view of the given flight
        :return: polygon representing the agent's fov
        """
        fov_vertices = []
        # center = [self.flights[flight_id].position]
        center_x, center_y = self.position.x, self.position.y
        fov_vertices.append(Point(center_x, center_y))
        track = self.track
        point_1_x = center_x + (self.fov_depth *
                                (math.cos((math.pi - (track + math.pi / 2)) - self.fov_angle / 2)))
        point_1_y = center_y + (self.fov_depth *
                                (math.sin((math.pi - (track + math.pi / 2)) - self.fov_angle / 2)))
        fov_vertices.append(Point(point_1_x, point_1_y))
        point_2_x = center_x + (self.fov_depth *
                                (math.cos((math.pi - (track + math.pi / 2)) + self.fov_angle / 2)))
        point_2_y = center_y + (self.fov_depth *
                                (math.sin((math.pi - (track + math.pi / 2)) + self.fov_angle / 2)))
        fov_vertices.append(Point(point_2_x, point_2_y))

        ##########################################################
        return Polygon(fov_vertices)
        ##########################################################

    @ property
    def components(self) -> Tuple:
        """
        X and Y Speed components (in kt)
        :return: speed components (HEADING)
        """
        dx = self.airspeed * math.sin(self.track)
        dy = self.airspeed * math.cos(self.track)

        return dx, dy

    @ property
    def distance(self) -> float:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        return self.position.distance(self.target)

    @ property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        drift is between [-PI, PI]
        :return:
        """
        drift = self.bearing - self.track

        if drift > math.pi:
            return -(u.circle - drift)
        elif drift < -math.pi:
            return (u.circle + drift)
        else:
            return drift

    @classmethod
    def random(cls, airspace: Airspace, min_speed: float, max_speed: float, flight_id: int, tol: float = 0., ):
        """
        Creates a random flight

        :param airspace: airspace where the flight is located
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param flight_id: identifier for a flight
        :param tol: tolerance to consider that the target has been reached (in meters)
        :return: random flight
        """

        def random_point_in_polygon(polygon: Polygon) -> Point:
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                point = Point(random.uniform(minx, maxx),
                              random.uniform(miny, maxy))
                if polygon.contains(point):
                    return point

        # random position
        position = random_point_in_polygon(airspace.polygon)

        # random target
        boundary = airspace.polygon.boundary
        while True:
            d = random.uniform(0, airspace.polygon.boundary.length)
            target = boundary.interpolate(d)
            if target.distance(position) > tol:
                break

        # random speed
        airspeed = random.uniform(min_speed, max_speed)

        return cls(position, target, airspeed, flight_id)

    @ classmethod
    def fixed(cls, flight_pos: Point, target_pos: Point, airspeed: float, airspace: Airspace, flight_id: int):
        """
        Creates a fixed flight

        :param flights_pos: positions of the flight in the airspace
        :return Flight: fixed flight
        """
        assert airspace.polygon.contains(
            flight_pos), "The point is outside of the Polygon"

        return cls(flight_pos, target_pos, airspeed, flight_id)
