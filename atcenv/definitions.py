"""
Definitions module
"""
from shapely.geometry import Point, Polygon
from dataclasses import dataclass, field
import atcenv.common.units as u
import math
import random
import numpy as np
from typing import Optional, Tuple, List, Dict

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

@dataclass
class Flight:
    """
    Flight class
    """
    position: Point
    target: Point
    optimal_airspeed: float
    flight_id: int

    airspeed: float = field(init=False)
    track: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed
        :return:
        """
        self.track = self.bearing
        self.airspeed = self.optimal_airspeed

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target
        :return:
        """
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        # bussola
        compass = math.atan2(dx, dy)
        return (compass + u.circle) % u.circle

    @property
    def heading_prediction(self, dt: Optional[float] = 120) -> Point:
        """
        Predicts the future position after dt seconds related to the heading direction (wind effect not included)   
        :param dt: prediction look-ahead time (in seconds) 
        """
        dx, dy = self.components
        return Point(self.position.x + dx*dt, self.position.y + dy*dt)

    # Implementare l'HEADING per la visualizzazione --> !!!!!!!!!!!!!!!!!!!!!!!!

    @property
    def fov(self, depth: Optional[float] = 50000., angle: Optional[float] = math.pi/2) -> Polygon:
        """
        Returns the field of view of the given flight
        :return: polygon representing the agent's fov
        """
        fov_vertices = []
        # center = [self.flights[flight_id].position]
        center_x, center_y = self.position.x, self.position.y
        fov_vertices.append(Point(center_x, center_y))
        bearing = self.track
        point_1_x = center_x + (depth *
                                (math.cos((math.pi-(bearing+math.pi/2)) - angle/2)))
        point_1_y = center_y + (depth *
                                (math.sin((math.pi-(bearing+math.pi/2)) - angle/2)))
        fov_vertices.append(Point(point_1_x, point_1_y))
        point_2_x = center_x + (depth *
                                (math.cos((math.pi-(bearing+math.pi/2)) + angle/2)))
        point_2_y = center_y + (depth *
                                (math.sin((math.pi-(bearing+math.pi/2)) + angle/2)))
        # point_2 = depth*((math.cos(self.flights[flight_id].bearing + angle/2))**2 + (
        #     math.sin(self.flights[flight_id].bearing + angle/2))**2)
        fov_vertices.append(Point(point_2_x, point_2_y))

        ##########################################################
        return Polygon(fov_vertices)
        ##########################################################

    @property 
    def components(self) -> Tuple:
        """
        X and Y Speed components (in kt)
        :return: speed components (HEADING)
        """        
        dx = self.airspeed * math.sin(self.track) 
        dy = self.airspeed * math.cos(self.track)

        return dx, dy

    @property
    def distance(self) -> float:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        return self.position.distance(self.target)

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.bearing - self.track
        drift = abs(drift)

        if drift == math.pi:
            return drift
        elif drift < math.pi:
            return drift
        elif self.bearing > self.track:
            return drift - u.circle
        else:
            return u.circle - drift

    @classmethod
    def fixed(cls, airspace: Airspace, position: Point, min_speed: float, max_speed: float, flight_id: int, tol: float = 0.):
        """
        Creates a fixed flight

        :param airspace: airspace where the flight is located
        :param position: flight position
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param flight_id: identifier for a flight
        :param tol: tolerance to consider that the target has been reached (in meters)
        :return: fixed flight
        """
        assert airspace.contains(
            position), "The point is outside of the Polygon"
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

    @classmethod
    def random(cls, airspace: Airspace, min_speed: float, max_speed: float, flight_id: int, tol: float = 0.,):
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
