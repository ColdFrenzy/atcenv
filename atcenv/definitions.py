"""
Definitions module
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

from shapely.geometry import Point, Polygon

import atcenv.common.units as u


@dataclass
class Airspace:
    """
    Airspace class
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float, min_h: float, max_h: float):
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

        def random_point_in_cylinder(radius: float, min_h: Optional[float] = 0, max_h: Optional[float] = 0) -> Point:
            point2D = random_point_in_circle(radius)
            z = random.uniform(min_h, max_h)
            return Point(point2D.x, point2D.y, z)

        p = [random_point_in_cylinder(R, min_h, max_h) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(random_point_in_cylinder(R, min_h, max_h))
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
    fov_depth: float
    fov_angle: float

    airspeed: float = field(init=False)
    track: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed
        :return:
        """
        self.track = self.bearing
        self.altitude_track = self.elevation_angle # 0 # TODO: forse è meglio metterlo uguale a zero assumendo che lungo Z all'inizio nessuno abbia nessuna componente (vanno tutti lungo li piano orizzontale)
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
    def elevation_angle(self) -> float:
        """
        Elevation angle between agent and its target
        :return:
        """
        xy_dist = self.position.distance(self.target)
        # if altitude>=0 --> Z agent is equal or higher than Z target, otherwise Z agent is lower than the Z target 
        altitude = self.position.z-self.target.z 
        # bussola
        # CHECK: Compute the elevation angle between the altitude (computed as the heigt between the agent and its target) and the XY Euclidean distance between the agent and its target 
        compass = math.atan2(altitude, xy_dist)
        return (compass + u.circle) % u.circle

    @property
    def heading_prediction(self, dt: Optional[float] = 120) -> Point:
        """
        Predicts the future position after dt seconds related to the heading direction (wind effect not included)   
        :param dt: prediction look-ahead time (in seconds) 
        """
        dx, dy, dz = self.components
        return Point(self.position.x + dx * dt, self.position.y + dy * dt, self.position.z + dz * dt)

    @property
    def fov(self) -> Polygon:
        """
        Returns the field of view of the given flight
        :return: polygon representing the agent's fov
        """
        fov_vertices = []
        # center = [self.flights[flight_id].position]
        center_x, center_y = self.position.x, self.position.y
        fov_vertices.append(Point(center_x, center_y))
        bearing = self.track
        point_1_x = center_x + (self.fov_depth *
                                (math.cos((math.pi - (bearing + math.pi / 2)) - self.fov_angle / 2)))
        point_1_y = center_y + (self.fov_depth *
                                (math.sin((math.pi - (bearing + math.pi / 2)) - self.fov_angle / 2)))
        fov_vertices.append(Point(point_1_x, point_1_y))
        point_2_x = center_x + (self.fov_depth *
                                (math.cos((math.pi - (bearing + math.pi / 2)) + self.fov_angle / 2)))
        point_2_y = center_y + (self.fov_depth *
                                (math.sin((math.pi - (bearing + math.pi / 2)) + self.fov_angle / 2)))
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
        dz = self.airspeed * math.sin(self.altitude_track) # CHECK: forse è math.sin perchè qua utilizzano gli assi invertiti (??)

        return dx, dy, dz

    @property
    def distance(self) -> float:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        position = [self.position.x, self.position.y, self.position.y]
        target = [self.target.x, self.target.y, self.target.y]
        return math.dist(position, target)
        #return self.position.distance(self.target)

    @property
    def drift_clip(self, drift):
        if drift == math.pi:
            return drift
        elif drift < math.pi:
            return drift
        elif self.bearing > self.track:
            return drift - u.circle
        else:
            return u.circle - drift

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.bearing - self.track
        drift = abs(drift)

        return self.drift_clip(drift)

    @property
    def elevation_drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.elevation_angle - self.altitude_track
        drift = abs(drift)

        return self.drift_clip(drift)

    @classmethod
    def fixed(cls, airspace: Airspace, position: Point, min_speed: float, max_speed: float, flight_id: int,
              tol: float = 0.):
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
            tp_dist = math.dist([position.x, position.y, position.z], [target.x, target.y, target.z])
            if tp_dist > tol:
                break
            #if target.distance(position) > tol:
            #    break

        # random speed
        airspeed = random.uniform(min_speed, max_speed)

        return cls(position, target, airspeed, flight_id)

    @classmethod
    def random(cls, airspace: Airspace, min_speed: float, max_speed: float, flight_id: int, tol: float = 0.):
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
            poly_coords = list(polygon.exterior.coords)
            minz, maxz = min([point[2] for point in poly_coords]), max([point[2] for point in poly_coords])

            while True:
                point = Point(random.uniform(minx, maxx),
                              random.uniform(miny, maxy),
                              random.uniform(minz, maxz))
                
                if polygon.contains(point): # --> A cosa serve questo controllo? --> i punti saranno sempre dentro al polifono per come vengono generati (minx, miny, ...)
                    return point

        # random position
        position = random_point_in_polygon(airspace.polygon)

        # random target
        boundary = airspace.polygon.boundary
        while True:
            d = random.uniform(0, airspace.polygon.boundary.length)
            target = boundary.interpolate(d)
            tp_dist = math.dist([position.x, position.y, position.z], [target.x, target.y, target.z])
            if tp_dist > tol:
                break
            #if target.distance(position) > tol:
            #    break

        # random speed
        airspeed = random.uniform(min_speed, max_speed)
        fov_depth = 20*u.nm
        fov_angle = math.pi / 2

        return cls(position, target, airspeed, flight_id, fov_depth=fov_depth, fov_angle=fov_angle)
