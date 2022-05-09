##########################################################
# REWARDS UTILS
##########################################################
def speed_penalty(f, weight: float, min_speed, max_speed):
    """
    each flight's speed should be as close as possible to optimal speed
    :param f: Flight object
    :param weight: weight to associate to this penalty
    :param min_speed:
    :param max_speed:
    :return speed_penalty:
    """
    cur_speed = f.airspeed
    optimal_speed = f.optimal_airspeed

    diff_speed = cur_speed - optimal_speed
    perc = 0
    # flight is going slower than optimal
    max_diff = max_speed - min_speed
    if diff_speed < 0:
        diff_speed = abs(diff_speed)
        perc = diff_speed / max_diff
    elif diff_speed > 0:
        perc = diff_speed / max_diff

    return perc * weight


def target_dist(f, max_distance) -> float:
    """
    Return the normalized distance between the flight and its target
    :param f: Flight object
    :param max_distance: 
    """
    dist = f.distance
    dist /= max_distance
    # During exploration it may happen that the Flight goes outside the Airspace
    if dist >= 1:
        dist = 1

    return dist


def target_reached(f, tol) -> bool:
    """
    Check if the flight has reached the target
    :param f: Flight object
    :param tol: tolerance parameter.
    """

    dist = f.distance

    if dist < tol:
        return True
    return False

