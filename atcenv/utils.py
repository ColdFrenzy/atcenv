from typing import Tuple, Dict, Optional
import numpy as np

def absolute_compass(n_dir: Optional[int] = 16) -> Dict[str, Tuple[float]]:
    """
    Return an absolute compass made up by a specific number of directions (referring to the wind directions).
    Angles (in degrees) are set by increasing them from 0° to 360° according to the desired number of directions:
    they are supposed to increase clockwise starting from the North direction (assumed to be directed along towards Y axis).
    Compass is assumed to be centered in (0,0). 

    :param n_dir: number of desired directions
    """

    # Quadrants number:
    n_quad = 4
    assert n_dir%n_quad==0, 'Compass direction must be a multiple of 4 (which is equal to the number of the main quadrants)!'
    
    # Main cardinal points:
    cardinal_points = ['N', 'E', 'S', 'W']
    # Degree increment for each considered direction:
    degree_incr = 360/n_dir
    # Degrees of each direction clockwise from the North: 
    cardinal_degrees = [degree_incr*i for i in range(n_dir)]
    # Number of direction per quadrant (exluded the first main direction for the considered quadrant):
    n_quad_dir = (n_dir-n_quad)/n_quad+1
    # Main quadrants degre ranges:
    q1 = range(0,90)
    q2 = range(90,180)
    q3 = range(180,270)
    q4 = range(270,360)

    # Assign all the available compass directions:
    compass = {}
    cur_card_count = 0
    for deg in cardinal_degrees:
        int_deg = int(deg)
        
        if int_deg in q1:
            if deg==q1[0]:
                card = 'N'
            else:
                card = 'NE'
              
        elif int_deg in q2:
            if deg==q2[0]:
                card = 'E'
            else:
                card = 'SE'
            
        elif int_deg in q3:
            if deg==q3[0]:
                card = 'S'
            else:
                card = 'SW'
            
        elif int_deg in q4:
            if deg==q4[0]:
                card = 'W'
            else:
                card = 'NW'
            
        
        if cur_card_count!=0:
            card += str(cur_card_count)

        if (cur_card_count+1)%n_quad_dir==0:
            cur_card_count = 0
        else:
            cur_card_count += 1


        # 'deg' is the angle between wind and North: 
        compass[card] = deg

    return compass

# Number of available wind directions:
n_wind_dir = 16
abs_compass = absolute_compass(n_wind_dir)
