# Wind environment

Now it is possible to set the wind speed (in kt) and its cardinal direction in _env.py_ and the number _n_wind_dir_ of available wind directions (e.g., N, E, S, W) in _atcenv/utils.py_.
Thus, if you set _n_wind_dir_=4, then you will be able to choose bewteen [N,E,S,W] cardinal directions when apply the wind noise. If you set instead _n_wind_dir_=16, then you will be able to choose between [N, NE1, NE2, NE3, E, SE1, SE2, SE3, S, SW1, SW2, SW3, W, NW1, NW2, NW3]. Remeber that _n_wind_dir_ must be obviously a multiple of 4 (which is the number of the 4 main quadrants).

Here below it is shown a snapshot of this windy environemnt.

![windy env](Images/wind_env_screen.png)

In this screenshot are rerepsented the following features:
  - long thin blue line: distance between the current agent and its target;
  - short thick yellow line: heading speed direction;
  - short thick green line: wind speed direction;
  - short thick blue line: track speed direction (i.e., the astual agent speed vector resulting from the combination of heading and wind speed vectors).
