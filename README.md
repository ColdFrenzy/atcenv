# Installing 
Python 3.8 is required to run the repo. 
For mac-os please upgrade to python 3.8.5 if you are experiencing a [Symbol not found error](https://github.com/ray-project/ray/issues/10428).

Use 
```pip install -r requirements.txt```
To install all the required modules. Then use:
```
pip install ray[tune] ray[rllib]
```
For the additional ray libraries

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

At present, however, the ATC simulator does not consider the vertical dimension (i.e., all aircraft are assumed to be at the same altitude), and consequently the policy can only learn speed and/or heading resolution actions. Furthermore, the simulator does not include uncertainty, meaning that the policy may not perform well in real-life situations, where uncertainty is inevitable. Last but not least, the PPO algorithm was not explicitly designed for multi-agent environments, and therefore other algorithms like Actor-Attention-Critic for Multi-Agent Reinforcement Learning (MAAC) or Deep Coordination Graphs (DCG) may achieve better performance.

We will provide you with the skeleton of a basic 2D ATC simulator (the environment) built on the Gym framework in this challenge (in Python). You will tailor this simulator by adding:

* The observation function (what do agents observe from the environment to take actions?)
* The reward function (how are agents reward or penalised by their actions?)
* The action space (e.g., heading change, speed change), which can be discrete or continuous 

[source code of the Environment](https://github.com/ramondalmau/atcenv/blob/main/atcenv/env.py)

and then you will train the optimal policy using a reinforcement learning algorithm of your choice. 

We also encourage you to explore any of the following bonus tasks:
* Implement the vertical dimension in the simulation environment
* Implement weather in the simulator (e.g, consider the effect of wind)
* Implement uncertainty in the simulation environment. For instance, due to measurement errors, the position observed by the agents may not perfectly correspond to the actual one, or agents may not react instantaneously to resolution actions. 

The jury will consider the following factors when evaluating the solutions proposed by the various teams:
* The performance of the policy (e.g., number of conflicts, extra distance / environmental impact, number of resolution actions)
* The learnt policy's realism and scalability to any number of agents/flights
* The originality and appropriateness of the approach
* The clarity of the presentation
* Bonus task will be positively considered as well

## References
---
[Dalmau, R. and Allard, E. "Air Traffic Control Using Message Passing Neural Networks and Multi-Agent Reinforcement Learning", 2020. 10th SESAR Innovation Days](https://www.researchgate.net/publication/352537798_Air_Traffic_Control_Using_Message_Passing_Neural_Networks_and_Multi-Agent_Reinforcement_Learning)
---

## Download

```bash
git clone https://github.com/ramondalmau/atcenv.git
```

## Installation 

The environment has been tested with Python 3.8 and the versions specified in the requirements.txt file

```bash
cd atcenv
pip install -r requirements.txt
python setup.py install
```

## Usage

```python
from atcenv import FlightEnv

# create environment
env = FlightEnv()

# reset the environment
obs = env.reset()

# set done status to false
done = False

# execute one episode
while not done:
    # compute the best action with your reinforcement learning policy
    action = ...

    # perform step
    obs, rew, done, info = env.step(action)

    # render (only recommended in debug mode)
    env.render()

env.close()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
