import numpy as np
from typing import Optional


class SARSA:
    def __init__(
            self,
            action_space_size: int,
            state_space_size: int,
            state_max_val: int,
            action_max_val: int,
            num_agents: int,
            share_parameters: Optional[bool] = True,
            learning_rate: Optional[float] = 0.01,  # alpha
            gamma: Optional[float] = 0.99,
            epsilon: Optional[float] = 0.01,
            epsilon_decr: Optional[bool] = False,
            epsilon_decrement: Optional[float] = 0.999,
            epsilon_min: Optional[float] = 0.01,
            epsilon_min2: Optional[float] = 0.1,
            max_value_for_Rmax: Optional[float] = 100.0,
    ):
        """
        SARSA algorithm initialization
        :param action_space_size: size of the environment's action space
        :param state_space_size: size of the environment's state space
        :param state_max_val: max_value that an element of the states may have (e.g if state_value = 5 then a  single state element should have a value between 0 and 5)
        :param action_max_val: max_value that an elemten of the actions may have (e.g. if action_value = 3 then a single action element should have a value between 0 and 3)
        :param num_agents: number of agents in the environment. This parameter is important only if shared_parameters==False
        :param share_parameters: if True, it uses a single Q-table for all the agents
        :param learning_rate: 
        :param gamma: discount factor
        :param epsilon:
        :param epsilon_decr: 
        :param epsilon_decrement: 
        :param epsilon_min:
        :param epsilon_min2:          
        :param max_value_for_Rmax:
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.state_max_val = state_max_val
        self.action_max_val = action_max_val
        self.num_agents = num_agents
        self.share_parameters = share_parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decr = epsilon_decr
        self.epsilon_decrement = epsilon_decrement
        self.epsilon_min = epsilon_min
        self.epsilon_min2 = epsilon_min2
        self.max_value_for_Rmax = max_value_for_Rmax

        if self.share_parameters:
            self.q_table = np.zeros(
                shape=(self.state_space_size, self.action_space_size),
                dtype='float32')
        else:
            print("Number of agents: ", self.num_agents)
            self.q_table = np.zeros(shape=(
                self.num_agents, self.state_space_size, self.action_space_size),
                dtype='float32')

    def update(self, state, state2, reward, action, action2, agent_indx: Optional[int] = None):
        """update the Q-table

        Since the state is a vector of dimension n and each element can have a value between
        [0,m]. We can access the state index through base-m counts
        state_indx = state[0]*m^0 + state[1]*m^1 + ... + state[n]*m^n
        Same thing for the action space.
        :param state:
        :param state2:
        :param reward:
        :param action:
        :param action2:
        :param agent_ind: required if the Q-table is not shared
        """
        if self.share_parameters:
            state1_indx = self.get_state_indx(state)
            action1_indx = self.get_action_indx(action)
            prediction = self.q_table[state1_indx, action1_indx]
            state2_indx = self.get_state_indx(state2)
            action2_indx = self.get_action_indx(action2)
            target = reward + self.gamma * \
                self.q_table[state2_indx, action2_indx]
            self.q_table[state1_indx, action1_indx] += \
                self.lr*(target-prediction)

        else:
            state1_indx = self.get_state_indx(state)
            action1_indx = self.get_action_indx(action)
            prediction = self.q_table[state1_indx, action1_indx]
            state2_indx = self.get_state_indx(state2)
            action2_indx = self.get_action_indx(action2)
            target = reward + self.gamma * \
                self.q_table[agent_indx, state2_indx, action2_indx]
            self.q_table[agent_indx, state1_indx, action1_indx] += \
                self.lr*(target-prediction)

    def get_state_indx(self, state):
        """return the state index in the Q-table given a state

        Since the state is a vector of dimension i and each element can have a value between
        [0,m]. We can access the state index through base-i counts
        state_indx = state[0]*i^0 + state[1]*i^1 + ... + state[n]*i^n
        """
        indx = 0
        m = self.state_max_val
        for i in range(len(state)):
            indx += state[i]*(m**i)

        return indx

    def get_action_indx(self, action):
        """return the action index in the Q-table given an action

        Since the action is a vector of dimension i and each element can have a value between
        [0,m]. We can access the action index through base-i counts
        action_indx = action[0]*i^0 + action[1]*i^1 + ... + action[n]*i^n
        """
        indx = 0
        m = self.action_max_val
        for i in range(len(action)):
            indx += action[i]*(m**i)

        return indx

    def choose_action(self, state, agent_indx: Optional[int] = None):
        """return the index of an action from the Q-table
        """
        action_indx = 0
        state_indx = self.get_state_indx(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action_indx = np.random.randint(self.action_space_size)
        else:
            if self.share_parameters:
                action_indx = np.argmax(self.q_table[state_indx, :])
            else:
                action_indx = np.argmax(
                    self.q_table[agent_indx, state_indx, :])

        return action_indx
