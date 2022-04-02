import numpy as np
import copy
from typing import Optional 

class RL_algorithms:
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
            lam: Optional[float] = 0.9,
            alg: Optional[str] = 'SARSAL',
            q_init: Optional[str] = 'zero'

    ):
        """
        SARSA algorithm initialization
        :param action_space_size: size of the environment's action space
        :param state_space_size: size of the environment's state space
        :param state_max_val: max_value that an element of the states may have (e.g if state_value = 5 then a  single state element should have a value between 0 and 5)
        :param action_max_val: max_value that an elemten of the actions may have (e.g. if action_value = 3 then a single action element should have a value between 0 and 3)
        :param num_agents: number of agents in the environment. This parameter is important only if shared_parameters==False
        :param share_parameters: if True, it uses a single Q-table for all the agents
        :param learning_rate: select the learning rate value
        :param gamma: selecte the discount factor value
        :param epsilon: select epsilon value for 'exploration' phase
        :param epsilon_decr:
        :param epsilon_decrement:
        :param epsilon_min:
        :param epsilon_min2:
        :param lam: lambda value
        :param alg: algorithm to use
        :param q_init: Q-Table and E-Table (if any) initialization ('zero' or 'rand')
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
        self.lam = lam
        self.algs = ['QL', 'SARSA', 'QLL', 'SARSAL']
        self.alg = alg
        self.alg_check()
        self.q_init = q_init
        self.q_table, self.e_table = self.tables_init()

    def tables_init(self):
        """
        Initialize Q-Table and E-Table (if any)
        """
        assert self.q_init=='zero' or self.q_init=='rand', 'Invalid parameter for Q-Table initialization!'
        q_table = np.empty(shape=0)
        e_table = np.empty(shape=0)
        if self.share_parameters:
            # Zero initialization
            if self.q_init=='zero':
                q_table = np.zeros(
                    shape=(self.state_space_size, self.action_space_size),
                    dtype='float32')
            # Random initialization
            elif self.q_init=='rand':
                q_table = np.random.rand(self.state_space_size, self.action_space_size)
            if self.alg=='SARSAL' or self.alg=='QLL':
                # E-Table initialized with the same values of the Q-Table
                e_table = copy.deepcopy(q_table)
            else:
                pass
        else:
            print("Number of agents: ", self.num_agents)
            if self.q_init=='zero':
                q_table = np.zeros(shape=(
                    self.num_agents, self.state_space_size, self.action_space_size),
                    dtype='float32')
            elif self.q_init=='rand':
                q_table = np.random.rand(self.num_agents, self.state_space_size, self.action_space_size, dtype='float32')
            if self.alg=='SARSAL' or self.alg=='QLL':
                e_table = copy.deepcopy(q_table)
            else:
                pass

        return q_table, e_table

    def alg_check(self):
        """
        Check on selected algorithm
        """
        assert self.alg in self.algs, 'Invalid selected algorithm! Choose between ' + str(self.algs)

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

        state1_indx = self.get_state_indx(state)
        action1_indx = self.get_action_indx(action)
        state2_indx = self.get_state_indx(state2)
        action2_indx = self.get_action_indx(action2)

        if self.alg=='QL':
            self.update_qlearning(state1_indx, state2_indx, reward, action1_indx, self.q_table, self.share_parameters)
        elif self.alg=='SARSA':
            self.update_sarsa(state1_indx, state2_indx, reward, action1_indx, action2_indx, self.q_table, self.share_parameters)
        elif self.alg=='SARSAL':
            self.update_sarsa_lambda(state1_indx, state2_indx, reward, action1_indx, action2_indx, self.state_space_size, self.action_space_size, self.q_table,
                        self.e_table, self.share_parameters)
        elif self.alg=='QLL':
            self.update_qlearning_lambda(state1_indx, state2_indx, reward, action1_indx, action2_indx, self.state_space_size, self.action_space_size, self.q_table,
                            self.e_table, self.share_parameters)

    def update_qlearning_lambda(self, state1_indx, state2_indx, reward, action1_indx, action2_indx, len_states, len_actions, q_matrix,
                                e_matrix, share_parameters, agent_indx: Optional[int] = None):
        """
        Q-learning(lambda) (Watkins's Q(lambda) algorithm) function to update the Q-value matrix and the Eligibility matrix
        """
        predict = q_matrix[state1_indx, action2_indx]
        if share_parameters:
            maxQ = np.amax(q_matrix[state2_indx, :])        # Find maximum value for the new state Q(s', a*)
            maxIndex = np.argmax(q_matrix[state2_indx, :])  # Find index of the maximum value a*
            target = reward + self.gamma * maxQ
            delta = target - predict
            e_matrix[state1_indx, action1_indx] = e_matrix[state1_indx, action1_indx] + 1
            # For all s, a
            for s in range(len_states):
                for a in range(len_actions):
                    q_matrix[s, a] = q_matrix[s, a] + self.lr * delta * e_matrix[s, a]
                    if action2_indx == maxIndex:
                        e_matrix[s, a] = self.gamma * self.lam * e_matrix[s, a]
                    else:
                        e_matrix[s, a] = 0
        else:
            maxQ = np.amax(q_matrix[agent_indx, state2_indx, :])        # Find maximum value for the new state Q(s', a*)
            maxIndex = np.argmax(q_matrix[agent_indx, state2_indx, :])  # Find index of the maximum value a*
            target = reward + self.gamma * maxQ
            delta = target - predict
            e_matrix[agent_indx, state1_indx, action1_indx] = e_matrix[agent_indx, state1_indx, action1_indx] + 1
            # For all s, a
            for s in range(len_states):
                for a in range(len_actions):
                    q_matrix[agent_indx, s, a] = q_matrix[agent_indx, s, a] + self.lr * delta * e_matrix[agent_indx, s, a]
                    if action2_indx == maxIndex:
                        e_matrix[agent_indx, s, a] = self.gamma * self.lam * e_matrix[agent_indx, s, a]
                    else:
                        e_matrix[agent_indx, s, a] = 0

    def update_sarsa_lambda(self, state1_indx, state2_indx, reward, action1_indx, action2_indx, len_states, len_actions, q_matrix,
                            e_matrix, share_parameters, agent_indx: Optional[int] = None):
        """
        SARSA(lambda) function to update the Q-value matrix and the Eligibility matrix
        """
        prediction = q_matrix[state1_indx, action1_indx]
        if share_parameters:
            target = reward + self.gamma * q_matrix[state2_indx, action2_indx]
            delta = target - prediction
            e_matrix[state1_indx, action1_indx] = e_matrix[state1_indx, action1_indx] + 1
            # For all s, a
            for s in range(len_states):
                for a in range(len_actions):
                    q_matrix[s, a] = q_matrix[s, a] + self.lr * delta * e_matrix[s, a]
                    e_matrix[s, a] = self.gamma * self.lam * e_matrix[s, a]
        else:
            target = reward + self.gamma * q_matrix[agent_indx, state2_indx, action2_indx]
            delta = target - prediction
            e_matrix[agent_indx, state1_indx, action1_indx] = e_matrix[agent_indx, state1_indx, action1_indx] + 1
            # For all s, a
            for s in range(len_states):
                for a in range(len_actions):
                    q_matrix[agent_indx, s, a] = q_matrix[agent_indx, s, a] + self.lr * delta * e_matrix[agent_indx, s, a]
                    e_matrix[agent_indx, s, a] = self.gamma * self.lam * e_matrix[agent_indx, s, a]

    def update_qlearning(self, state1_indx, state2_indx, reward, action1_indx, q_matrix, share_parameters, agent_indx: Optional[int] = None):
        """
        # Q-learning function to learn the Q-value
        """
        prediction = q_matrix[state1_indx, action1_indx]
        if share_parameters:
            maxQ = np.amax(q_matrix[state2_indx, :])  # Find maximum value for the new state
            target = reward + self.gamma * maxQ
            q_matrix[state1_indx, action1_indx] += self.lr * (target - prediction)
        else:
            maxQ = np.amax(q_matrix[agent_indx, state2_indx, :])  # Find maximum value for the new state
            target = reward + self.gamma * maxQ
            q_matrix[agent_indx, state1_indx, action1_indx] += self.lr * (target - prediction)

    def update_sarsa(self, state1_indx, state2_indx, reward, action1_indx, action2_indx, q_matrix, share_parameters, agent_indx: Optional[int] = None):
        """
        SARSA function to learn the Q-value
        """
        prediction = q_matrix[state1_indx, action1_indx]
        if share_parameters:
            target = reward + self.gamma * q_matrix[state2_indx, action2_indx]
            q_matrix[state1_indx, action1_indx] += self.lr * (target - prediction)
        else:
            target = reward + self.gamma * q_matrix[agent_indx, state2_indx, action2_indx]
            q_matrix[agent_indx, state1_indx, action1_indx] += self.lr * (target - prediction)


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
