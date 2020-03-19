import random

import numpy as np


class QLearn:
    def __init__(self, labels, learning_rate=0.7, gamma=0.8, epsilon=0.1, no_repeat=False, verbose=False):
        """
        Q-learn object

        :param labels: The list of symbols that represents every possible policy
        :param learning_rate:
        :param gamma:
        :param epsilon: Chances of taking random policy every new interaction
        :param no_repeat: If true, the random policy is chosen only from not explored policies
        :param verbose: If true, print some useful info
        """
        self.epsilon = epsilon
        self.data = {}
        self.labels = labels
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.memory = dict()
        self.no_repeat = no_repeat

    def push_q_data(self, state):
        """
        Makes sure that state is known by q-learn

        :param state: State representation
        :type state: str
        :return:
        """
        if state not in self.data.keys():
            self.data[state] = np.zeros(len(self.labels))
        if state not in self.memory.keys():
            self.memory[state] = np.zeros(len(self.labels))

    def select_best(self, state):
        """
        Given a state selects the known best policy

        :param state: State representation
        :type state: str
        :return: Chosen policy from self.labels
        """
        if self.verbose:
            print("select from: {}".format(self.data[state]))
        return self.labels[self.data[state].argmax()]

    def select_random(self, policies):
        """
        Select a random item of a policies list

        :param policies: List of selectable policies
        :type policies: list
        :return: Chosen policy from self.labels
        """
        if self.verbose:
            print("random select")
        return random.choice(policies)

    def select_rand_not_explored(self, state):
        """
        Random select a not explored policy given the state

        :param state: State representation
        :type state: str
        :return: Not explored policy
        """
        not_explored = []
        self.push_q_data(state)
        for i in range(len(self.memory[state])):
            if self.memory[state][i] == 0:
                not_explored.append(self.labels[i])
        if not_explored:
            return self.select_random(not_explored)
        else:
            return self.select_best(state)

    def interact(self, state):
        """
        Takes a decision based on the Q-learning learning algorithm given a state and preloaded hyper-params

        :param state: State representation (list or numpy.array)
        :type state: list, numpy.array
        :return: Chosen policy from self.labels
        """
        state = str(state)
        self.push_q_data(state)
        if random.uniform(0, 1) < self.epsilon:
            if self.no_repeat:
                return self.select_rand_not_explored(state)
            else:
                return self.select_random(self.labels)
        else:
            return self.select_best(state)

    def update_q(self, old_state, actual_state, reward, policy):
        """
        Updates the q-table given old_state actual_state reward and associated policy

        q_t := (1 - learning_rate) * q_(t-1) + learning_rate * (reward + gamma * max(q_t) )

        :param old_state: Representation of the old_state (list or numpy.array)
        :param actual_state: Representation of actual_state (list or numpy.array)
        :param reward: Reward obtained by going from old_state to actual_state
        :param policy: Action taken in old_state that leads to actual_state
        :return:

        :type old_state: list, numpy.array
        :type actual_state: list, numpy.array
        :type reward: float
        :type policy:
        """
        old_state = str(old_state)
        actual_state = str(actual_state)
        self.push_q_data(actual_state)
        self.push_q_data(old_state)
        self.memory[old_state][policy] = 1
        qt = self.data[actual_state].max()
        self.data[old_state][policy] = self.data[old_state][policy] + self.learning_rate * (
                reward + (self.gamma * qt) - self.data[old_state][policy])
        if self.verbose:
            print("old: {}".format(old_state))
            print("actual: {}".format(actual_state))
            print("policy: {}".format(policy))
            print("reward: {}".format(reward))
