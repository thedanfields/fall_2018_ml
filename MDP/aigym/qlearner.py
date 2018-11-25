import numpy as np
import random as rand


class QLearner(object):
    if __name__ == "__main__":
        print "Remember Q from Star Trek? Well, this isn't him"

    def author(self):
        return 'dfields34'

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.dyna = dyna
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr

        self.q_table = np.zeros(shape=(num_states, num_actions))
        self.r_table = np.zeros(shape=(num_states, num_actions))

        self.tc_table = np.zeros(shape=(num_states, num_actions, num_states))
        self.tc_table.fill(0.00001)

    @staticmethod
    def should_take_random_action(rate_of_random_action):
        return (rand.randrange(1, 100) / 100.0) <= rate_of_random_action

    @staticmethod
    def get_random_action(number_of_possible_actions):
        return rand.randint(0, number_of_possible_actions - 1)

    @staticmethod
    def update_q_table(q_table, state, action, state_prime, reward, alpha, gamma):

        best_action_for_future_state = np.argmax(q_table[state_prime])
        future_discounted_rewards = alpha * (reward + gamma * q_table[state_prime][best_action_for_future_state])

        current_q_value = q_table[state][action]
        new_q_value = (1 - alpha) * current_q_value + future_discounted_rewards

        q_table[state][action] = new_q_value

    def get_next_action(self, s_prime):
        is_random_action = self.should_take_random_action(self.rar)
        self.rar = self.rar * self.radr

        if is_random_action:
            return self.get_random_action(self.num_actions)

        next_action = np.argmax(self.q_table[s_prime])

        return next_action

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.get_next_action(s)
        self.a = action

        if self.verbose: print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # update dyna models
        #self.update_tc_table(self.tc_table, self.s, self.a, s_prime)
        #self.update_r_table(self.r_table, self.s, self.a, r, self.alpha)

        self.update_q_table(self.q_table, self.s, self.a, s_prime, r, self.alpha, self.gamma)

        if self.dyna > 0:
            self.apply_dyna(self.dyna, self.q_table, self.tc_table, self.r_table, self.alpha, self.gamma)

        action = self.get_next_action(s_prime)

        # set internal state
        self.s = s_prime
        self.a = action

        if self.verbose: print "s =", s_prime, "a =", action, "r =", r
        return action

    @staticmethod
    def get_random_state(number_of_states):
        return rand.randint(0, number_of_states - 1)

    @staticmethod
    def update_tc_table(tc_table, state, action, next_action):
        tc_table[state][action][next_action] = tc_table[state][action][next_action] + 1

    @staticmethod
    def update_r_table(r_table, state, action, reward, alpha):
        current_reward_value = r_table[state][action]
        new_reward_value = (1 - alpha) * current_reward_value + alpha * reward
        r_table[state][action] = new_reward_value

    def apply_dyna(self, dyna, q_table, tc_table, r_table, alpha, gamma):
        t_table = self.build_t_table(tc_table)
        for _ in range(dyna):
            self.hallucinate(q_table, t_table, r_table, alpha, gamma)

    @staticmethod
    def build_t_table(tc_table):
        t_table = np.true_divide(tc_table, tc_table.sum(keepdims=True))
        return t_table

    def hallucinate(self, q_table, t_table, r_table, alpha, gamma):
        state = self.get_random_state(self.num_states)
        action = self.get_random_action(self.num_actions)

        next_state = np.argmax(t_table[state][action])
        reward = r_table[state][action]

        self.update_q_table(q_table, state, action, next_state, reward, alpha, gamma)
