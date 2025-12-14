import numpy as np
import random


def initialize_q_table(n_states, n_actions):
    """
    Initialize the Q-table with zeros.
    """
    return np.zeros((n_states, n_actions))


def epsilon_greedy_policy(Q, state, epsilon):
    """
    Choose an action using an epsilon-greedy policy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, Q.shape[1] - 1)
    else:
        return np.argmax(Q[state])


def update_q_table(Q, state, action, reward, next_state, alpha, gamma):
    """
    Update the Q-table using the Q-learning update rule.
    """
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state, best_next_action]
    td_error = td_target - Q[state, action]

    Q[state, action] += alpha * td_error
