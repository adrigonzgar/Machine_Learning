import gymnasium as gym
import numpy as np

from q_learning import (
    initialize_q_table,
    epsilon_greedy_policy,
    update_q_table
)


def train_q_learning(
    env,
    n_episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.999
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = initialize_q_table(n_states, n_actions)
    rewards_per_episode = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            update_q_table(Q, state, action, reward, next_state, alpha, gamma)

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return Q, rewards_per_episode


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)

    Q, rewards = train_q_learning(
        env,
        n_episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )

    print("Training finished.")
    print("Average reward:", np.mean(rewards[-100:]))
