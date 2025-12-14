import gymnasium as gym
import numpy as np

from train import train_q_learning


def evaluate_agent(env, Q, n_episodes=1000):
    successes = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward > 0:
            successes += 1

    return successes / n_episodes


def print_training_parameters(idx, cfg, n_episodes):
    print("\n" + "=" * 50)
    print(f"Configuration {idx}")
    print(f"  alpha         = {cfg['alpha']}")
    print(f"  gamma         = {cfg['gamma']}")
    print(f"  epsilon_decay = {cfg['epsilon_decay']}")
    print(f"  episodes      = {n_episodes}")
    print("=" * 50)


if __name__ == "__main__":

    train_env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    configurations = [
        {"alpha": 0.1,  "gamma": 0.99, "epsilon_decay": 0.999},
        {"alpha": 0.2,  "gamma": 0.95, "epsilon_decay": 0.995},
        {"alpha": 0.05, "gamma": 0.99, "epsilon_decay": 0.9995},
        {"alpha": 0.1,  "gamma": 0.9,  "epsilon_decay": 0.999},
        {"alpha": 0.1,  "gamma": 0.999,"epsilon_decay": 0.999},
        {"alpha": 0.1,  "gamma": 0.99, "epsilon_decay": 0.99},
        {"alpha": 0.1,  "gamma": 0.99, "epsilon_decay": 0.9999},
        {"alpha": 0.5,  "gamma": 0.95, "epsilon_decay": 0.995},
        {"alpha": 0.01, "gamma": 0.99, "epsilon_decay": 0.999},
        {"alpha": 0.1,  "gamma": 0.8,  "epsilon_decay": 0.99},
    ]

    n_episodes = 10000
    results = []

    for i, cfg in enumerate(configurations, start=1):
        print_training_parameters(i, cfg, n_episodes)

        Q, _ = train_q_learning(
            train_env,
            n_episodes=n_episodes,
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            epsilon=1.0,
            epsilon_decay=cfg["epsilon_decay"]
        )

        success_rate = evaluate_agent(train_env, Q)

        print("Policy:", np.argmax(Q, axis=1))
        print("Success rate:", success_rate)

        results.append({
            "alpha": cfg["alpha"],
            "gamma": cfg["gamma"],
            "epsilon_decay": cfg["epsilon_decay"],
            "success_rate": success_rate
        })

    # ----------------------------
    #
