"""
Exercise 4 – Final Full Version (Different Start per Model)
UC3M Machine Learning Robotics – Tutorial 4

✅ Linear Regression & Decision Tree → random start (anywhere)
✅ MLP Deep → start near θ≈π (downward)
✅ Real-time render + saved plots
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import pandas as pd
import gymnasium as gym
import joblib
import time
import matplotlib.pyplot as plt

# ============================================================
# Configuración
# ============================================================
N_EPISODES = 3
MAX_STEPS = 300
OUTPUT_GAIN = 1.9
RENDER = True
RENDER_DELAY = 0.02
NOISE_THETA = 0.05

# ============================================================
# Modelos
# ============================================================
env = gym.make("Pendulum-v1", render_mode="human" if RENDER else None)
np.random.seed(42)

print("\n🚀 Loading trained models...\n")
linear_model = joblib.load("linear_regression_model.joblib")
tree_model = joblib.load("decision_tree_model.joblib")
mlp_model = joblib.load("mlp_best_model.joblib")
scaler = joblib.load("mlp_scaler.joblib")

models = {
    "Linear Regression": {"model": linear_model, "type": "linear"},
    "Decision Tree": {"model": tree_model, "type": "tree"},
    "MLP Deep": {"model": mlp_model, "type": "mlp"}
}

# ============================================================
# Funciones auxiliares
# ============================================================
def reset_random_start(env):
    """Reset aleatorio en todo el rango del péndulo."""
    obs, _ = env.reset()
    theta0 = np.random.uniform(-np.pi, np.pi)
    theta_dot0 = np.random.uniform(-1.0, 1.0)
    env.unwrapped.state = (theta0, theta_dot0)
    obs = np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)
    return obs

def reset_downward_start(env, noise_theta=0.05):
    """Reset cerca de la posición hacia abajo (θ≈π)."""
    obs, _ = env.reset()
    theta0 = np.pi + np.random.uniform(-noise_theta, noise_theta)
    theta_dot0 = 0.0
    env.unwrapped.state = (theta0, theta_dot0)
    obs = np.array([np.cos(theta0), np.sin(theta0), theta_dot0], dtype=np.float32)
    return obs

def run_agent(env, model, model_type, scaler, n_episodes=3, max_steps=200, gain=1.0, render=False):
    """Ejecuta el modelo sobre el entorno."""
    total_rewards, all_trajs = [], []

    for ep in range(n_episodes):
        # Selección de inicio según tipo de modelo
        if model_type == "mlp":
            obs = reset_downward_start(env)
        else:
            obs = reset_random_start(env)

        ep_reward = 0
        traj = {"time": [], "sin_theta": [], "torque": []}

        for t in range(max_steps):
            x, y, ang_vel = np.array(obs).flatten()
            features = pd.DataFrame([[x, y, ang_vel]],
                                    columns=["x", "y", "Angular_velocity"])

            if model_type == "mlp":
                features = scaler.transform(features)
                action_pred = model.predict(features)[0] * gain
            else:
                action_pred = model.predict(features)[0]

            action = float(np.clip(action_pred, -2, 2))
            obs, reward, terminated, truncated, _ = env.step([action])
            ep_reward += reward

            traj["time"].append(t)
            traj["sin_theta"].append(y)
            traj["torque"].append(action)

            if render:
                env.render()
                time.sleep(RENDER_DELAY)
            if terminated or truncated:
                break

        print(f"{model_type.upper()} | Episode {ep+1}: Total Reward = {ep_reward:.2f}")
        total_rewards.append(ep_reward)
        all_trajs.append(traj)

    return np.mean(total_rewards), np.std(total_rewards), all_trajs

# ============================================================
# Ejecución
# ============================================================
results = []
trajectories = {}

for name, info in models.items():
    gain = OUTPUT_GAIN if info["type"] == "mlp" else 1.0
    print(f"\n==================== {name} ====================")
    mean_r, std_r, trajs = run_agent(env, info["model"], info["type"],
                                     scaler, n_episodes=N_EPISODES,
                                     max_steps=MAX_STEPS, gain=gain, render=RENDER)
    results.append({"Model": name, "Mean Reward": mean_r, "Std Dev": std_r})
    trajectories[name] = trajs

# ============================================================
# Resultados y gráficos
# ============================================================
df_results = pd.DataFrame(results)
df_results.to_csv("EX4_results_log.csv", index=False)
print("\n📁 Results saved to EX4_results_log.csv")

print("\n==================== SUMMARY ====================")
print(df_results.to_string(index=False))

# ---- Gráfico 1: Recompensas ----
plt.figure(figsize=(6,4))
plt.bar(df_results["Model"], df_results["Mean Reward"],
        yerr=df_results["Std Dev"], capsize=5,
        color=['skyblue','salmon','limegreen'])
plt.ylabel("Mean Reward (↑ better)")
plt.title("Model Performance Comparison")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("EX4_rewards_comparison.png", dpi=150)
print("📊 Saved: EX4_rewards_comparison.png")

# ---- Gráfico 2: Trayectorias (sinθ) ----
plt.figure(figsize=(8,5))
for name in models.keys():
    traj = trajectories[name][0]
    plt.plot(traj["time"], traj["sin_theta"], label=name)
plt.xlabel("Time step")
plt.ylabel("sin(θ)")
plt.title("Pendulum Angle Evolution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("EX4_trajectories.png", dpi=150)
print("📈 Saved: EX4_trajectories.png")

# ---- Gráfico 3: Torque ----
plt.figure(figsize=(8,5))
for name in models.keys():
    traj = trajectories[name][0]
    plt.plot(traj["time"], traj["torque"], label=name)
plt.xlabel("Time step")
plt.ylabel("Torque")
plt.title("Torque Applied Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("EX4_torque_comparison.png", dpi=150)
print("🔧 Saved: EX4_torque_comparison.png")

plt.close('all')
env.close()
print("\n✅ Simulation and visualization completed successfully.\n")
