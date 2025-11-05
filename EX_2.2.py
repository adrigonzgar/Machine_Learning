"""
Exercise 2 – 3D Visualization of the State Space
UC3M Machine Learning Robotics – Tutorial 4
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
train_df = pd.read_csv("training_data.csv")

# Create 3D plot
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot: position (x, y) and applied torque (Action)
p = ax.scatter(
    train_df["x"],
    train_df["y"],
    train_df["Angular_velocity"],
    c=train_df["Action"],
    cmap="coolwarm",
    alpha=0.8,
    s=15
)

# Labels
ax.set_xlabel("x (sinθ)")
ax.set_ylabel("y (cosθ)")
ax.set_zlabel("Angular velocity")
fig.colorbar(p, label="Action (torque)")
plt.title("3D Visualization of the State Space – Training Data")

# Save figure
plt.tight_layout()
plt.savefig("LinearRegression_state_space_3D.png", dpi=150)
plt.show()
print("3D state space plot saved to LinearRegression_state_space_3D.png")
