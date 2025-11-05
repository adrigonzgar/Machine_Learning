"""
Exercise 3.2 – Hyperparameter tuning for MLP Regressor
UC3M Machine Learning Robotics – Tutorial 4
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# =====================================================
# 1. Load data
# =====================================================
train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["x", "y", "Angular_velocity"]]
y_train = train_df["Action"]
X_test = test_df[["x", "y", "Angular_velocity"]]
y_test = test_df["Action"]

# =====================================================
# 2. Normalize features
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 3. Define parameter grid for tuning
# =====================================================
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (64, 64), (64, 64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.0005, 0.001],
}

# =====================================================
# 4. Grid Search (3-fold CV)
# =====================================================
mlp = MLPRegressor(solver="adam", max_iter=500, random_state=42)

grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("\nRunning grid search over parameter combinations...")
grid.fit(X_train_scaled, y_train)

best_params = grid.best_params_
print("\n✅ Best parameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# =====================================================
# 5. Evaluate the best model on test data
# =====================================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Test set evaluation with best parameters ===")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# =====================================================
# 6. Save model, scaler, and results
# =====================================================
joblib.dump(best_model, "mlp_best_model.joblib")
joblib.dump(scaler, "mlp_scaler.joblib")
print("\n💾 Saved model: mlp_best_model.joblib")
print("💾 Saved scaler: mlp_scaler.joblib")

pd.DataFrame([{
    "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, **best_params
}]).to_csv("mlp_best_config.csv", index=False)
print("📊 Results saved to mlp_best_config.csv")

# =====================================================
# 7. Visualization
# =====================================================
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Action")
plt.ylabel("Predicted Action")
plt.title("Best MLP Regressor – Predicted vs Actual")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("MLP_BestConfig_results.png", dpi=150)
plt.show()
