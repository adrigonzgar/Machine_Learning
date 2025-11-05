"""
Exercise 3: Multi-Layer Perceptron (MLP Regressor)
UC3M Machine Learning Robotics – Tutorial 4

This script trains and evaluates an MLP Regressor to predict
the continuous target variable 'Action' (torque) from state features
(x, y, Angular_velocity). It compares performance with previous models.
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# Data loading and preprocessing
# ============================================================

def load_data():
    """Load training and test datasets."""
    print("Loading data...")
    train_df = pd.read_csv("training_data.csv")
    test_df = pd.read_csv("test_data.csv")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Split into features (X) and target (y)."""
    features = ["x", "y", "Angular_velocity"]
    X_train, y_train = train_df[features], train_df["Action"]
    X_test, y_test = test_df[features], test_df["Action"]
    print(f"\nFeature columns: {features}")
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set:     X={X_test.shape}, y={y_test.shape}")
    return X_train, y_train, X_test, y_test


# ============================================================
# Model training and evaluation
# ============================================================

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train an MLP Regressor and evaluate on the test set."""
    print("\n" + "=" * 60)
    print("Training MLP Regressor...")
    print("=" * 60)

    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    return model, y_pred, {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ============================================================
# Visualization
# ============================================================

def plot_results(y_test, y_pred, save_path="MLPRegressor_results.png"):
    """Plot predicted vs actual values and residuals."""
    plt.figure(figsize=(10, 5))

    # Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual Action")
    plt.ylabel("Predicted Action")
    plt.title("Predicted vs Actual Values")
    plt.legend()
    plt.grid(alpha=0.3)

    # Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted Action")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")


# ============================================================
# Main execution
# ============================================================

def main():
    print("=" * 60)
    print("Exercise 3: MLP Regressor")
    print("=" * 60)

    # Load and prepare data
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

    # Train model and evaluate
    model, y_pred, metrics = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Save results
    results_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R²"],
        "MLPRegressor": [
            metrics["mse"],
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"]
        ],
        "DecisionTree_Tuned": [None, 0.4909, 0.2272, 0.5948],
        "LinearRegression": [0.6660, 0.8161, 0.7053, -0.1200]
    })
    results_df.to_csv("mlp_results.csv", index=False)
    print("\nResults saved to mlp_results.csv")

    # Plot
    plot_results(y_test, y_pred)

    print("\n" + "=" * 60)
    print("Exercise 3 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
