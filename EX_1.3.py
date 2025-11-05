"""
Exercise 1.3: Decision Tree Regressor – Retrain with All Training Data
UC3M Machine Learning Robotics – Tutorial 4

This script:
- Loads the same dataset used in Exercise 1.2.
- Retrains the best model (Decision Tree with tuned hyperparameters)
  using ALL the training data (no cross-validation).
- Evaluates performance on the test set.
- Compares results with the average cross-validation metrics
  from Exercise 1.2 to quantify the difference.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import joblib

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# Data loading and preprocessing
# ============================================================

def load_data():
    """Load training and test data from CSV files."""
    print("Loading data...")
    train_df = pd.read_csv("training_data.csv")
    test_df = pd.read_csv("test_data.csv")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Extract features (X) and target (y)."""
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

def train_and_evaluate(X_train, y_train, X_test, y_test, **params):
    """Train Decision Tree on all training data (no CV) and evaluate."""
    print("\n" + "=" * 60)
    print("Training Decision Tree Regressor on ALL training data...")
    print(f"Parameters: {params}")

    model = DecisionTreeRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")

    return model, y_pred, {"test_rmse": rmse, "test_mae": mae, "test_r2": r2}


# ============================================================
# Visualization
# ============================================================

def plot_results(y_test, y_pred, save_path="DecisionTree_noCV.png"):
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
    print("Exercise 1.3: Retraining Best Model with All Training Data")
    print("=" * 60)

    # Load data
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

    # Tuned parameters from Exercise 1.2
    best_params = dict(max_depth=10, min_samples_split=5, min_samples_leaf=2)

    # Train and evaluate on full training data
    model, y_pred, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, **best_params)

    joblib.dump(model, "decision_tree_model.joblib")
    print("✅ Saved tuned Decision Tree model to decision_tree_model.joblib")
    # ============================================================
    # Quantitative comparison with CV metrics (from Exercise 1.2)
    # ============================================================
    prev_rmse = 1.3269   # Cross-validation mean RMSE
    prev_mae  = 0.9960   # Cross-validation mean MAE
    prev_r2   = None     # Not computed in CV

    diff_rmse = abs(metrics["test_rmse"] - prev_rmse)
    diff_mae  = abs(metrics["test_mae"] - prev_mae)

    print("\n" + "=" * 60)
    print("Quantitative Comparison (CV mean vs No-CV)")
    print("=" * 60)
    print(f"CV RMSE (Exercise 1.2):     {prev_rmse:.4f}")
    print(f"No-CV Test RMSE (1.3):      {metrics['test_rmse']:.4f}")
    print(f"ΔRMSE:                      {diff_rmse:.4f}\n")
    print(f"CV MAE (Exercise 1.2):      {prev_mae:.4f}")
    print(f"No-CV Test MAE (1.3):       {metrics['test_mae']:.4f}")
    print(f"ΔMAE:                       {diff_mae:.4f}")

    # Save comparison to CSV
    comparison_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE"],
        "CV_mean": [prev_rmse, prev_mae],
        "No_CV_Test": [metrics["test_rmse"], metrics["test_mae"]],
        "Difference": [diff_rmse, diff_mae]
    })
    comparison_df.to_csv("decision_tree_noCV_results.csv", index=False)
    print("\nResults saved to decision_tree_noCV_results.csv")

    # Plot
    plot_results(y_test, y_pred)

    print("\n" + "=" * 60)
    print("Exercise 1.3 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

    