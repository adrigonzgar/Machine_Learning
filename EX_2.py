"""
Exercise 2: Linear Regression
UC3M Machine Learning Robotics – Tutorial 4

This script trains and evaluates a Linear Regression model
to predict the continuous target variable 'Action' (torque)
from the state features (x, y, Angular_velocity).

It compares the results with the Decision Tree Regressor
from Exercise 1.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set random seed
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
    """Train Linear Regression model and evaluate on test set."""
    print("\n" + "=" * 60)
    print("Training Linear Regression model...")
    print("=" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Coefficients
    print("\nModel coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"  {feature}: {coef:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    # Predictions and metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation metrics:")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")

    return model, y_pred, {"test_rmse": rmse, "test_mae": mae, "test_r2": r2}


# ============================================================
# Visualization
# ============================================================

def plot_results(y_test, y_pred, save_path="LinearRegression_results.png"):
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
    print("Exercise 2: Linear Regression")
    print("=" * 60)

    # Load and prepare data
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

    # Train model and evaluate
    model, y_pred, metrics = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Save results
    results_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R²"],
        "LinearRegression": [
            metrics["test_rmse"],
            metrics["test_mae"],
            metrics["test_r2"]
        ],
        "DecisionTree_Tuned": [0.4909, 0.2272, 0.5948]
    })
    results_df.to_csv("linear_regression_results.csv", index=False)
    print("\nResults saved to linear_regression_results.csv")

    # Plot
    plot_results(y_test, y_pred)

    print("\n" + "=" * 60)
    print("Exercise 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
