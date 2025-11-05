"""
Exercise 1.2: Decision Tree Regressor with Cross-Validation
UC3M Machine Learning Robotics – Tutorial 4

- Performs a real 5-fold cross-validation to estimate model performance.
- Trains and compares Default vs Tuned Decision Tree configurations.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)


def load_data():
    print("Loading data...")
    train_df = pd.read_csv("training_data.csv")
    test_df = pd.read_csv("test_data.csv")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df


def prepare_features(train_df, test_df):
    features = ["x", "y", "Angular_velocity"]
    X_train, y_train = train_df[features], train_df["Action"]
    X_test, y_test = test_df[features], test_df["Action"]
    return X_train, y_train, X_test, y_test


def cross_validate_tree(X, y, **params):
    """Manual 5-fold CV returning mean RMSE/MAE."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses, maes = [], []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = DecisionTreeRegressor(random_state=42, **params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        maes.append(mean_absolute_error(y_val, preds))

    return np.mean(rmses), np.mean(maes)


def evaluate_on_test(X_train, y_train, X_test, y_test, **params):
    model = DecisionTreeRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2, preds


def plot_results(y_test, y_pred, name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Action")
    plt.ylabel("Predicted Action")
    plt.title(f"{name} – Predicted vs Actual")

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted Action")
    plt.ylabel("Residuals")
    plt.title(f"{name} – Residual Plot")

    plt.tight_layout()
    plt.savefig(f"{name}_CV.png", dpi=150)
    print(f"Plot saved to {name}_CV.png")


def main():
    print("=" * 60)
    print("Exercise 1.2: Decision Tree Regressor with Cross-Validation")
    print("=" * 60)

    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)

    # Default model
    print("\n--- Default model ---")
    rmse_cv_d, mae_cv_d = cross_validate_tree(X_train, y_train)
    rmse_t_d, mae_t_d, r2_d, y_pred_d = evaluate_on_test(
        X_train, y_train, X_test, y_test
    )

    # Tuned model
    print("\n--- Tuned model ---")
    tuned_params = dict(max_depth=10, min_samples_split=5, min_samples_leaf=2)
    rmse_cv_t, mae_cv_t = cross_validate_tree(X_train, y_train, **tuned_params)
    rmse_t_t, mae_t_t, r2_t, y_pred_t = evaluate_on_test(
        X_train, y_train, X_test, y_test, **tuned_params
    )

    # Results table
    table = pd.DataFrame(
        {
            "Model": ["Default", "Tuned"],
            "CV_RMSE": [rmse_cv_d, rmse_cv_t],
            "CV_MAE": [mae_cv_d, mae_cv_t],
            "Test_RMSE": [rmse_t_d, rmse_t_t],
            "Test_MAE": [mae_t_d, mae_t_t],
            "Test_R2": [r2_d, r2_t],
        }
    )
    print("\n=== Results ===")
    print(table.round(4))

    table.to_csv("decision_tree_CV_results.csv", index=False)
    print("Saved to decision_tree_CV_results.csv")

    plot_results(y_test, y_pred_t, "DecisionTree_Tuned")

    print("\nExercise 1.2 completed!")


if __name__ == "__main__":
    main()
