# ------------------------------------------------------------
# GOAL: Learn f: (X1 Mpa, X2 Celcius, X3 seconds) -> Y1(%)
# MODEL: Small feed-forward ANN (MLP) for tabular regression
# WHY: Nonlinear mapping, interactions between process variables
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Hardcoded path to sample data CSV file
    csv_path = Path("Sample_Data_Sheet.csv")
    
    # --- 1) LOAD DATA -------------------------------------------------------
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(csv_path)

    # --- 2) VERIFY & CLEAN COLUMN NAMES -------------------------------------
    # Strip spaces from column headers to avoid issues with trailing spaces
    df.columns = [c.strip() for c in df.columns]

    # Required columns for features and target
    needed = ["X1 (Mpa)", "X2(Celcius)", "X3 (seconds)", "Y1(%)"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}. Found: {df.columns.tolist()}")

    # --- 3) SELECT FEATURES & TARGET ----------------------------------------
    # Separate predictors (X) and response (y) for modeling
    X = df[["X1 (Mpa)", "X2(Celcius)", "X3 (seconds)"]].copy()
    y = df["Y1(%)"].copy()

    # --- 4) HANDLE MISSING VALUES --------------------------------------------
    # Check for missing values and drop rows with any NA values
    if X.isna().any().any() or y.isna().any():
        print("WARNING: Missing values found. Dropping rows with NA.")
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X.loc[mask], y.loc[mask]

    # --- 5) TRAIN/TEST SPLIT -------------------------------------------------
    # Split data into training and testing sets (80% train, 20% test)
    # This ensures honest evaluation on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.20, random_state=42
    )

    # --- 6) PIPELINE: SCALE FEATURES & TRAIN MLP -----------------------------
    # Create a pipeline that standardizes features and trains an MLP regressor
    # Scaling is important because features have different units and scales
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),   # Two hidden layers with 64 and 32 neurons
            activation="relu",             # ReLU activation for nonlinearity
            solver="adam",                 # Adam optimizer for training
            learning_rate_init=1e-3,       # Initial learning rate
            max_iter=2000,                 # Maximum iterations for convergence
            random_state=42,
            early_stopping=True,           # Stop early if validation score doesn't improve
            validation_fraction=0.2,       # Use 20% of training data for validation
            n_iter_no_change=20,           # Number of iterations with no improvement to wait before stopping
            verbose=False
        ))
    ])

    # --- 7) TRAIN THE MODEL --------------------------------------------------
    pipe.fit(X_train, y_train)

    # --- 8) EVALUATE MODEL PERFORMANCE ON TEST SET ---------------------------
    y_pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2   = r2_score(y_test, y_pred)

    print("\n=== Y1 Single-Output ANN (Test) ===")
    print(f"MAE  : {mae:.3f}  (avg absolute error in Y1 units)")
    print(f"RMSE : {rmse:.3f}  (penalizes big errors)")
    print(f"R^2  : {r2:.3f}  (variance explained; closer to 1 is better)")

    # --- 9) DIAGNOSTIC PLOT --------------------------------------------------
    # Plot predicted vs actual values to visually assess model performance
    plt.figure()
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], color='red')  # 45° ideal line
    plt.xlabel("Actual Y1 (%)")
    plt.ylabel("Predicted Y1 (%)")
    plt.title("Y1: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("y1_pred_vs_actual.png", dpi=160)
    print("Saved plot: y1_pred_vs_actual.png")

    # --- 10) SAVE MODEL PIPELINE ---------------------------------------------
    # Save the entire pipeline (scaler + model) for later inference
    joblib.dump(pipe, "ann_y1_pipeline.joblib")
    print("Saved model: ann_y1_pipeline.joblib")

    # Return the trained pipeline for further use if needed
    return pipe

if __name__ == "__main__":
    # Run main function and get trained model pipeline
    model_pipeline = main()
