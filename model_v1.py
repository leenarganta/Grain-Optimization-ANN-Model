# filename: model_v1.py
# ------------------------------------------------------------
# GOAL: Learn f: (X1 Mpa, X2 Celcius, X3 seconds) -> Y1(%)
# MODEL: Small feed-forward ANN (MLP) for tabular regression
# WHY: Nonlinear mapping, interactions between process variables
# ------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main(csv_path: str):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # --- 1) VERIFY & CLEAN COLUMN NAMES -------------------------------------
    # Keep your exact headers; just strip spaces so accidental trailing spaces don't bite
    df.columns = [c.strip() for c in df.columns]

    needed = ["X1 (Mpa)", "X2(Celcius)", "X3 (seconds)", "Y1(%)"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}. Found: {df.columns.tolist()}")

    # --- 2) SELECT FEATURES & TARGET ----------------------------------------
    # Why: separate predictors (X) and response (y) so the pipeline can scale X only.
    X = df[["X1 (Mpa)", "X2(Celcius)", "X3 (seconds)"]].copy()
    y = df["Y1(%)"].copy()

    # Sanity checks (catch weird entries early)
    if X.isna().any().any() or y.isna().any():
        print("WARNING: Missing values found. Dropping rows with NA.")
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X.loc[mask], y.loc[mask]

    # --- 3) TRAIN/TEST SPLIT -------------------------------------------------
    # Why: honest generalization check; test is never used for training or early stopping.
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.20, random_state=42
    )

    # --- 4) PIPELINE: SCALE -> MLP ------------------------------------------
    # Why scaling? Inputs are on different scales (MPa vs seconds), which destabilizes training.
    # Early stopping: stops when val score plateaus, preventing overfitting on small data.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),   # capacity: 64->32 neurons (Step-up if underfitting)
            activation="relu",             # nonlinearity to learn complex patterns
            solver="adam",                 # adaptive SGD (stable, good default)
            learning_rate_init=1e-3,       # start here; tune if needed
            max_iter=2000,                 # allow enough iterations
            random_state=42,
            early_stopping=True,           # critical for small datasets
            validation_fraction=0.2,       # internal val split from TRAIN only
            n_iter_no_change=20,           # patience
            verbose=False
        ))
    ])

    # --- 5) TRAIN ------------------------------------------------------------
    pipe.fit(X_train, y_train)

    # --- 6) EVALUATE ON HELD-OUT TEST ---------------------------------------
    y_pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2   = r2_score(y_test, y_pred)

    print("\n=== Y1 Single-Output ANN (Test) ===")
    print(f"MAE  : {mae:.3f}  (avg absolute error in Y1 units)")
    print(f"RMSE : {rmse:.3f}  (penalizes big errors)")
    print(f"R^2  : {r2:.3f}  (variance explained; closer to 1 is better)")

    # --- 7) DIAGNOSTIC PLOT --------------------------------------------------
    plt.figure()
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([lo, hi], [lo, hi])  # 45° ideal line
    plt.xlabel("Actual Y1 (%)")
    plt.ylabel("Predicted Y1 (%)")
    plt.title("Y1: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("y1_pred_vs_actual.png", dpi=160)
    print("Saved plot: y1_pred_vs_actual.png")

    # --- 8) SAVE ARTIFACTS ---------------------------------------------------
    # Saving the whole pipeline means scaling + model are preserved together for inference.
    joblib.dump(pipe, "ann_y1_pipeline.joblib")
    print("Saved model: ann_y1_pipeline.joblib")

if __name__ == "__main__":
    # Example: python train_y1_ann.py --csv "Sample Data Sheet.csv"
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    args = p.parse_args()
    main(args.csv)
