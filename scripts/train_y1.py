"""Train a single-output ANN to predict Y1 from X1,X2,X3 with strict spec.

Requirements implemented:
 - reads Sample_Data_Sheet.csv with columns X1,X2,X3,Y1..Y5
 - train_test_split(test_size=0.2, random_state=42)
 - pipeline: StandardScaler -> MLPRegressor with provided hyperparams
 - prints train/val MAE, RMSE, R^2
 - saves models/y1_baseline/model.joblib and models/y1_baseline/scaler.joblib
 - saves plot to reports/figures/y1_pred_vs_actual.png
 - exits with non-zero code if validation R^2 < 0.6
"""

import sys
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main(csv_path: Path):
    # directories
    model_dir = Path("models/y1_baseline")
    figs_dir = Path("reports/figures")
    model_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_csv(csv_path)
    # Normalize column names coming from the provided sheet
    # Normalize column names: trim + collapse inner spaces
    df.columns = [' '.join(c.strip().split()) for c in df.columns]

    rename_map = {
    "X1 (Mpa)": "X1", "X1 (MPa)": "X1", "X1(Mpa)": "X1",
    "X2 (Celcius)": "X2", "X2 (Celsius)": "X2", "X2(Celsius)": "X2",
    "X3 (seconds)": "X3", "X3(seconds)": "X3", "X3 (s)": "X3",
    "Y1(%)": "Y1", "Y1 (%)": "Y1",
    "Y2(%)": "Y2", "Y2 (%)": "Y2",
    "Y3(-)": "Y3", "Y3 (-)": "Y3",
    "Y4(%)": "Y4", "Y4 (%)": "Y4",
    "Y5(%)": "Y5", "Y5 (%)": "Y5",
    }
    df = df.rename(columns=rename_map)

    required = ["X1", "X2", "X3", "Y1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Still missing columns after rename: {missing}. Found: {df.columns.tolist()}")


    X = df[["X1", "X2", "X3"]].copy()
    y = df["Y1"].copy()

    # drop NA
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[mask]
    y = y.loc[mask]

    # train/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )

    # from trainval, create a train/val split for early evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42
    )

    # scaler fit on train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # MLP with exact params
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=32,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.2,
        random_state=42,
        verbose=False,
    )

    # train
    mlp.fit(X_train_s, y_train)

    # predictions
    y_train_pred = mlp.predict(X_train_s)
    y_val_pred = mlp.predict(X_val_s)
    y_test_pred = mlp.predict(X_test_s)

    # metrics
    train_mae, train_rmse, train_r2 = calc_metrics(y_train, y_train_pred)
    val_mae, val_rmse, val_r2 = calc_metrics(y_val, y_val_pred)
    test_mae, test_rmse, test_r2 = calc_metrics(y_test, y_test_pred)

    print("=== Y1 MLP Training Results ===")
    print(f"Train MAE: {train_mae:.4f}  RMSE: {train_rmse:.4f}  R2: {train_r2:.4f}")
    print(f"Val   MAE: {val_mae:.4f}  RMSE: {val_rmse:.4f}  R2: {val_r2:.4f}")
    print(f"Test  MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  R2: {test_r2:.4f}")

    # save model and scaler separately as requested
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(mlp, model_path)
    joblib.dump(scaler, scaler_path)

    # save diagnostic plot: val predicted vs actual
    fig_path = figs_dir / "y1_pred_vs_actual.png"
    plt.figure(figsize=(6, 4))
    lo = min(y_val.min(), y_val_pred.min())
    hi = max(y_val.max(), y_val_pred.max())
    plt.scatter(y_val, y_val_pred, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], color='red', linestyle='--')
    plt.xlabel('Actual Y1')
    plt.ylabel('Predicted Y1')
    plt.title('Y1: Predicted vs Actual (Validation)')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved plot: {fig_path}")

    # write a small metrics file
    metrics_path = model_dir / "metrics.txt"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"train_mae={train_mae}\n")
        f.write(f"train_rmse={train_rmse}\n")
        f.write(f"train_r2={train_r2}\n")
        f.write(f"val_mae={val_mae}\n")
        f.write(f"val_rmse={val_rmse}\n")
        f.write(f"val_r2={val_r2}\n")
        f.write(f"test_mae={test_mae}\n")
        f.write(f"test_rmse={test_rmse}\n")
        f.write(f"test_r2={test_r2}\n")

    # exit non-zero if val R2 < 0.6
    if val_r2 < 0.6:
        print(f"Validation R^2 = {val_r2:.4f} is below threshold 0.6. Exiting with non-zero status.")
        sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("Sample_Data_Sheet.csv"), help="Path to CSV (expects columns X1,X2,X3,Y1..Y5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.csv)
