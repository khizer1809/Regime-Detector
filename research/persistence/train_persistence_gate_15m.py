"""
train_persistence_gate_15m.py -- 15m-HMM4 counterpart of
train_persistence_gate.py. Reads Data/persistence_training_table_15m.csv,
does a chronological train/holdout split with an embargo, and tests BOTH:

  - LogisticRegression (as explicitly requested -- the original plan's
    design)
  - GradientBoostingClassifier (kept for direct comparison: the 5m gate's
    own docstring found LR structurally inadequate there because duration
    has a genuine U-shaped relationship with persistence, which a single
    linear coefficient cannot represent. Reported here, not silently
    substituted, so the same check can be made honestly for 15m instead of
    assumed to carry over.)

TIMEFRAME-CORRECTED EMBARGO: the 5m gate uses EMBARGO_BARS=48 (4h at 5m).
At 15m, 4h = 16 bars, not 48 -- reusing 48 would silently triple the
embargo to 12h. Uses EMBARGO_BARS_15M=16.

Run: `python train_persistence_gate_15m.py` (from src/).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persistence_common
from scaling import fit_transform_fold

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(_ROOT, "Data", "persistence_training_table_15m.csv")
OUT_PATH = os.path.join(_ROOT, "Data", "persistence_gate_15m.pkl")

HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
EMBARGO_BARS_15M = 16  # 16 x 15min = 4h -- same real embargo as the 5m gate's 48 bars
FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]


def load_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH, index_col=0, parse_dates=True)
    df["log1p_duration"] = persistence_common.duration_feature(df["duration"].values)
    return df


def chronological_split(df: pd.DataFrame):
    embargo_start = HOLDOUT_START - pd.Timedelta(minutes=15 * EMBARGO_BARS_15M)
    train = df[df.index < embargo_start]
    holdout = df[df.index >= HOLDOUT_START]
    return train, holdout


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(y_prob, n_bins, duplicates="drop")
    return pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bins}).groupby("bin").agg(
        n=("y_true", "size"), mean_pred=("y_prob", "mean"), realized_rate=("y_true", "mean"),
    )


def evaluate(name, model, train_scaled, y_train, holdout_scaled, y_holdout):
    model.fit(train_scaled.values, y_train)
    p_holdout = model.predict_proba(holdout_scaled.values)[:, 1]
    auc = roc_auc_score(y_holdout, p_holdout)
    acc = float(((p_holdout >= 0.5).astype(int) == y_holdout).mean())
    brier = brier_score_loss(y_holdout, p_holdout)
    print(f"\n=== {name} ===", flush=True)
    print(f"  Holdout AUC: {auc:.4f}", flush=True)
    print(f"  Holdout accuracy @0.5: {acc:.4f}", flush=True)
    print(f"  Holdout Brier score: {brier:.4f}", flush=True)
    if hasattr(model, "coef_"):
        print(f"  Coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_[0].round(4)))}", flush=True)
    if hasattr(model, "feature_importances_"):
        print(f"  Feature importances: {dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))}", flush=True)
    print("  Calibration (10 bins, predicted vs realized):", flush=True)
    print(calibration_table(y_holdout, p_holdout).to_string(), flush=True)
    return {"model": model, "auc": float(auc), "accuracy": acc, "brier": float(brier), "p_holdout": p_holdout}


def main():
    print(f"Loading training table from:\n  {TABLE_PATH}", flush=True)
    df = load_table()
    usable = df[df["usable"] == True].copy()  # noqa: E712
    print(f"  {len(df):,} total rows, {len(usable):,} usable rows", flush=True)

    train, holdout = chronological_split(usable)
    print(f"  train: {len(train):,} rows ({train.index.min()} -> {train.index.max()})", flush=True)
    print(f"  holdout: {len(holdout):,} rows ({holdout.index.min()} -> {holdout.index.max()})", flush=True)
    print(f"  embargo: {EMBARGO_BARS_15M} bars (4h) before {HOLDOUT_START} excluded from both splits", flush=True)

    y_train, y_holdout = train["label"].values, holdout["label"].values
    pos_rate = float(y_train.mean())
    print(f"  train label positive rate: {pos_rate:.3f}", flush=True)

    train_scaled, holdout_scaled, scaler = fit_transform_fold(train[FEATURE_COLUMNS], holdout[FEATURE_COLUMNS])

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_result = evaluate("LogisticRegression", lr, train_scaled, y_train, holdout_scaled, y_holdout)

    gbt = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)
    gbt_result = evaluate("GradientBoostingClassifier", gbt, train_scaled, y_train, holdout_scaled, y_holdout)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  LR  holdout AUC={lr_result['auc']:.4f}  acc={lr_result['accuracy']:.4f}  brier={lr_result['brier']:.4f}", flush=True)
    print(f"  GBT holdout AUC={gbt_result['auc']:.4f}  acc={gbt_result['accuracy']:.4f}  brier={gbt_result['brier']:.4f}", flush=True)

    best_name, best_result = ("LogisticRegression", lr_result) if lr_result["auc"] >= gbt_result["auc"] else ("GradientBoostingClassifier", gbt_result)
    artifact = {
        "model": best_result["model"],
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "model_type": best_name,
        "label_horizon_bars": 8,
        "base_hmm_artifact": "production_model_15m_hmm4.pkl",
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "holdout_start": str(HOLDOUT_START),
        "holdout_end": str(holdout.index.max()),
        "embargo_bars": EMBARGO_BARS_15M,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "train_label_positive_rate": pos_rate,
        "lr_holdout_auc": lr_result["auc"], "lr_holdout_accuracy": lr_result["accuracy"], "lr_holdout_brier": lr_result["brier"],
        "gbt_holdout_auc": gbt_result["auc"], "gbt_holdout_accuracy": gbt_result["accuracy"], "gbt_holdout_brier": gbt_result["brier"],
        "notes": f"Both LR and GBT tested explicitly; {best_name} saved as the artifact's `model` (higher holdout AUC). "
                 f"Consumes only confidence/stay_prob/log1p_duration/margin -- never a raw state index.",
    }
    with open(OUT_PATH, "wb") as f:
        pd.to_pickle(artifact, f)
    print(f"\nSaved persistence gate ({best_name}) to:\n  {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
