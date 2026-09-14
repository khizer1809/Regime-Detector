"""
train_persistence_gate.py -- trains the persistence gate: a shallow
Gradient Boosted Tree that turns 4 HMM-derived features (confidence,
stay-probability, duration, posterior margin) into P(trend persists for
the next ~2h / 24 bars). Reads Data/persistence_training_table.csv (built
by persistence_dataset.py), does a chronological train/holdout split with
an embargo (no test-set peeking -- same house rule as the HMM's own
walk-forward validation), and writes Data/persistence_gate.pkl.

Tree model chosen over Logistic Regression because an earlier run showed
duration has a genuine U-SHAPED relationship with persistence (highest at
very short and very long durations, lowest in the middle) -- a linear
coefficient structurally cannot represent that; a shallow tree can split on
duration more than once and capture it. confidence/margin showed weaker,
mostly-monotonic relationships that a tree also captures without trouble.

Consumes only confidence/stay_prob/duration/margin -- never a raw HMM state
index -- so this gate should keep working unmodified across future monthly
HMM retrains without needing its own retrain in lockstep (state indices are
not stable across retrains, but these derived continuous features are).

Run: `python train_persistence_gate.py` (from src/).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import persistence_common
from infer_regime import load_artifact
from scaling import fit_transform_fold

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(_ROOT, "Data", "persistence_training_table.csv")
OUT_PATH = os.path.join(_ROOT, "Data", "persistence_gate.pkl")

N_STATES = 4
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
EMBARGO_BARS = 48  # 4h -- comfortably above LABEL_HORIZON_BARS=24, matches this project's existing embargo convention
FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]


def load_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH, index_col=0, parse_dates=True)
    df["log1p_duration"] = persistence_common.duration_feature(df["duration"].values)
    return df


def chronological_split(df: pd.DataFrame):
    embargo_start = HOLDOUT_START - pd.Timedelta(minutes=5 * EMBARGO_BARS)
    train = df[df.index < embargo_start]
    holdout = df[df.index >= HOLDOUT_START]
    return train, holdout


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(y_prob, n_bins, duplicates="drop")
    return pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bins}).groupby("bin").agg(
        n=("y_true", "size"), mean_pred=("y_prob", "mean"), realized_rate=("y_true", "mean"),
    )


def main():
    print(f"Loading training table from:\n  {TABLE_PATH}", flush=True)
    df = load_table()
    usable = df[df["usable"] == True].copy()  # noqa: E712 -- explicit bool compare matches the CSV's own semantics
    print(f"  {len(df):,} total rows, {len(usable):,} usable rows", flush=True)

    train, holdout = chronological_split(usable)
    print(f"  train: {len(train):,} rows ({train.index.min()} -> {train.index.max()})", flush=True)
    print(f"  holdout: {len(holdout):,} rows ({holdout.index.min()} -> {holdout.index.max()})", flush=True)
    print(f"  embargo: {EMBARGO_BARS} bars before {HOLDOUT_START} excluded from both splits", flush=True)

    y_train, y_holdout = train["label"].values, holdout["label"].values
    pos_rate = float(y_train.mean())
    print(f"  train label positive rate: {pos_rate:.3f}", flush=True)

    # Scaling is unnecessary for a tree model (split points are scale-invariant)
    # but kept for interface consistency with persistence_gate.py's live scoring path.
    train_scaled, holdout_scaled, scaler = fit_transform_fold(train[FEATURE_COLUMNS], holdout[FEATURE_COLUMNS])

    model = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(train_scaled.values, y_train)

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    print(f"\n  Feature importances: {importances}", flush=True)
    if max(importances.values()) < 1e-6:
        print("  WARNING: all feature importances are ~0 -- the model isn't using any feature. "
              "Investigate before trusting holdout metrics.", flush=True)

    p_holdout = model.predict_proba(holdout_scaled.values)[:, 1]
    auc = roc_auc_score(y_holdout, p_holdout)
    acc = float(((p_holdout >= 0.5).astype(int) == y_holdout).mean())
    brier = brier_score_loss(y_holdout, p_holdout)
    print(f"\n  Holdout AUC: {auc:.4f}", flush=True)
    print(f"  Holdout accuracy @0.5: {acc:.4f}", flush=True)
    print(f"  Holdout Brier score: {brier:.4f}", flush=True)
    if auc <= 0.5:
        print("  WARNING: AUC <= 0.5 -- the gate carries no information. Do not ship this model.", flush=True)

    print("\n  Calibration (10 bins, predicted vs realized):", flush=True)
    print(calibration_table(y_holdout, p_holdout).to_string(), flush=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "label_horizon_bars": persistence_common.LABEL_HORIZON_BARS,
        "base_hmm_artifact": f"production_model_hmm{N_STATES}.pkl",
        "base_hmm_train_end": load_artifact(N_STATES)["train_end"],
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "holdout_start": str(HOLDOUT_START),
        "holdout_end": str(holdout.index.max()),
        "embargo_bars": EMBARGO_BARS,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "train_label_positive_rate": pos_rate,
        "holdout_auc": float(auc),
        "holdout_accuracy": acc,
        "holdout_brier": float(brier),
        "notes": (
            "Consumes only confidence/stay_prob/duration/margin -- never a raw state index -- "
            "so this gate should keep working unmodified across future monthly HMM retrains; "
            "does NOT need retraining in lockstep with the HMM. Historical decode used a single "
            "production HMM fit on all history through base_hmm_train_end (see "
            "persistence_dataset.py for the accepted look-ahead caveat on confidence/margin "
            "for old bars)."
        ),
    }
    with open(OUT_PATH, "wb") as f:
        pd.to_pickle(artifact, f)
    print(f"\nSaved persistence gate to:\n  {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
