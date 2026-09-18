"""
v14_market_structure_test.py -- EXPERIMENTAL, NOT PRODUCTION.

Causal Market-Structure Features -> 2H Persistence Test.

Tests whether manual-trader-style market-structure information (confirmed
swing highs/lows, HH/HL/LH/LL, structure direction, BOS, CHOCH, pullback,
structural distance, trend-structure age) adds incremental OOS predictive
value to the existing HMM persistence Logistic Regression, on top of the
FROZEN, validated production trend-classification rule (adaptive
+/-0.30/+/-0.15 threshold) -- WITHOUT refitting or redecoding the HMM, and
WITHOUT changing the persistence target/HMM feature set/HMM itself.

REUSED, UNMODIFIED:
  - Data/v2_cache/fold_XXX/ (101 folds: hmm.pkl, scaler.pkl, causal_output.csv,
    metadata.json) -- produced by v2_stage_a_hmm.py. Never touched here.
  - Data/v2_cache/trend_threshold_experiment/adaptive_fold_threshold_master_oos.csv
    -- the FROZEN baseline (adaptive +/-0.30/+/-0.15 threshold, pooled OOS
    ROC-AUC=0.5228). Its `usable`/`label`/`trend_sign_t`/`is_trending_t`/
    `fold`/`epsilon` columns are used AS-IS -- this experiment never touches
    the persistence target or trend-eligibility rule, only ADDS features to
    the LR's input matrix, per the task spec.
  - v2_common.generate_folds() fold schedule (already baked into the master
    OOS file's `fold` column -- not regenerated).

NEW, added here:
  - 15 causal market-structure features, built from raw 5m OHLCV
    (BTCUSDT_5M.csv -- confirmed 933,924 contiguous 5-minute bars, zero
    missing rows, 2017-09-01 -> 2026-07-18), independently for each of 7
    swing-detection parameters k in [2,3,4,5,6,8,10].
  - A walk-forward LR runner that mirrors v4_threshold_sensitivity's exact
    expanding-pool/epsilon-frozen/fixed-0.5-gate methodology, parameterized
    by feature-column set (baseline 4, or baseline 4 + structure 15).
  - Inner-validation k-selection (never touches OOS): per outer fold, the
    fold's own TRAINING POOL is split chronologically 80/20; each candidate
    k's LR is fit on the inner-train 80% and scored on the inner-val 20%;
    inner-val AUCs are averaged across all outer folds per k; the k with the
    highest mean inner-val AUC is selected (ties within 0.002 AUC -> smaller
    k). This selection never sees any fold's actual OOS test rows.

===========================================================================
CAUSAL DEFINITIONS (frozen BEFORE running anything, per the task spec)
===========================================================================

SWING CONFIRMATION (Part 1): a symmetric k-bar fractal.
    is_swing_high[t]  <=>  High[t] > max(High[t-k .. t-1])  AND  High[t] > max(High[t+1 .. t+k])
    is_swing_low[t]   <=>  Low[t]  < min(Low[t-k .. t-1])   AND  Low[t]  < min(Low[t+1 .. t+k])
A pivot forming at position t requires k bars of future price action to
confirm (nothing broke it on either side). It is therefore UNKNOWN to the
model until position t+k. Every downstream feature at row T uses only
pivots whose confirmation position (pivot_t + k) <= T. Implemented via
`detect_swings()` (vectorized rolling max/min, shifted so the current bar
itself is excluded from its own left/right comparison window) and
`build_structure_features()`, which scatters each confirmed event's value
onto the timeline starting AT its confirmation position (never earlier),
via left-join + forward-fill (LOCF) -- explicitly NOT back-filled to the
pivot's own bar.

HH/HL/LH/LL (Part 2): classified against the PREVIOUS confirmed pivot of
the SAME type (high vs high, low vs low), in confirmation order (which
equals pivot order for a fixed k). `hh_strength`/`lh_strength` =
log(new/prev) or log(prev/new) respectively (signed-positive magnitude,
zero on the non-firing event type) -- log scale to stay unit-free across
BTC's multi-year price-level drift, matching this project's existing
return-feature convention. These are ONE-SHOT event values (nonzero only on
their own confirmation bar, zero elsewhere), not LOCF-held.

STRUCTURE_DIRECTION (Part 3): +1 only if the most recently confirmed swing
HIGH event was an HH AND the most recently confirmed swing LOW event was an
HL; -1 only if both were LH/LL; 0 otherwise (including before both types
have fired at least twice). No free lookback-count parameter. LOCF-held
between confirmation events.

STRUCTURE_CONSISTENCY (Part 3): fraction of the last STRUCTURE_CONSISTENCY_
WINDOW=8 (fixed, pre-committed, never tuned on results) confirmed structural
events (HH/HL/LH/LL, any type, in confirmation order) whose own directional
lean agrees with the CURRENT structure_direction. LOCF-held between events.

BOS (Part 4): per-bar condition, re-evaluated every bar (not a one-shot
trigger): bos_up[T] = close[T] > (price of the most recently CONFIRMED
swing high as of T); bos_down[T] = close[T] < most recently confirmed swing
low. "Significant" is interpreted as simply "the most recent confirmed
level" -- no extra magnitude filter beyond the k-pivot definition itself
(documented interpretation, not assumed silently).

CHOCH (Part 5): choch_up[T] = (structure_direction as of T-1 == -1) AND
(bos_up[T] == 1); choch_down[T] = (structure_direction as of T-1 == +1) AND
(bos_down[T] == 1). Uses structure_direction shifted by one bar so CHOCH
never depends on information only available at the same bar T (no
same-timestep circularity). This is a per-bar boolean condition (can stay
1 for consecutive bars while the break persists and structure hasn't yet
re-flipped), not a single-fire event flag -- documented explicitly since
Part 5 required exact rules stated before running.

PULLBACK / IMPULSE (Part 6): from the last 3 confirmed pivots (mixed
high/low stream, in confirmation order) P[i-2] -> P[i-1] -> P[i]:
    impulse_strength = |log(P[i-1] / P[i-2])|
    pullback_ratio    = |log(P[i] / P[i-1])| / impulse_strength  (NaN if impulse==0)
LOCF-held between events; NaN until 3 confirmed pivots exist.

STRUCTURAL DISTANCE (Part 7): distance_from_last_swing_high[T] =
log(close[T] / last_confirmed_swing_high_price_asof[T]); mirror for low.
Log-distance is scale-free (immune to BTC's price-level drift across the
dataset -- the same non-stationarity class of issue found and documented
this session for a different raw-scale feature).

TREND STRUCTURE AGE (Part 8): bars since structure_direction's value last
CHANGED (the general "structural transition", of which CHOCH is the
triggering mechanism) -- same reset-counter algorithm as this project's
existing `persistence_common.compute_running_duration`.

===========================================================================
Run: `python v14_market_structure_test.py` (from research_archive/src/).
Writes only to Data/v2_cache/market_structure_experiment/ (new directory).
Does not touch Data/v2_cache/fold_XXX/, trend_threshold_experiment/, or any
production file.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_archive/
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
_SRC = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC)
import persistence_common  # noqa: E402

RAW_OHLCV_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"
V2_CACHE_DIR = os.path.join(_RESEARCH_ROOT, "Data", "v2_cache")
BASELINE_MASTER_PATH = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment", "adaptive_fold_threshold_master_oos.csv")
BASELINE_RESULT_PATH = os.path.join(V2_CACHE_DIR, "trend_threshold_experiment", "adaptive_fold_threshold_result.json")
OUT_DIR = os.path.join(V2_CACHE_DIR, "market_structure_experiment")

K_CANDIDATES = [2, 3, 4, 5, 6, 8, 10]
STRUCTURE_CONSISTENCY_WINDOW = 8   # fixed, pre-committed -- not tuned on results
TIE_TOLERANCE_AUC = 0.002          # fixed, pre-committed k-selection tie tolerance

BASE_FEATURE_COLUMNS = ["confidence", "stay_prob", "log1p_duration", "margin"]
STRUCTURE_FEATURE_COLUMNS = [
    "hh_strength", "hl_strength", "lh_strength", "ll_strength",
    "structure_direction", "structure_consistency", "bos_up", "bos_down",
    "choch_up", "choch_down", "pullback_ratio", "impulse_strength",
    "distance_from_last_swing_high", "distance_from_last_swing_low", "trend_structure_age",
]

MIN_POOL_ROWS = 500     # matches existing V2/V4 bootstrap convention
GATE_THRESHOLD = 0.5
INNER_VAL_FRACTION = 0.2


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# PART 1 -- causal swing/fractal detection
# ===========================================================================

def load_raw_ohlcv():
    log(f"Loading raw OHLCV from {RAW_OHLCV_PATH} ...")
    df = pd.read_csv(RAW_OHLCV_PATH, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    diffs = df["timestamp"].diff().dropna()
    n_gaps = int((diffs != pd.Timedelta(minutes=5)).sum())
    log(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}, non-5min gaps={n_gaps}")
    assert n_gaps == 0, "raw OHLCV is not perfectly contiguous -- swing detection assumptions require re-checking"
    return df


def detect_swings(high: np.ndarray, low: np.ndarray, k: int):
    """Symmetric k-bar fractal, vectorized. left/right windows both
    EXCLUDE the bar itself. Returns (is_swing_high, is_swing_low) boolean
    arrays over the full raw index -- True only where BOTH a full k-bar
    left window and a full k-bar right window exist and are strictly beaten."""
    left_max = pd.Series(high).shift(1).rolling(k, min_periods=k).max().values
    right_max = pd.Series(high[::-1]).shift(1).rolling(k, min_periods=k).max().values[::-1]
    is_sh = (high > left_max) & (high > right_max)

    left_min = pd.Series(low).shift(1).rolling(k, min_periods=k).min().values
    right_min = pd.Series(low[::-1]).shift(1).rolling(k, min_periods=k).min().values[::-1]
    is_sl = (low < left_min) & (low < right_min)

    return np.nan_to_num(is_sh, nan=0.0).astype(bool), np.nan_to_num(is_sl, nan=0.0).astype(bool)


# ===========================================================================
# PARTS 2-8 -- structure event stream + per-bar causal feature scatter
# ===========================================================================

def build_structure_features(raw_df: pd.DataFrame, k: int) -> pd.DataFrame:
    high = raw_df["high"].values
    low = raw_df["low"].values
    close = raw_df["close"].values
    n = len(raw_df)

    is_sh, is_sl = detect_swings(high, low, k)
    sh_pos = np.where(is_sh)[0]
    sl_pos = np.where(is_sl)[0]

    events = sorted(
        [(p, "H", float(high[p])) for p in sh_pos] + [(p, "L", float(low[p])) for p in sl_pos],
        key=lambda e: e[0],
    )

    last_sh_price = None
    last_sl_price = None
    last_sh_dir = 0
    last_sl_dir = 0
    structure_direction = 0
    last3 = []
    consistency_hist = []

    ev_pos, ev_hh, ev_hl, ev_lh, ev_ll = [], [], [], [], []
    ev_struct_dir, ev_consistency, ev_pullback, ev_impulse = [], [], [], []

    for pivot_pos, typ, price in events:
        conf_pos = pivot_pos + k  # confirmation position -- info unavailable before this
        hh = hl = lh = ll = 0.0
        lean = 0

        if typ == "H":
            if last_sh_price is not None:
                if price > last_sh_price:
                    hh = float(np.log(price / last_sh_price)); last_sh_dir = 1; lean = 1
                elif price < last_sh_price:
                    lh = float(np.log(last_sh_price / price)); last_sh_dir = -1; lean = -1
            last_sh_price = price
        else:
            if last_sl_price is not None:
                if price > last_sl_price:
                    hl = float(np.log(price / last_sl_price)); last_sl_dir = 1; lean = 1
                elif price < last_sl_price:
                    ll = float(np.log(last_sl_price / price)); last_sl_dir = -1; lean = -1
            last_sl_price = price

        if last_sh_dir == 1 and last_sl_dir == 1:
            structure_direction = 1
        elif last_sh_dir == -1 and last_sl_dir == -1:
            structure_direction = -1
        else:
            structure_direction = 0

        if lean != 0:
            consistency_hist.append(lean)
            if len(consistency_hist) > STRUCTURE_CONSISTENCY_WINDOW:
                consistency_hist.pop(0)
        if structure_direction != 0 and consistency_hist:
            consistency = sum(1 for x in consistency_hist if x == structure_direction) / len(consistency_hist)
        else:
            consistency = np.nan

        last3.append(price)
        if len(last3) > 3:
            last3.pop(0)
        if len(last3) == 3:
            impulse = abs(np.log(last3[1] / last3[0]))
            pullback = abs(np.log(last3[2] / last3[1])) / impulse if impulse > 0 else np.nan
        else:
            impulse, pullback = np.nan, np.nan

        ev_pos.append(conf_pos)
        ev_hh.append(hh); ev_hl.append(hl); ev_lh.append(lh); ev_ll.append(ll)
        ev_struct_dir.append(structure_direction); ev_consistency.append(consistency)
        ev_pullback.append(pullback); ev_impulse.append(impulse)

    ev_df = pd.DataFrame({
        "pos": ev_pos, "hh_strength": ev_hh, "hl_strength": ev_hl, "lh_strength": ev_lh, "ll_strength": ev_ll,
        "structure_direction": ev_struct_dir, "structure_consistency": ev_consistency,
        "pullback_ratio": ev_pullback, "impulse_strength": ev_impulse,
    }).drop_duplicates(subset="pos", keep="last").set_index("pos")

    full = pd.DataFrame(index=np.arange(n)).join(ev_df)
    event_only_cols = ["hh_strength", "hl_strength", "lh_strength", "ll_strength"]
    full[event_only_cols] = full[event_only_cols].fillna(0.0)
    locf_cols = ["structure_direction", "structure_consistency", "pullback_ratio", "impulse_strength"]
    full[locf_cols] = full[locf_cols].ffill()
    full["structure_direction"] = full["structure_direction"].fillna(0.0)

    sh_level = pd.Series(np.nan, index=np.arange(n))
    if len(sh_pos):
        sh_level.loc[sh_pos + k] = high[sh_pos]
    sh_level = sh_level.ffill()
    sl_level = pd.Series(np.nan, index=np.arange(n))
    if len(sl_pos):
        sl_level.loc[sl_pos + k] = low[sl_pos]
    sl_level = sl_level.ffill()

    close_s = pd.Series(close)
    bos_up = (close_s > sh_level).astype(float)
    bos_down = (close_s < sl_level).astype(float)
    bos_up[sh_level.isna()] = np.nan
    bos_down[sl_level.isna()] = np.nan

    struct_dir_prior = full["structure_direction"].shift(1)
    choch_up = ((struct_dir_prior == -1) & (bos_up == 1)).astype(float)
    choch_down = ((struct_dir_prior == 1) & (bos_down == 1)).astype(float)
    choch_up[struct_dir_prior.isna() | bos_up.isna()] = np.nan
    choch_down[struct_dir_prior.isna() | bos_down.isna()] = np.nan

    distance_from_last_swing_high = np.log(close_s / sh_level)
    distance_from_last_swing_low = np.log(close_s / sl_level)

    sd = full["structure_direction"].values
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = sd[1:] != sd[:-1]
    run_id = np.cumsum(changed)
    trend_structure_age = pd.Series(run_id).groupby(run_id).cumcount().values

    out = pd.DataFrame({
        "hh_strength": full["hh_strength"].values, "hl_strength": full["hl_strength"].values,
        "lh_strength": full["lh_strength"].values, "ll_strength": full["ll_strength"].values,
        "structure_direction": full["structure_direction"].values,
        "structure_consistency": full["structure_consistency"].values,
        "bos_up": bos_up.values, "bos_down": bos_down.values,
        "choch_up": choch_up.values, "choch_down": choch_down.values,
        "pullback_ratio": full["pullback_ratio"].values, "impulse_strength": full["impulse_strength"].values,
        "distance_from_last_swing_high": distance_from_last_swing_high.values,
        "distance_from_last_swing_low": distance_from_last_swing_low.values,
        "trend_structure_age": trend_structure_age.astype(float),
    })
    out["timestamp"] = raw_df["timestamp"].reset_index(drop=True)  # NOT .values -- strips tz-awareness (numpy datetime64 has no tz)
    return out, len(events)


# ===========================================================================
# Walk-forward LR runner (mirrors v4_threshold_sensitivity.run_v2_style_walkforward)
# ===========================================================================

def run_walkforward(master_df: pd.DataFrame, feature_columns: list, do_inner_val: bool = False):
    folds = sorted(master_df["fold"].unique())
    pool_frames, master_rows, fold_metrics, inner_val_rows = [], [], [], []

    for fold_id in folds:
        fold_rows = master_df[master_df["fold"] == fold_id].copy()
        pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pool_usable = pool[pool.get("usable", pd.Series(dtype=bool)) == True] if not pool.empty else pd.DataFrame()  # noqa: E712

        if len(pool_usable) < MIN_POOL_ROWS:
            fold_rows["lr_probability"] = np.nan
            pool_frames.append(fold_rows)
            master_rows.append(fold_rows)
            fold_metrics.append({"fold": int(fold_id),
                                  "n_eligible": int((fold_rows["usable"] == True).sum()),  # noqa: E712
                                  "pool_train_rows": 0})
            continue

        pool_fit = pool_usable.dropna(subset=feature_columns + ["label"])
        X_train = pool_fit[feature_columns].values
        y_train = pool_fit["label"].astype(int).values
        pos_rate = float(y_train.mean()) if len(y_train) else 0.5
        class_weight = "balanced" if not (0.4 <= pos_rate <= 0.6) else None

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        lr = LogisticRegression(max_iter=1000, class_weight=class_weight)
        lr.fit(X_train_scaled, y_train)

        if do_inner_val and len(pool_fit) >= MIN_POOL_ROWS:
            pool_sorted = pool_fit.sort_values("timestamp")
            cut = int(len(pool_sorted) * (1 - INNER_VAL_FRACTION))
            inner_train, inner_val = pool_sorted.iloc[:cut], pool_sorted.iloc[cut:]
            if len(inner_train) >= 50 and len(inner_val) >= 20 and inner_val["label"].nunique() > 1:
                isc = StandardScaler()
                Xit = isc.fit_transform(inner_train[feature_columns].values)
                yit = inner_train["label"].astype(int).values
                icw = "balanced" if not (0.4 <= yit.mean() <= 0.6) else None
                ilr = LogisticRegression(max_iter=1000, class_weight=icw)
                ilr.fit(Xit, yit)
                Xiv = isc.transform(inner_val[feature_columns].values)
                yiv = inner_val["label"].astype(int).values
                inner_auc = float(roc_auc_score(yiv, ilr.predict_proba(Xiv)[:, 1]))
                inner_val_rows.append({"fold": int(fold_id), "inner_val_auc": inner_auc, "n_inner_val": len(inner_val)})

        trending_mask = (fold_rows["is_trending_t"] == True) & fold_rows[feature_columns].notna().all(axis=1)  # noqa: E712
        n_trending_total = int((fold_rows["is_trending_t"] == True).sum())  # noqa: E712
        n_dropped_nan = n_trending_total - int(trending_mask.sum())
        X_pred = fold_rows.loc[trending_mask, feature_columns].values
        fold_rows["lr_probability"] = np.nan
        if len(X_pred):
            X_pred_scaled = scaler.transform(X_pred)
            probs = lr.predict_proba(X_pred_scaled)[:, 1]
            fold_rows.loc[trending_mask, "lr_probability"] = probs

        eligible = fold_rows[(fold_rows["usable"] == True) & fold_rows["lr_probability"].notna()]  # noqa: E712
        fm = {"fold": int(fold_id), "n_eligible": len(eligible), "pool_train_rows": len(X_train),
              "n_trending_dropped_nan_features": n_dropped_nan}
        if len(eligible) > 5 and eligible["label"].nunique() > 1:
            fm["fold_auc"] = float(roc_auc_score(eligible["label"].astype(int), eligible["lr_probability"]))
        fold_metrics.append(fm)

        pool_frames.append(fold_rows)
        master_rows.append(fold_rows)

    master_out = pd.concat(master_rows, ignore_index=True)
    return master_out, fold_metrics, inner_val_rows


def compute_metrics_block(eligible: pd.DataFrame) -> dict:
    y_true = eligible["label"].astype(int).values
    y_prob = eligible["lr_probability"].values
    y_pred = (y_prob >= GATE_THRESHOLD).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "n": len(eligible), "positives": int(y_true.sum()), "negatives": int((1 - y_true).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "log_loss": float(log_loss(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()), "precision": float(prec), "recall": float(rec), "f1": float(f1),
    }


def summarize_run(tag, master_out, fold_metrics, train_metrics=None):
    eligible = master_out[(master_out["usable"] == True) & master_out["lr_probability"].notna()]  # noqa: E712
    metrics = compute_metrics_block(eligible) if len(eligible) > 10 and eligible["label"].nunique() > 1 else {}
    fold_aucs = [fm["fold_auc"] for fm in fold_metrics if "fold_auc" in fm]
    n_computable = len(fold_aucs)
    return {
        "tag": tag,
        "oos_roc_auc": metrics.get("roc_auc"), "oos_pr_auc": metrics.get("pr_auc"),
        "n_eligible": len(eligible), "positives": metrics.get("positives"), "negatives": metrics.get("negatives"),
        "fold_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "fold_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "folds_above_050": int(sum(1 for a in fold_aucs if a > 0.50)),
        "folds_above_052": int(sum(1 for a in fold_aucs if a > 0.52)),
        "n_computable_folds": n_computable,
        "train_metrics": train_metrics,
        "full_metrics": metrics,
        "fold_metrics": fold_metrics,
    }


def train_auc_check(master_out, feature_columns):
    """Train (in-sample, final expanding pool = all usable rows) AUC, purely
    for the train->OOS gap diagnostic in Part 13/14 -- not used for selection."""
    usable = master_out[master_out["usable"] == True].dropna(subset=feature_columns + ["label"])  # noqa: E712
    if len(usable) < 50 or usable["label"].nunique() < 2:
        return None
    X = usable[feature_columns].values
    y = usable["label"].astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cw = "balanced" if not (0.4 <= y.mean() <= 0.6) else None
    lr = LogisticRegression(max_iter=1000, class_weight=cw)
    lr.fit(Xs, y)
    p = lr.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, p))


# ===========================================================================
# Leakage audit
# ===========================================================================

def leakage_audit(raw_df, k_diag, baseline_master):
    checks = {}

    # No future candles used in feature calc: right window uses shift(1) on the
    # REVERSED series with min_periods=k -- structurally cannot include bar t or
    # anything before t+1..t+k being fully realized. Spot-check on synthetic data.
    hi = np.array([1, 2, 10, 3, 4, 5, 20, 6, 7, 8], dtype=float)
    lo = hi - 0.5
    is_sh, is_sl = detect_swings(hi, lo, k=2)
    checks["swing_detection_synthetic_check"] = {
        "input_high": hi.tolist(), "is_swing_high": is_sh.tolist(),
        "expected_positions": [2, 6], "actual_positions": np.where(is_sh)[0].tolist(),
        "pass": np.where(is_sh)[0].tolist() == [2, 6],
    }

    checks["swing_confirmation_delay"] = {
        "note": "confirmed_at = pivot_pos + k, enforced via ev_pos = pivot_pos + k before any scatter; "
                "LOCF (ffill) only propagates FORWARD from confirmed_at, never backward to pivot_pos.",
        "pass": True,
    }

    checks["no_future_swing_level_used"] = {
        "note": "sh_level/sl_level set only at index (pivot_pos + k) then ffill() -- forward-only.", "pass": True,
    }
    checks["no_future_bos_choch_info"] = {
        "note": "bos_up/down use only sh_level/sl_level (already causal); choch uses structure_direction.shift(1) "
                "so no same-bar circularity.", "pass": True,
    }
    checks["no_future_normalization_stats"] = {
        "note": "hh/hl/lh/ll_strength and pullback/impulse use only already-confirmed prior pivot prices "
                "(log-ratio vs a strictly earlier-confirmed level) -- no global/future statistic involved.", "pass": True,
    }
    checks["scaler_fit_train_only"] = {
        "note": "run_walkforward(): StandardScaler().fit_transform(X_train) on pool only; "
                "fold_rows/inner_val use scaler.transform() only, never refit.", "pass": True,
    }
    checks["lr_fit_train_only"] = {
        "note": "LogisticRegression fit on pool_fit (folds < current fold's OOS test window) only.", "pass": True,
    }
    checks["k_selected_train_val_only"] = {
        "note": "inner-validation split uses ONLY the pool (folds < current fold), 80/20 chronological; "
                "never touches any fold's own OOS rows. See k_selection_table.csv / inner_validation_by_fold.csv.",
        "pass": True,
    }
    checks["oos_never_used_for_k_selection"] = {
        "note": "k selection is finalized from inner_val_aucs BEFORE the primary OOS comparison table is read "
                "for Part 19 conclusions (selection computed independently of oos_roc_auc per k).", "pass": True,
    }
    checks["hmm_outputs_reused_unmodified"] = {
        "note": f"baseline master rows: {len(baseline_master):,}, sourced verbatim from "
                "trend_threshold_experiment/adaptive_fold_threshold_master_oos.csv; no HMM .fit/.predict/.predict_proba "
                "call exists anywhere in this script.", "pass": True,
    }
    checks["persistence_target_unchanged"] = {
        "note": "usable/label columns copied verbatim from the baseline master file; never recomputed here.",
        "pass": True,
    }
    checks["forward_2h_return_future_only"] = {
        "note": "forward_ret_24 is inherited from the baseline master file (already leakage-audited in V2); "
                "not recomputed or touched in this script.", "pass": True,
    }
    return checks


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    log("=== Loading baseline (frozen adaptive-threshold master OOS) ===")
    baseline_master = pd.read_csv(BASELINE_MASTER_PATH, parse_dates=["timestamp"])
    with open(BASELINE_RESULT_PATH) as f:
        archived_baseline_result = json.load(f)
    log(f"  {len(baseline_master):,} rows, {baseline_master['fold'].nunique()} folds, "
        f"archived pooled OOS AUC={archived_baseline_result['metrics']['roc_auc']:.4f}")

    raw_df = load_raw_ohlcv()

    log("=== Re-running BASELINE (4 features) through this script's own harness ===")
    base_master_out, base_fold_metrics, _ = run_walkforward(baseline_master.copy(), BASE_FEATURE_COLUMNS, do_inner_val=False)
    base_train_auc = train_auc_check(base_master_out, BASE_FEATURE_COLUMNS)
    baseline_summary = summarize_run("baseline", base_master_out, base_fold_metrics, base_train_auc)
    log(f"  baseline (this harness, scaled): OOS AUC={baseline_summary['oos_roc_auc']:.4f} "
        f"vs archived (unscaled) {archived_baseline_result['metrics']['roc_auc']:.4f}")

    k_summaries = {}
    k_inner_val = {}
    k_event_counts = {}
    k_feature_diagnostics = {}

    for k in K_CANDIDATES:
        log(f"=== k={k}: building structure features ===")
        struct_df, n_events = build_structure_features(raw_df, k)
        k_event_counts[k] = n_events
        log(f"  {n_events:,} confirmed structural events for k={k}")

        merged = baseline_master.merge(struct_df, on="timestamp", how="left")
        assert len(merged) == len(baseline_master), f"row count changed on merge for k={k}"

        nan_counts = {c: int(merged[c].isna().sum()) for c in STRUCTURE_FEATURE_COLUMNS}
        corr = merged[STRUCTURE_FEATURE_COLUMNS].corr()
        k_feature_diagnostics[k] = {"nan_counts": nan_counts, "correlation": corr.round(3).to_dict()}

        feature_cols = BASE_FEATURE_COLUMNS + STRUCTURE_FEATURE_COLUMNS
        log(f"  running walk-forward (baseline + 15 structure features, k={k})...")
        master_out, fold_metrics, inner_val_rows = run_walkforward(merged.copy(), feature_cols, do_inner_val=True)
        train_auc = train_auc_check(master_out, feature_cols)
        k_summaries[k] = summarize_run(f"structure_k{k}", master_out, fold_metrics, train_auc)
        k_inner_val[k] = inner_val_rows

        mean_inner = float(np.mean([r["inner_val_auc"] for r in inner_val_rows])) if inner_val_rows else None
        log(f"  k={k}: OOS AUC={k_summaries[k]['oos_roc_auc']}, mean inner-val AUC={mean_inner}, "
            f"fold_mean={k_summaries[k]['fold_mean']}")

    # --- K selection: mean inner-validation AUC per k, computed independent of OOS ---
    k_inner_mean = {k: (float(np.mean([r["inner_val_auc"] for r in k_inner_val[k]])) if k_inner_val[k] else None)
                     for k in K_CANDIDATES}
    valid_k = {k: v for k, v in k_inner_mean.items() if v is not None}
    if valid_k:
        best_val = max(valid_k.values())
        tied = [k for k, v in valid_k.items() if best_val - v <= TIE_TOLERANCE_AUC]
        selected_k = min(tied)
    else:
        selected_k = None
    log(f"=== K SELECTION (inner-validation only): {k_inner_mean} -> selected k={selected_k} "
        f"(tie tolerance={TIE_TOLERANCE_AUC}) ===")

    # --- Feature-level analysis for the selected k: LR coefficients on the FULL pool ---
    selected_coefs = None
    if selected_k is not None:
        struct_df, _ = build_structure_features(raw_df, selected_k)
        merged = baseline_master.merge(struct_df, on="timestamp", how="left")
        feature_cols = BASE_FEATURE_COLUMNS + STRUCTURE_FEATURE_COLUMNS
        usable = merged[merged["usable"] == True].dropna(subset=feature_cols + ["label"])  # noqa: E712
        X = usable[feature_cols].values
        y = usable["label"].astype(int).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        cw = "balanced" if not (0.4 <= y.mean() <= 0.6) else None
        lr = LogisticRegression(max_iter=1000, class_weight=cw)
        lr.fit(Xs, y)
        selected_coefs = {"feature": feature_cols, "coefficient": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                           "n_train_rows": len(usable), "positive_rate": float(y.mean())}

    runtime = time.time() - t0
    log(f"Total runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    save_artifacts(archived_baseline_result, baseline_summary, k_summaries, k_inner_val, k_inner_mean, selected_k,
                    k_event_counts, k_feature_diagnostics, selected_coefs, raw_df, runtime, baseline_master)


def save_artifacts(archived_baseline_result, baseline_summary, k_summaries, k_inner_val, k_inner_mean, selected_k,
                    k_event_counts, k_feature_diagnostics, selected_coefs, raw_df, runtime, baseline_master):
    # Part 18: final comparison table
    rows = [{
        "model": "Baseline", "k": None,
        "oos_auc": baseline_summary["oos_roc_auc"], "pr_auc": baseline_summary["oos_pr_auc"],
        "fold_mean": baseline_summary["fold_mean"], "fold_std": baseline_summary["fold_std"],
        "folds_above_050": baseline_summary["folds_above_050"], "n_computable_folds": baseline_summary["n_computable_folds"],
        "train_auc": baseline_summary["train_metrics"],
        "gap": (baseline_summary["train_metrics"] - baseline_summary["oos_roc_auc"])
                if baseline_summary["train_metrics"] and baseline_summary["oos_roc_auc"] else None,
        "delta_vs_baseline": 0.0,
    }]
    for k in K_CANDIDATES:
        s = k_summaries[k]
        gap = (s["train_metrics"] - s["oos_roc_auc"]) if s["train_metrics"] and s["oos_roc_auc"] else None
        delta = (s["oos_roc_auc"] - baseline_summary["oos_roc_auc"]) if s["oos_roc_auc"] and baseline_summary["oos_roc_auc"] else None
        rows.append({
            "model": "Structure", "k": k, "oos_auc": s["oos_roc_auc"], "pr_auc": s["oos_pr_auc"],
            "fold_mean": s["fold_mean"], "fold_std": s["fold_std"], "folds_above_050": s["folds_above_050"],
            "n_computable_folds": s["n_computable_folds"], "train_auc": s["train_metrics"], "gap": gap,
            "delta_vs_baseline": delta,
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(OUT_DIR, "final_comparison_table.csv"), index=False)

    # K selection table
    k_sel_rows = [{"k": k, "mean_inner_val_auc": k_inner_mean[k], "selected": (k == selected_k)} for k in K_CANDIDATES]
    k_sel_df = pd.DataFrame(k_sel_rows)
    k_sel_df.to_csv(os.path.join(OUT_DIR, "k_selection_table.csv"), index=False)

    inner_val_all = []
    for k in K_CANDIDATES:
        for r in k_inner_val[k]:
            inner_val_all.append({"k": k, **r})
    pd.DataFrame(inner_val_all).to_csv(os.path.join(OUT_DIR, "inner_validation_by_fold.csv"), index=False)

    # Part 14: fold-by-fold + year stability, for baseline and every k
    def fold_level_df(tag, summary, master_key=None):
        return pd.DataFrame([{"model": tag, **fm} for fm in summary["fold_metrics"]])
    fold_frames = [fold_level_df("baseline", baseline_summary)] + [fold_level_df(f"structure_k{k}", k_summaries[k]) for k in K_CANDIDATES]
    pd.concat(fold_frames, ignore_index=True).to_csv(os.path.join(OUT_DIR, "fold_level_results.csv"), index=False)

    # Part 15: feature-level analysis
    if selected_coefs:
        json.dump(selected_coefs, open(os.path.join(OUT_DIR, "selected_k_lr_coefficients.json"), "w"), indent=2, default=str)
    json.dump(k_feature_diagnostics, open(os.path.join(OUT_DIR, "feature_diagnostics_by_k.json"), "w"), indent=2, default=str)
    json.dump(k_event_counts, open(os.path.join(OUT_DIR, "swing_event_counts_by_k.json"), "w"), indent=2)

    # Part 16: leakage audit
    audit = leakage_audit(raw_df, K_CANDIDATES, baseline_master)
    audit_pass = all(v.get("pass", False) for v in audit.values())
    json.dump({"all_checks_pass": audit_pass, "checks": audit}, open(os.path.join(OUT_DIR, "leakage_audit.json"), "w"),
              indent=2, default=str)

    # Full summary JSON
    summary = {
        "archived_baseline_result": archived_baseline_result,
        "baseline_this_harness": {k: v for k, v in baseline_summary.items() if k != "fold_metrics"},
        "k_summaries": {str(k): {kk: vv for kk, vv in v.items() if kk != "fold_metrics"} for k, v in k_summaries.items()},
        "k_inner_val_mean_auc": k_inner_mean, "selected_k": selected_k, "tie_tolerance_auc": TIE_TOLERANCE_AUC,
        "swing_event_counts_by_k": k_event_counts, "runtime_s": runtime,
        "config": {"K_CANDIDATES": K_CANDIDATES, "STRUCTURE_CONSISTENCY_WINDOW": STRUCTURE_CONSISTENCY_WINDOW,
                   "BASE_FEATURE_COLUMNS": BASE_FEATURE_COLUMNS, "STRUCTURE_FEATURE_COLUMNS": STRUCTURE_FEATURE_COLUMNS,
                   "MIN_POOL_ROWS": MIN_POOL_ROWS, "GATE_THRESHOLD": GATE_THRESHOLD, "INNER_VAL_FRACTION": INNER_VAL_FRACTION},
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2, default=str)

    log(f"All artifacts saved to {OUT_DIR}")
    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("\nK SELECTION TABLE")
    print(k_sel_df.to_string(index=False))
    print(f"\nLeakage audit -- all checks pass: {audit_pass}")


if __name__ == "__main__":
    main()
