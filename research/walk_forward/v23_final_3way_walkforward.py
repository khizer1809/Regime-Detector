"""
v23_final_3way_walkforward.py -- EXPERIMENTAL, NOT PRODUCTION.

FINAL 3-WAY WALK-FORWARD VALIDATION.

Tests whether the fast-screening advantages found for 15m-HMM4 and 30m-HMM6
(research_archive/src/v22_fast_timeframe_screening.py, full-history
development fits) survive genuine out-of-sample walk-forward evaluation
against the existing 5m-HMM4 baseline. No new models, no new features, no
hyperparameter search, no D/MTF, no production modification.

===========================================================================
WALK-FORWARD METHODOLOGY -- REUSED, NOT INVENTED
===========================================================================
Fold generation is the EXPANDING-window schedule from
research_archive/src/v2_common.py's generate_folds() (DATA_START=2017-09-01,
WARMUP_MONTHS=6, TEST_MONTHS=1, STEP_MONTHS=1) -- the only actual prior
walk-forward implementation found in this codebase, and the one whose
EXPANDING variant this project's own Data/rolling_vs_expanding_final_report.md
validated and save_production_model.py adopted for production (expanding
beat rolling on OOS log-likelihood in 91-96% of folds in that experiment).
This task's own illustrative example ("Fold1: Train=Jan-Jun ... Fold2:
Train=Feb-Jul") describes a ROLLING 6-month window, which conflicts with
that established/validated/production choice. Per the task's own higher-
priority instruction ("derive exact calendar boundaries from the EXISTING
implementation, do not invent new fold boundaries"), the actual existing
implementation (EXPANDING) is used, and the conflict with the illustrative
example is disclosed here and in the final report rather than silently
picked either way.

EMBARGO is applied as a genuine 4-hour timedelta (`test_start - 4h`), not a
bar count, so it is correct and identical in real time across 5m/15m/30m
(48 5m-bars = 16 15m-bars = 8 30m-bars = 4h, all equivalent).

Per-fold procedure (identical shape for all 3 models, only the underlying
timeframe/feature table/state count differs):
  1. Slice each timeframe's own precomputed, gap-flagged feature table by
     [train_start, train_end) and [test_start, test_end).
  2. Restrict to valid (gap-aware, non-NaN) rows within each slice
     independently; recompute per-slice contiguous segments (native bar
     interval) so hmmlearn's `lengths=` never models a transition across a
     real gap OR across the train/test boundary itself.
  3. Fit StandardScaler on TRAIN ONLY; transform train and test with that
     same fitted scaler (scaling.fit_transform_fold's exact procedure).
  4. Fit GaussianHMM on TRAIN ONLY, 5 restarts (seeds 42-46), select best by
     TRAINING log-likelihood only (identical architecture/hyperparameters
     to save_production_model.py: diag covariance, n_iter=100).
  5. Characterize states (Uptrend/Downtrend/Ranging x Vol level) from
     TRAIN-fitted means only.
  6. Decode the ENTIRE test fold in one pass (`model.predict`/`predict_proba`
     with `lengths=test_lengths`) -- the model is already fully finalized
     (fit + restart selection + labeling all used train data only) before
     this happens, so decoding the whole fold at once introduces no leakage
     (nothing about the fit was influenced by test data); this is simpler
     and matches how OOS log-likelihood is computed everywhere else in this
     project (e.g. the rolling-vs-expanding experiment's own
     `expanding_test_ll_per_sample`), rather than a bar-by-bar causal-window
     simulation (that mechanism, in v2_common.causal_decode_at, exists for a
     different purpose -- simulating what a live single-bar inference call
     would have seen -- not needed for a fold-level regime-structure
     comparison).
  7. Test states map onto TRAIN-derived regime labels automatically (same
     fitted model object decodes both; state 0 in the fold's decode is
     state 0's train-characterized label, no separate remapping needed).

Run: `python v23_final_3way_walkforward.py` (from research_archive/src/).
"""

import json
import os

# CPU-oversubscription fix, must run before numpy/scipy/sklearn/hmmlearn are
# imported (BLAS reads these at import time): this machine's OpenBLAS
# defaults to 6 threads PER PROCESS, and fit_hmm() below runs 5 restarts
# CONCURRENTLY in one process via ThreadPoolExecutor -- unpatched, that's up
# to 5x6=30 BLAS threads contending for 6 physical cores. An initial A/B
# test of this seemed to show it made things WORSE (245.5s vs 156.8s
# unrestricted) -- that comparison turned out to be invalid: a stray
# duplicate process from an earlier run had not actually been killed and
# was silently contending for CPU during that test. Re-measured with a
# verified-clean process list (Get-CimInstance showed only this process):
# 169.4s unrestricted vs 153.6s pinned, a real ~9% speedup. Purely an
# execution-speed change -- does not alter the HMM architecture, restart
# count, fold boundaries, or any other measured quantity.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pickle
import platform
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import sklearn
import hmmlearn
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, _RESEARCH_ROOT + "/src")

import persistence_common  # noqa: E402
from save_production_model import load_masked_features  # noqa: E402
from infer_regime import RETURN_COLS as RETURN_COLS_5M, VOL_COLS as VOL_COLS_5M, characterize_state as characterize_state_5m  # noqa: E402

import v22_fast_timeframe_screening as v22  # noqa: E402 -- reused verbatim for 15m/30m features

EXPERIMENT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "final_3way_walkforward")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(EXPERIMENT_DIR, "final_3way_walkforward_report.md")

# ---------------------------------------------------------------------------
# Fold schedule -- reused from v2_common.py (see module docstring)
# ---------------------------------------------------------------------------
DATA_START = pd.Timestamp("2017-09-01", tz="UTC")
WARMUP_MONTHS = 6   # == task's TRAIN_MONTHS
TEST_MONTHS = 1
STEP_MONTHS = 1
EMBARGO_HOURS = 4   # == 48 5m-bars == 16 15m-bars == 8 30m-bars, applied as real time

# ---------------------------------------------------------------------------
# HMM hyperparameters -- identical across all 3 models, copied verbatim from
# save_production_model.py. Do not change.
# ---------------------------------------------------------------------------
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)

MODELS = {
    "5m_hmm4": {"timeframe": "5m", "n_states": 4, "bar_minutes": 5.0},
    "15m_hmm4": {"timeframe": "15m", "n_states": 4, "bar_minutes": 15.0},
    "30m_hmm6": {"timeframe": "30m", "n_states": 6, "bar_minutes": 30.0},
}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def generate_folds(last_valid_ts: pd.Timestamp) -> list:
    """Exact reproduction of v2_common.generate_folds()'s algorithm and
    parameters, generalized to a real-time (not bar-count) embargo so it is
    identical in wall-clock terms across every timeframe."""
    folds = []
    first_test_start = DATA_START + pd.DateOffset(months=WARMUP_MONTHS)
    k = 0
    while True:
        test_start = first_test_start + pd.DateOffset(months=STEP_MONTHS * k)
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        if test_start > last_valid_ts:
            break
        test_end = min(test_end, last_valid_ts + pd.Timedelta(minutes=5))
        train_end = test_start - pd.Timedelta(hours=EMBARGO_HOURS)
        folds.append({"fold_id": k, "train_start": DATA_START, "train_end": train_end,
                       "test_start": test_start, "test_end": test_end})
        k += 1
    return folds


# ===========================================================================
# Per-timeframe full-history feature table (built once, sliced per fold)
# ===========================================================================

def build_feature_table(timeframe: str):
    """Returns (feat_df, valid_series, bar_interval, feat_cols) for one
    timeframe. 5m reuses the existing production feature file verbatim
    (Data/features_out_masked.csv via save_production_model.load_masked_features,
    the SAME 18 production columns, the SAME gap-aware `valid` flag already
    on disk). 15m/30m reuse v22_fast_timeframe_screening.py's own functions
    verbatim (build_agg_candles/build_tf_features/build_validity_mask_native)
    -- the exact feature definitions validated in the fast screening, not
    rebuilt or altered here."""
    if timeframe == "5m":
        df, model_cols = load_masked_features()
        feat_df = df[model_cols]
        valid = df["valid"] & df[model_cols].notna().all(axis=1)
        bar_interval = pd.Timedelta(minutes=5)
        return feat_df, valid, bar_interval, model_cols
    else:
        raw5m = _RAW5M_CACHE["df"]
        agg = v22.build_agg_candles(raw5m, timeframe)
        feat_df = v22.build_tf_features(agg, timeframe)
        max_lookback = max(v22.EMA_SLOW_BARS, max(v22.RETURN_HORIZON_BARS[timeframe].values()))
        valid_flag = v22.build_validity_mask_native(agg, feat_df.index, max_lookback)
        valid = valid_flag & feat_df.notna().all(axis=1)
        bar_interval = pd.Timedelta(minutes=v22.TF_MINUTES[timeframe])
        return feat_df, valid, bar_interval, list(feat_df.columns)


_RAW5M_CACHE = {}


# ===========================================================================
# HMM fit (identical hyperparameters/procedure across all 3 models)
# ===========================================================================

def _fit_one(train_values, lengths, n_states, seed):
    model = GaussianHMM(n_components=n_states, covariance_type=COVARIANCE_TYPE, n_iter=N_ITER, random_state=seed)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            model.fit(train_values, lengths=lengths)
            ll = model.score(train_values, lengths=lengths)
            warn_msgs = [str(w.message) for w in wlist]
            return {"ll": ll, "model": model, "error": None, "runtime": time.time() - t0,
                    "converged": bool(model.monitor_.converged), "n_iter_run": int(model.monitor_.iter),
                    "warnings": warn_msgs}
        except Exception as e:
            return {"ll": None, "model": None, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0,
                    "converged": None, "n_iter_run": None, "warnings": [str(w.message) for w in wlist]}


def fit_hmm(train_values, lengths, n_states):
    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    best = {"ll": float("-inf"), "model": None, "seed": None}
    restart_log = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one, train_values, lengths, n_states, s): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            r = fut.result()
            restart_log.append({"seed": seed, "ll": r["ll"], "error": r["error"], "runtime_s": r["runtime"],
                                 "converged": r.get("converged"), "n_iter_run": r.get("n_iter_run"),
                                 "warnings": r.get("warnings")})
            if r["error"] is None and r["ll"] > best["ll"]:
                best = {"ll": r["ll"], "model": r["model"], "seed": seed}
    lls = [r["ll"] for r in restart_log if r["ll"] is not None]
    spread = (max(lls) - min(lls)) if len(lls) > 1 else 0.0
    n_converged = sum(1 for r in restart_log if r.get("converged"))
    n_failed = sum(1 for r in restart_log if r["error"] is not None)
    return best["model"], best["ll"], best["seed"], restart_log, spread, n_converged, n_failed


# ===========================================================================
# Failure/warning checks (Section 22)
# ===========================================================================

def check_failures(model, states, confidence, n_states, regime_labels, restart_log, n_converged, n_failed):
    flags = []
    occ = pd.Series(states).value_counts(normalize=True)
    max_occ = float(occ.max()) if len(occ) else 1.0
    if max_occ > 0.95:
        flags.append(f"one_state_domination(max_occ={max_occ:.3f})")
    if len(set(states)) < n_states:
        flags.append(f"missing_states(observed={len(set(states))}/{n_states})")
    if n_failed > 0:
        flags.append(f"restart_failures({n_failed}/{N_RESTARTS})")
    if n_converged < N_RESTARTS:
        flags.append(f"non_converged_restarts({N_RESTARTS - n_converged}/{N_RESTARTS})")
    covars = model.covars_
    min_var = float(np.min([covars[s][i, i] for s in range(n_states) for i in range(covars.shape[1])]))
    if min_var < 1e-8:
        flags.append(f"covariance_collapse(min_var={min_var:.2e})")
    if not any("Uptrend" in r for r in regime_labels):
        flags.append("no_uptrend_state")
    if not any("Downtrend" in r for r in regime_labels):
        flags.append("no_downtrend_state")
    if not any("Ranging" in r for r in regime_labels):
        flags.append("no_ranging_state")
    if np.isnan(confidence).any() or np.isinf(confidence).any():
        flags.append("nan_or_inf_confidence")
    return flags


# ===========================================================================
# Per-fold, per-model run
# ===========================================================================

def run_fold(model_key, cfg, fold, feat_df, valid, bar_interval, feat_cols):
    tf = cfg["timeframe"]
    n_states = cfg["n_states"]

    train_mask = valid & (feat_df.index >= fold["train_start"]) & (feat_df.index < fold["train_end"])
    test_mask = valid & (feat_df.index >= fold["test_start"]) & (feat_df.index < fold["test_end"])
    train_idx = feat_df.index[train_mask]
    test_idx = feat_df.index[test_mask]

    if len(train_idx) < 500 or len(test_idx) < 5:
        return {"fold_id": fold["fold_id"], "status": "skipped_insufficient_data",
                "n_train": len(train_idx), "n_test": len(test_idx)}

    train_values_raw = feat_df.loc[train_idx].values
    test_values_raw = feat_df.loc[test_idx].values
    train_lengths = v22.segment_lengths_native(train_idx, bar_interval)
    test_lengths = v22.segment_lengths_native(test_idx, bar_interval)
    assert sum(train_lengths) == len(train_idx)
    assert sum(test_lengths) == len(test_idx)
    assert np.isfinite(train_values_raw).all(), f"{model_key} fold {fold['fold_id']}: non-finite in TRAIN features"
    assert np.isfinite(test_values_raw).all(), f"{model_key} fold {fold['fold_id']}: non-finite in TEST features"
    assert test_idx.min() >= fold["test_start"] and test_idx.max() < fold["test_end"]
    assert train_idx.max() < fold["train_end"]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values_raw)   # TRAIN ONLY
    test_scaled = scaler.transform(test_values_raw)         # SAME scaler, never refit

    t0 = time.time()
    model, best_ll, best_seed, restart_log, spread, n_converged, n_failed = fit_hmm(train_scaled, train_lengths, n_states)
    fit_time = time.time() - t0
    if model is None:
        return {"fold_id": fold["fold_id"], "status": "all_restarts_failed", "restart_log": restart_log}

    means_df = pd.DataFrame(model.means_, columns=feat_cols)
    if tf == "5m":
        regime_labels = [characterize_state_5m(means_df.iloc[s]) for s in range(n_states)]
        trend_scores = np.array([float(means_df.iloc[s][[c for c in RETURN_COLS_5M if c in means_df.columns]].mean())
                                  for s in range(n_states)])
        vol_scores = np.array([float(means_df.iloc[s][[c for c in VOL_COLS_5M if c in means_df.columns]].mean())
                                for s in range(n_states)])
    else:
        labels_scores = [v22.characterize_state_tf(means_df.iloc[s], tf) for s in range(n_states)]
        regime_labels = [x[0] for x in labels_scores]
        trend_scores = np.array([x[1] for x in labels_scores])
        vol_scores = np.array([x[2] for x in labels_scores])

    t1 = time.time()
    test_states = model.predict(test_scaled, lengths=test_lengths)
    test_post = model.predict_proba(test_scaled, lengths=test_lengths)
    decode_time = time.time() - t1
    test_confidence = test_post[np.arange(len(test_states)), test_states]
    test_ll = model.score(test_scaled, lengths=test_lengths)  # OOS LL -- reporting only, never used to select anything

    seg_id_test = persistence_common.segment_ids_from_lengths(test_lengths)
    segments = v22.build_segments(test_states, seg_id_test, test_idx, cfg["bar_minutes"])
    segments["regime"] = segments["state"].map(dict(enumerate(regime_labels)))
    transitions = v22.build_transitions(segments)

    n_test = len(test_idx)
    total_days = (test_idx.max() - test_idx.min()).total_seconds() / 86400 if len(test_idx) > 1 else \
        cfg["bar_minutes"] / 1440
    total_days = max(total_days, cfg["bar_minutes"] / 1440)

    dur = segments["duration_minutes"]
    n_transitions = len(transitions)
    occ = pd.Series(test_states).value_counts(normalize=True).sort_index()

    flags = check_failures(model, test_states, test_confidence, n_states, regime_labels, restart_log, n_converged, n_failed)

    fold_row = {
        "fold_id": fold["fold_id"], "model": model_key, "timeframe": tf, "n_states": n_states,
        "train_start": str(fold["train_start"]), "train_end": str(fold["train_end"]),
        "test_start": str(fold["test_start"]), "test_end": str(fold["test_end"]),
        "test_year": fold["test_start"].year,
        "n_train": len(train_idx), "n_test": n_test,
        "n_transitions": n_transitions, "n_segments": len(segments),
        "transitions_per_1k_bars": n_transitions / n_test * 1000 if n_test else None,
        "transitions_per_day": n_transitions / total_days if total_days > 0 else None,
        "transitions_per_hour": n_transitions / (total_days * 24) if total_days > 0 else None,
        "median_duration_minutes": float(dur.median()) if len(dur) else None,
        "mean_duration_minutes": float(dur.mean()) if len(dur) else None,
        "min_duration_minutes": float(dur.min()) if len(dur) else None,
        "max_duration_minutes": float(dur.max()) if len(dur) else None,
        "pct_gt30m": float((dur > 30).mean() * 100) if len(dur) else None,
        "pct_gt1h": float((dur > 60).mean() * 100) if len(dur) else None,
        "pct_gt2h": float((dur > 120).mean() * 100) if len(dur) else None,
        "pct_gt4h": float((dur > 240).mean() * 100) if len(dur) else None,
        "pct_gt8h": float((dur > 480).mean() * 100) if len(dur) else None,
        "pct_1bar": float((segments["number_of_bars"] == 1).mean() * 100) if len(segments) else None,
        "n_distinct_states_observed": int(pd.Series(test_states).nunique()),
        "n_distinct_labels_observed": len(set(np.array(regime_labels)[sorted(pd.Series(test_states).unique())])),
        "n_distinct_labels_trained": len(set(regime_labels)),
        "max_state_occupancy_pct": float(occ.max() * 100),
        "state_occupancy": {int(k): float(v * 100) for k, v in occ.items()},
        "regime_labels_trained": {int(s): r for s, r in enumerate(regime_labels)},
        "train_ll": best_ll, "train_ll_per_obs": best_ll / len(train_idx),
        "test_ll": test_ll, "test_ll_per_obs": test_ll / n_test,
        "best_seed": best_seed, "restart_spread": spread,
        "restart_spread_normalized": spread / abs(best_ll) if best_ll else None,
        "n_restarts_converged": n_converged, "n_restarts_failed": n_failed,
        "mean_stay_prob": float(np.mean(np.diag(model.transmat_))),
        "median_stay_prob": float(np.median(np.diag(model.transmat_))),
        "trend_score_range": float(trend_scores.max() - trend_scores.min()),
        "vol_score_range": float(vol_scores.max() - vol_scores.min()),
        "fit_time_s": fit_time, "decode_time_s": decode_time,
        "failure_flags": flags, "status": "ok",
    }

    regime_rows = pd.DataFrame({
        "fold_id": fold["fold_id"], "timestamp": test_idx, "state": test_states,
        "regime": [regime_labels[s] for s in test_states], "confidence": test_confidence,
    })

    # future-return coherence (descriptive only -- Section 15/16)
    close_col = "close" if "close" in feat_df.columns else None
    future_ret_df = None
    if tf != "5m":
        agg = _AGG_CACHE[tf]
        close_test = agg["close"].reindex(test_idx)
        log_close = np.log(close_test.values)
        fwd_by_h = {}
        for h_name, h_bars in v22.RETURN_HORIZON_BARS[tf].items():
            j = np.arange(n_test) + h_bars
            fwd_valid = (j < n_test) & (seg_id_test[np.minimum(j, n_test - 1)] == seg_id_test)
            fr = np.full(n_test, np.nan)
            fr[fwd_valid] = log_close[j[fwd_valid]] - log_close[np.arange(n_test)[fwd_valid]]
            fwd_by_h[h_name] = fr
        future_ret_df = v22.future_return_stats(test_states, fwd_by_h, n_states, regime_labels)
        future_ret_df["fold_id"] = fold["fold_id"]

    return {"fold_row": fold_row, "regime_rows": regime_rows, "restart_log": restart_log,
            "future_ret_df": future_ret_df, "status": "ok"}


_AGG_CACHE = {}


# ===========================================================================
# main
# ===========================================================================

def main():
    t_start = time.time()
    log("Loading raw 5m OHLCV (shared source for 15m/30m candle construction)...")
    _RAW5M_CACHE["df"] = v22.load_raw_5m()

    log("Building full-history feature tables (once each; sliced per fold below)...")
    tables = {}
    last_valid_ts_by_tf = {}
    for tf in ["5m", "15m", "30m"]:
        feat_df, valid, bar_interval, feat_cols = build_feature_table(tf)
        tables[tf] = (feat_df, valid, bar_interval, feat_cols)
        last_valid_ts_by_tf[tf] = feat_df.index[valid].max()
        log(f"  {tf}: {len(feat_df):,} rows, {int(valid.sum()):,} valid, last_valid={last_valid_ts_by_tf[tf]}")
        if tf != "5m":
            _AGG_CACHE[tf] = v22.build_agg_candles(_RAW5M_CACHE["df"], tf)

    last_valid_ts = min(last_valid_ts_by_tf.values())
    folds = generate_folds(last_valid_ts)
    log(f"Generated {len(folds)} folds (expanding, WARMUP={WARMUP_MONTHS}mo, TEST={TEST_MONTHS}mo, "
        f"STEP={STEP_MONTHS}mo, EMBARGO={EMBARGO_HOURS}h). First test={folds[0]['test_start'].date()}, "
        f"last test={folds[-1]['test_start'].date()}")

    # --- benchmark: time the smallest and a mid-size fold per model, then estimate total ---
    log("Benchmarking 1 small fold + 1 mid fold per model to estimate total runtime...")
    bench_fold_ids = sorted({0, len(folds) // 2})
    bench_results = {}
    for model_key, cfg in MODELS.items():
        feat_df, valid, bar_interval, feat_cols = tables[cfg["timeframe"]]
        times = []
        for fid in bench_fold_ids:
            t0 = time.time()
            run_fold(model_key, cfg, folds[fid], feat_df, valid, bar_interval, feat_cols)
            dt = time.time() - t0
            times.append(dt)
            log(f"  {model_key} fold {fid} (n_train scale check): {dt:.1f}s")
        bench_results[model_key] = times

    # crude per-model estimate: mean of the two benchmarked folds x total fold count
    est_total = 0.0
    for model_key, times in bench_results.items():
        per_fold_est = float(np.mean(times))
        model_est = per_fold_est * len(folds)
        est_total += model_est
        log(f"  {model_key}: ~{per_fold_est:.1f}s/fold x {len(folds)} folds ~= {model_est/60:.1f} min")
    log(f"ESTIMATED total runtime (3 models x {len(folds)} folds): ~{est_total/60:.1f} min "
        f"(~{est_total/3600:.2f} h). Proceeding automatically now.")

    all_fold_rows = {k: [] for k in MODELS}
    all_regime_rows = {k: [] for k in MODELS}
    all_future_ret = {k: [] for k in MODELS}
    all_restart_logs = {k: [] for k in MODELS}
    skipped = {k: [] for k in MODELS}

    checkpoint_path = os.path.join(EXPERIMENT_DIR, "checkpoint.json")

    jobs = [(model_key, fold) for model_key in MODELS for fold in folds]
    pbar = tqdm(jobs, desc="3-way walk-forward")
    for model_key, fold in pbar:
        pbar.set_postfix_str(f"{model_key} fold {fold['fold_id']}")
        cfg = MODELS[model_key]
        feat_df, valid, bar_interval, feat_cols = tables[cfg["timeframe"]]
        try:
            result = run_fold(model_key, cfg, fold, feat_df, valid, bar_interval, feat_cols)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"  {model_key} fold {fold['fold_id']} FAILED: {e}\n{tb}")
            skipped[model_key].append({"fold_id": fold["fold_id"], "error": str(e)})
            continue

        if result.get("status") != "ok":
            skipped[model_key].append(result)
            continue

        all_fold_rows[model_key].append(result["fold_row"])
        all_regime_rows[model_key].append(result["regime_rows"])
        if result.get("future_ret_df") is not None:
            all_future_ret[model_key].append(result["future_ret_df"])
        all_restart_logs[model_key].extend([{**r, "model": model_key, "fold_id": fold["fold_id"]}
                                             for r in result["restart_log"]])

        # checkpoint after every fold
        pd.DataFrame(all_fold_rows[model_key]).to_csv(
            os.path.join(EXPERIMENT_DIR, f"{model_key}_fold_results.csv"), index=False)
        json.dump({"skipped": skipped, "n_completed": {k: len(v) for k, v in all_fold_rows.items()}},
                   open(checkpoint_path, "w"), indent=2, default=str)

    log("Writing OOS regime parquets...")
    for model_key in MODELS:
        if all_regime_rows[model_key]:
            pd.concat(all_regime_rows[model_key], ignore_index=True).to_parquet(
                os.path.join(EXPERIMENT_DIR, f"{model_key}_oos_regimes.parquet"), index=False)
        if all_future_ret[model_key]:
            pd.concat(all_future_ret[model_key], ignore_index=True).to_csv(
                os.path.join(EXPERIMENT_DIR, f"{model_key}_future_returns.csv"), index=False)
        pd.DataFrame(all_restart_logs[model_key]).to_csv(
            os.path.join(EXPERIMENT_DIR, f"{model_key}_restart_log.csv"), index=False)

    # yearly summary
    yearly_rows = []
    for model_key in MODELS:
        df = pd.DataFrame(all_fold_rows[model_key])
        if len(df) == 0:
            continue
        for year, grp in df.groupby("test_year"):
            yearly_rows.append({
                "model": model_key, "year": int(year), "n_folds": len(grp),
                "n_oos_obs": int(grp["n_test"].sum()),
                "mean_transitions_per_day": float(grp["transitions_per_day"].mean()),
                "median_duration_minutes": float(grp["median_duration_minutes"].median()),
                "mean_duration_minutes": float(grp["mean_duration_minutes"].mean()),
                "mean_state_occupancy_max_pct": float(grp["max_state_occupancy_pct"].mean()),
                "mean_test_ll_per_obs": float(grp["test_ll_per_obs"].mean()),
                "mean_n_distinct_labels_observed": float(grp["n_distinct_labels_observed"].mean()),
            })
    pd.DataFrame(yearly_rows).to_csv(os.path.join(EXPERIMENT_DIR, "yearly_summary.csv"), index=False)

    # model comparison tables (Section 19) -- separate measurement tables, no composite score
    comp_rows = []
    for model_key in MODELS:
        df = pd.DataFrame(all_fold_rows[model_key])
        if len(df) == 0:
            continue
        comp_rows.append({
            "model": model_key, "n_folds": len(df), "oos_obs_total": int(df["n_test"].sum()),
            "transitions_per_day_mean": float(df["transitions_per_day"].mean()),
            "transitions_per_day_median": float(df["transitions_per_day"].median()),
            "transitions_per_day_std": float(df["transitions_per_day"].std()),
            "median_duration_minutes_mean": float(df["median_duration_minutes"].mean()),
            "mean_duration_minutes_mean": float(df["mean_duration_minutes"].mean()),
            "pct_gt1h_mean": float(df["pct_gt1h"].mean()), "pct_gt2h_mean": float(df["pct_gt2h"].mean()),
            "pct_gt4h_mean": float(df["pct_gt4h"].mean()),
            "n_states": MODELS[model_key]["n_states"],
            "distinct_labels_mean": float(df["n_distinct_labels_observed"].mean()),
            "distinct_labels_trained_mode": int(df["n_distinct_labels_trained"].mode().iloc[0]),
            "test_ll_per_obs_mean": float(df["test_ll_per_obs"].mean()),
            "restart_spread_mean": float(df["restart_spread"].mean()),
            "restart_spread_normalized_mean": float(df["restart_spread_normalized"].mean()),
            "trend_score_range_mean": float(df["trend_score_range"].mean()),
            "vol_score_range_mean": float(df["vol_score_range"].mean()),
            "mean_stay_prob_mean": float(df["mean_stay_prob"].mean()),
            "n_folds_with_failure_flags": int((df["failure_flags"].apply(len) > 0).sum()),
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(EXPERIMENT_DIR, "model_comparison.csv"), index=False)

    runtime = time.time() - t_start
    config_out = {
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "hmmlearn_version": hmmlearn.__version__, "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "models": MODELS, "data_start": str(DATA_START), "warmup_months": WARMUP_MONTHS,
        "test_months": TEST_MONTHS, "step_months": STEP_MONTHS, "embargo_hours": EMBARGO_HOURS,
        "covariance_type": COVARIANCE_TYPE, "n_iter": N_ITER, "random_state_base": RANDOM_STATE,
        "n_restarts": N_RESTARTS, "n_workers": N_WORKERS, "n_folds": len(folds),
        "fold_dates": [{"fold_id": f["fold_id"], "train_start": str(f["train_start"]),
                         "train_end": str(f["train_end"]), "test_start": str(f["test_start"]),
                         "test_end": str(f["test_end"])} for f in folds],
        "runtime_s": runtime,
    }
    json.dump(config_out, open(os.path.join(EXPERIMENT_DIR, "experiment_config.json"), "w"), indent=2, default=str)

    log(f"Total runtime: {runtime:.1f}s ({runtime/3600:.2f} h)")
    for model_key in MODELS:
        log(f"  {model_key}: {len(all_fold_rows[model_key])}/{len(folds)} folds completed, "
            f"{len(skipped[model_key])} skipped/failed")
    log("Done. EXIT_MARKER_V23_DONE")


if __name__ == "__main__":
    main()
