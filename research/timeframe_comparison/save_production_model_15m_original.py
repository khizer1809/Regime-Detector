"""
save_production_model_15m.py -- fits and pickles the production 15m-HMM4
regime-detector model: a clean, dedicated production build (not a reused
research/screening byproduct), mirroring save_production_model.py's own
methodology and artifact schema exactly, but on genuine 15m candles built
from the raw 5m OHLCV file.

15m-HMM4 was chosen for production over the existing 5m-HMM4 baseline based
on this project's completed 3-way walk-forward validation
(experiments/final_3way_walkforward/): across 101 expanding-window OOS
folds spanning 2018-2026, 15m-HMM4 had the lowest transitions/day of any
timeframe tested in every single year, and was the only configuration with
zero training-characterization failures (5m-HMM4 lost all directional
states in 39/101 folds; 30m-HMM6 lost states in 20/101 folds and showed
chronic restart instability; 15m-HMM4 had neither problem).

Same hyperparameters as save_production_model.py (do not change without
re-validating): diag covariance, n_iter=100, 5 restarts (seeds 42-46), best
selected by TRAINING log-likelihood only, expanding/full-history strategy
(fit on ALL available history -- there is nothing to hold out for a model
about to be deployed forward onto genuinely new data).

Feature set: the exact 18 features validated in the fast timeframe
screening (research_archive/src/v22_fast_timeframe_screening.py) -- genuine
15m candles (clock-aligned resample of the raw 5m file), economic-time-
scaled returns/volatility/EMA-trend/volume/order-flow/distributional
features. Gap-aware masking uses the same timeframe-native validity logic
validated there (v22.build_validity_mask_native + segment_lengths_native),
not the original gap_aware.py's hardcoded-5-minute functions.

Run: `python save_production_model_15m.py` (from src/). Writes
Data/production_model_15m_hmm4.pkl.
"""

import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "research_archive", "src"))
import v22_fast_timeframe_screening as v22  # noqa: E402

OUT_PATH = os.path.join(_ROOT, "Data", "production_model_15m_hmm4.pkl")
N_STATES = 4
TIMEFRAME = "15m"

# Identical to save_production_model.py -- do not change without re-validating.
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _fit_one_segmented(train_values, lengths, n_states, seed):
    model = GaussianHMM(n_components=n_states, covariance_type=COVARIANCE_TYPE, n_iter=N_ITER, random_state=seed)
    t0 = time.time()
    try:
        model.fit(train_values, lengths=lengths)
        ll = model.score(train_values, lengths=lengths)
        return {"ll": ll, "model": model, "error": None, "runtime": time.time() - t0,
                "converged": bool(model.monitor_.converged), "n_iter_run": int(model.monitor_.iter)}
    except Exception as e:
        return {"ll": None, "model": None, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0}


def main():
    log("Loading raw 5m OHLCV and building genuine 15m candles + 18-feature table...")
    raw5m = v22.load_raw_5m()
    agg = v22.build_agg_candles(raw5m, TIMEFRAME)
    feat_df = v22.build_tf_features(agg, TIMEFRAME)
    feat_cols = list(feat_df.columns)
    max_lookback = max(v22.EMA_SLOW_BARS, max(v22.RETURN_HORIZON_BARS[TIMEFRAME].values()))
    valid_flag = v22.build_validity_mask_native(agg, feat_df.index, max_lookback)
    valid_mask = valid_flag & feat_df.notna().all(axis=1)
    valid_index = feat_df.index[valid_mask]
    train_values_raw = feat_df.loc[valid_index].values
    bar_interval = __import__("pandas").Timedelta(minutes=v22.TF_MINUTES[TIMEFRAME])
    lengths = v22.segment_lengths_native(valid_index, bar_interval)
    assert sum(lengths) == len(train_values_raw)
    log(f"  {len(feat_df):,} candles, {len(valid_index):,} valid ({len(lengths)} contiguous segments), "
        f"{len(feat_cols)} features")

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_values_raw)

    log(f"Fitting HMM-{N_STATES}: {N_RESTARTS} restarts (up to {N_WORKERS} concurrent), "
        f"n_iter={N_ITER}, covariance={COVARIANCE_TYPE}...")
    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    best_model, best_ll, best_seed, best_restart = None, float("-inf"), None, None
    restart_log = []
    t_all0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one_segmented, train_matrix, lengths, N_STATES, s): (r, s)
                   for r, s in enumerate(seeds)}
        for fut in as_completed(futures):
            r, seed = futures[fut]
            result = fut.result()
            failed = result["error"] is not None
            restart_log.append({"restart": r, "seed": seed, "ll": result["ll"], "failed": failed,
                                 "error": result["error"], "runtime_s": result["runtime"]})
            status = "FAILED" if failed else f"ll={result['ll']:.1f}"
            log(f"  restart {r} (seed={seed}): {status}  ({result['runtime']:.1f}s, "
                f"elapsed={time.time()-t_all0:.1f}s)")
            if not failed and result["ll"] > best_ll:
                best_model, best_ll, best_seed, best_restart = result["model"], result["ll"], seed, r

    if best_model is None:
        raise RuntimeError("All restarts failed to fit -- no model to save.")

    log(f"Selected restart {best_restart} (seed={best_seed}), train_ll={best_ll:.2f} "
        f"({best_ll/len(valid_index):.4f} per sample), converged={best_model.monitor_.converged}, "
        f"n_iter_run={best_model.monitor_.iter}")

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_columns": feat_cols,
        "n_states": N_STATES,
        "timeframe": TIMEFRAME,
        "covariance_type": COVARIANCE_TYPE,
        "n_iter": N_ITER,
        "n_restarts": N_RESTARTS,
        "selected_restart": best_restart,
        "selected_seed": best_seed,
        "train_start": str(valid_index.min()),
        "train_end": str(valid_index.max()),
        "n_train_valid": len(valid_index),
        "n_train_segments": len(lengths),
        "train_log_likelihood": best_ll,
        "train_log_likelihood_per_sample": best_ll / len(valid_index),
        "restart_log": restart_log,
        "training_strategy": "expanding (all available history through train_end)",
        "source": "genuine 15m candles resampled from raw 5m OHLCV (research_archive/src/v22_fast_timeframe_screening.py feature design)",
        "validation": "experiments/final_3way_walkforward/ (101-fold OOS walk-forward, most stable/reliable of 4 configurations tested)",
        "notes": (
            "Fit with timeframe-native gap-aware masking (no dropna): candles whose feature "
            "lookback touches a zero-volume/gap bar, or that are NaN for any other reason, are "
            "excluded from fitting; the model is fit segment-aware (hmmlearn lengths=) so no "
            "transition is ever modeled across an excluded gap. To score new bars: build genuine "
            "15m candles + the same 18 feature columns via v22_fast_timeframe_screening.py's "
            "build_agg_candles/build_tf_features, apply this artifact's `scaler`.transform(), then "
            "`model`.predict()/.score()."
        ),
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log(f"Saved production model to:\n  {OUT_PATH}")
    log(f"File size: {os.path.getsize(OUT_PATH)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
