"""
Fits and pickles the production 15m-HMM4 regime classifier.

Selected over the 5m and 30m alternatives based on the completed 101-fold
expanding-window walk-forward validation (research/walk_forward/): lowest
transitions/day of any timeframe tested in every year 2018-2026, and the
only configuration with zero training-characterization failures (see
research/walk_forward/final_3way_walkforward_report.md).

Same hyperparameters as the original 5m production model: diagonal
covariance, n_iter=100, 5 restarts (seeds 42-46), best restart selected by
TRAINING log-likelihood only. Fits on ALL available history -- there is
nothing to hold out for a model about to be deployed forward.

Run: `python train_production_model.py` (from production/hmm_15m/).
Writes model/production_model_15m_hmm4.pkl.
"""

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from features_15m import build_15m_candles, build_feature_table, load_raw_5m, segment_lengths

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "model", "production_model_15m_hmm4.pkl")

N_STATES = 4
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _fit_one(train_values, lengths, seed):
    model = GaussianHMM(n_components=N_STATES, covariance_type=COVARIANCE_TYPE, n_iter=N_ITER, random_state=seed)
    t0 = time.time()
    try:
        model.fit(train_values, lengths=lengths)
        ll = model.score(train_values, lengths=lengths)
        return {"ll": ll, "model": model, "error": None, "runtime": time.time() - t0}
    except Exception as e:
        return {"ll": None, "model": None, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0}


def main():
    log("Loading raw 5m OHLCV and building the 15m feature table...")
    raw5m = load_raw_5m()
    feat_df, valid = build_feature_table(raw5m)
    valid_index = feat_df.index[valid]
    train_values_raw = feat_df.loc[valid_index].values
    lengths = segment_lengths(valid_index)
    assert sum(lengths) == len(train_values_raw)
    log(f"  {len(feat_df):,} candles, {len(valid_index):,} valid ({len(lengths)} contiguous segments)")

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_values_raw)

    log(f"Fitting HMM-{N_STATES}: {N_RESTARTS} restarts, n_iter={N_ITER}, covariance={COVARIANCE_TYPE}...")
    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    best_model, best_ll, best_seed, restart_log = None, float("-inf"), None, []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one, train_matrix, lengths, s): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            result = fut.result()
            restart_log.append({"seed": seed, "ll": result["ll"], "error": result["error"]})
            status = "FAILED" if result["error"] else f"ll={result['ll']:.1f}"
            log(f"  seed={seed}: {status} ({result['runtime']:.1f}s)")
            if not result["error"] and result["ll"] > best_ll:
                best_model, best_ll, best_seed = result["model"], result["ll"], seed

    if best_model is None:
        raise RuntimeError("All restarts failed -- no model to save.")
    log(f"Selected seed={best_seed}, train_ll={best_ll:.2f} ({best_ll/len(valid_index):.4f}/sample), "
        f"converged={best_model.monitor_.converged}")

    artifact = {
        "model": best_model, "scaler": scaler, "feature_columns": list(feat_df.columns),
        "n_states": N_STATES, "timeframe": "15m",
        "covariance_type": COVARIANCE_TYPE, "n_iter": N_ITER, "n_restarts": N_RESTARTS,
        "selected_seed": best_seed, "restart_log": restart_log,
        "train_start": str(valid_index.min()), "train_end": str(valid_index.max()),
        "n_train_valid": len(valid_index), "n_train_segments": len(lengths),
        "train_log_likelihood": best_ll, "train_log_likelihood_per_sample": best_ll / len(valid_index),
        "training_strategy": "expanding (all available history through train_end)",
        "validated_by": "research/walk_forward/final_3way_walkforward_report.md (101-fold OOS walk-forward)",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    log(f"Saved production model to {OUT_PATH}")


if __name__ == "__main__":
    main()
