"""
save_production_model.py -- fits and pickles a production regime-detector
model using the expanding-window strategy validated in the rolling-vs-
expanding experiment (see Data/rolling_vs_expanding_final_report.md),
trained on ALL available valid history (not a held-out test slice -- there
is nothing to hold out for a model that is about to be deployed forward
onto genuinely new, unseen data).

Self-contained production script: reads Data/features_out_masked.csv
(features.build_feature_matrix(raw, dropna=False) + gap_aware.py's `valid`
column -- see gap_aware.py for how that file is built), applies gap-aware
masking (no dropna, no deleted raw rows -- excludes only bars whose feature
lookback touches a verified zero-volume/gap bar, or that are NaN for any
other reason such as the warm-up period at the very start of history),
fits segment-aware (hmmlearn's `lengths=`, so no transition is ever modeled
across an excluded gap), 5 restarts, best restart picked by TRAINING
log-likelihood (never test data -- there is no test split here).

HMM_CONFIG below is the single source of truth for these hyperparameters
going forward; it matches what walk_forward.py/regime_persistence_test.py
used throughout the rolling-vs-expanding experiment so this model is
consistent with those validated results.

State count is a command-line argument (defaults to 4): both HMM-4 and
HMM-6 are legitimate choices depending on what downstream use needs --
HMM-6 has the better out-of-sample likelihood fit (wins every fold in the
experiment), HMM-4 has longer/more persistent regimes (fewer, broader
states) and is ~2x cheaper to retrain. See the final report for the full
comparison.

Run: `python save_production_model.py [N_STATES]` (from src/). Writes to
Data/production_model_hmm<N_STATES>.pkl -- does not overwrite a different
N_STATES' file, so switching which one is "the" production model is a
matter of which file the deployment code points at, not a destructive swap.
"""

import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gap_aware import valid_values_and_lengths
from scaling import fit_transform_fold

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKED_PATH = os.path.join(_ROOT, "Data", "features_out_masked.csv")

N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT_PATH = os.path.join(_ROOT, "Data", f"production_model_hmm{N_STATES}.pkl")

# Matches walk_forward.py / regime_persistence_test.py throughout the
# rolling-vs-expanding experiment -- do not change without re-validating,
# since these are what the experiment's recommendation was based on.
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)


def load_masked_features():
    """Loads Data/features_out_masked.csv and adds a combined validity flag:
    a bar is usable only if it clears BOTH gap_aware.py's zero-volume-gap
    mask AND has no NaN in the model's own feature columns (the latter
    catches the history warm-up period, which the gap mask alone doesn't)."""
    print(f"Loading full gap-aware feature matrix from:\n  {MASKED_PATH}", flush=True)
    t0 = time.time()
    df = pd.read_csv(MASKED_PATH, index_col=0, parse_dates=True)
    model_cols = [c for c in df.columns if c != "valid"]
    print(f"  loaded {len(df):,} rows x {len(model_cols)} model columns in {time.time()-t0:.1f}s", flush=True)

    gap_invalid = ~df["valid"]
    nan_invalid = ~df[model_cols].notna().all(axis=1)
    n = len(df)
    n_valid = int((~(gap_invalid | nan_invalid)).sum())
    print(
        f"  valid-for-model: {n_valid:,}/{n:,} ({n_valid/n:.1%})  "
        f"excluded: gap-only={int((gap_invalid & ~nan_invalid).sum()):,} "
        f"nan-only(warm-up/other)={int((nan_invalid & ~gap_invalid).sum()):,} "
        f"both={int((gap_invalid & nan_invalid).sum()):,}",
        flush=True,
    )
    return df, model_cols


def _fit_one_segmented(train_values, lengths, n_states, seed):
    """One restart. hmmlearn `lengths=` keeps the EM/Viterbi recursions from
    ever crossing a segment boundary (a genuine data gap)."""
    model = GaussianHMM(n_components=n_states, covariance_type=COVARIANCE_TYPE,
                         n_iter=N_ITER, random_state=seed)
    t0 = time.time()
    try:
        model.fit(train_values, lengths=lengths)
        ll = model.score(train_values, lengths=lengths)
        return {"ll": ll, "model": model, "error": None, "runtime": time.time() - t0}
    except Exception as e:
        return {"ll": None, "model": None, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0}


def _fit_one_segmented_with_retry(train_values, lengths, n_states, seed, max_retries=2):
    """Retries only on transient-looking failures (MemoryError/OSError from
    shared-thread-pool resource contention) with the SAME seed -- a
    deterministic numerical failure (bad covariance, etc.) would just fail
    identically again, so those are not retried."""
    attempt = 0
    while True:
        result = _fit_one_segmented(train_values, lengths, n_states, seed)
        if result["error"] is None:
            return result, attempt
        transient = any(k in result["error"] for k in ("MemoryError", "OSError"))
        attempt += 1
        if not transient or attempt > max_retries:
            return result, attempt - 1


def main():
    print("Loading full gap-aware feature matrix (training on ALL available history)...", flush=True)
    df, model_cols = load_masked_features()

    train_vals, train_lengths, valid_index = valid_values_and_lengths(df, model_cols)
    print(f"Training window: {df.index.min()} -> {df.index.max()}", flush=True)
    print(f"Valid rows: {len(train_vals):,} / {len(df):,}  ({len(train_lengths)} contiguous segments)", flush=True)

    train_matrix, _, scaler = fit_transform_fold(train_vals, train_vals.iloc[:min(10, len(train_vals))])
    train_matrix = train_matrix.values

    print(f"\nFitting HMM-{N_STATES}: {N_RESTARTS} restarts (up to {N_WORKERS} concurrent), "
          f"n_iter={N_ITER}, covariance={COVARIANCE_TYPE}...", flush=True)
    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    best_model, best_ll, best_seed, best_restart = None, float("-inf"), None, None
    restart_log = []
    t_all0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one_segmented_with_retry, train_matrix, train_lengths, N_STATES, s): (r, s)
                   for r, s in enumerate(seeds)}
        for fut in as_completed(futures):
            r, seed = futures[fut]
            result, n_retries = fut.result()
            failed = result["error"] is not None
            restart_log.append({"restart": r, "seed": seed, "ll": result["ll"], "failed": failed,
                                 "error": result["error"], "runtime_s": result["runtime"]})
            status = "FAILED" if failed else f"ll={result['ll']:.1f}"
            print(f"  restart {r} (seed={seed}): {status}  ({result['runtime']:.1f}s, "
                  f"elapsed={time.time()-t_all0:.1f}s)", flush=True)
            if not failed and result["ll"] > best_ll:
                best_model, best_ll, best_seed, best_restart = result["model"], result["ll"], seed, r

    if best_model is None:
        raise RuntimeError("All restarts failed to fit -- no model to save.")

    print(f"\nSelected restart {best_restart} (seed={best_seed}), train_ll={best_ll:.2f} "
          f"({best_ll/len(train_vals):.4f} per sample), converged={best_model.monitor_.converged}, "
          f"n_iter_run={best_model.monitor_.iter}", flush=True)

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_columns": model_cols,
        "n_states": N_STATES,
        "covariance_type": COVARIANCE_TYPE,
        "n_iter": N_ITER,
        "n_restarts": N_RESTARTS,
        "selected_restart": best_restart,
        "selected_seed": best_seed,
        "train_start": str(df.index.min()),
        "train_end": str(df.index.max()),
        "n_train_valid": len(train_vals),
        "n_train_segments": len(train_lengths),
        "train_log_likelihood": best_ll,
        "train_log_likelihood_per_sample": best_ll / len(train_vals),
        "restart_log": restart_log,
        "training_strategy": "expanding (all available history through train_end)",
        "source_features_file": "Data/features_out_masked.csv",
        "notes": (
            "Fit with gap-aware masking (no dropna): rows whose feature lookback touches a "
            "verified zero-volume/gap bar, or that are NaN for any other reason, are excluded "
            "from fitting; the model is fit segment-aware (hmmlearn lengths=) so no transition "
            "is modeled across an excluded gap. To score new bars: build the same 18 feature "
            "columns (session columns excluded) with features.build_feature_matrix(dropna=False), "
            "apply this artifact's `scaler`.transform(), then `model`.predict()/.score()."
        ),
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved production model to:\n  {OUT_PATH}", flush=True)
    print(f"File size: {os.path.getsize(OUT_PATH)/1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
