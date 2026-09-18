"""
v13_ofi_stationarity_test.py -- EXPERIMENTAL, NOT PRODUCTION.

Cheap diagnostic (no persistence-pipeline rerun) for the question: does
`ofi_raw`'s cross-era scale drift (std ranges ~8.9 in 2017 to ~54.3 in 2022
to ~30.7 in 2025 -- confirmed on real data, a smaller-but-real analog of a
non-stationarity bug found and fixed in a separate project for a raw
dollar-scale slope feature) meaningfully corrupt the production HMM's state
assignment?

Method: refit TWO HMMs, byte-identical hyperparameters/seeds/data to
save_production_model.py (diag covariance, n_iter=100, 5 restarts seeds
42-46, best restart by train log-likelihood), differing in EXACTLY ONE
thing:
    Model A (control)   -- all 18 production feature columns (incl. ofi_raw)
    Model B (treatment)  -- the same 18 minus ofi_raw (17 columns)

Then compare, on the shared valid-bar index:
  1. Per-bar state-LABEL agreement (Uptrend/Downtrend/Ranging x Vol-level --
     NOT raw state index, which is known from this project's own prior work
     to be arbitrary/incomparable across independent fits) + Adjusted Rand
     Index (permutation-invariant, so it doesn't matter that state 0 in A
     might correspond to state 2 in B).
  2. Per-state mean comparison on the 17 SHARED features (did dropping
     ofi_raw shift what the remaining features say a state "means"?).
  3. Correlation of confidence / stay_prob / log1p_duration / margin between
     A and B (the exact 4 features the persistence LR consumes) -- this is
     the most direct proxy for "would the downstream persistence gate see a
     materially different world."

Does NOT touch Data/production_model_hmm4.pkl, Data/persistence_gate.pkl,
or any other production file. Writes only to
research_archive/Data/ofi_stationarity_test/.

Run: `python v13_ofi_stationarity_test.py` (from research_archive/src/).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_archive/
_PROJECT_ROOT = os.path.dirname(_ROOT)
_SRC = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC)

from gap_aware import valid_values_and_lengths
from scaling import fit_transform_fold
from infer_regime import characterize_state
import persistence_common
MASKED_PATH = os.path.join(_PROJECT_ROOT, "Data", "features_out_masked.csv")
OUT_DIR = os.path.join(_ROOT, "Data", "ofi_stationarity_test")

N_STATES = 4
COVARIANCE_TYPE = "diag"
N_ITER = 100
RANDOM_STATE = 42
N_RESTARTS = 5
N_WORKERS = min(6, os.cpu_count() or 1)
DROP_COL = "ofi_raw"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_masked_features():
    df = pd.read_csv(MASKED_PATH, index_col=0, parse_dates=True)
    model_cols = [c for c in df.columns if c != "valid"]
    return df, model_cols


def _fit_one(train_values, lengths, n_states, seed):
    model = GaussianHMM(n_components=n_states, covariance_type=COVARIANCE_TYPE,
                         n_iter=N_ITER, random_state=seed)
    t0 = time.time()
    try:
        model.fit(train_values, lengths=lengths)
        ll = model.score(train_values, lengths=lengths)
        return {"ll": ll, "model": model, "error": None, "runtime": time.time() - t0}
    except Exception as e:
        return {"ll": None, "model": None, "error": f"{type(e).__name__}: {e}", "runtime": time.time() - t0}


def fit_hmm(label, df, model_cols):
    log(f"=== Fitting {label} ({len(model_cols)} features: {model_cols}) ===")
    train_vals, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    log(f"  valid rows: {len(train_vals):,}  segments: {len(lengths)}")

    train_matrix, _, scaler = fit_transform_fold(train_vals, train_vals.iloc[:min(10, len(train_vals))])
    train_matrix = train_matrix.values

    seeds = [RANDOM_STATE + r for r in range(N_RESTARTS)]
    best_model, best_ll, best_seed = None, float("-inf"), None
    t_all0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_fit_one, train_matrix, lengths, N_STATES, s): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            result = fut.result()
            status = "FAILED" if result["error"] else f"ll={result['ll']:.1f}"
            log(f"  restart seed={seed}: {status} ({result['runtime']:.1f}s)")
            if result["error"] is None and result["ll"] > best_ll:
                best_model, best_ll, best_seed = result["model"], result["ll"], seed

    log(f"  selected seed={best_seed}  train_ll={best_ll:.2f}  "
        f"per_sample={best_ll/len(train_vals):.4f}  elapsed={time.time()-t_all0:.1f}s")

    return {
        "model": best_model, "scaler": scaler, "feature_columns": model_cols,
        "valid_index": valid_index, "lengths": lengths, "train_matrix": train_matrix,
        "train_ll": best_ll, "train_ll_per_sample": best_ll / len(train_vals),
        "n_params": best_model.n_features * N_STATES * 2 + N_STATES * N_STATES,  # diag means+vars + transmat, rough
    }


def state_labels_for(fit_result):
    model = fit_result["model"]
    means_df = pd.DataFrame(model.means_, columns=fit_result["feature_columns"])
    labels = [characterize_state(means_df.iloc[s]) for s in range(N_STATES)]
    states = model.predict(fit_result["train_matrix"], lengths=fit_result["lengths"])
    posteriors = model.predict_proba(fit_result["train_matrix"], lengths=fit_result["lengths"])
    per_bar_label = np.array([labels[s] for s in states])
    return states, posteriors, labels, per_bar_label, means_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df, model_cols = load_masked_features()
    log(f"Loaded {len(df):,} rows, {len(model_cols)} production feature columns")
    assert DROP_COL in model_cols, f"{DROP_COL} not found in production feature columns: {model_cols}"
    model_cols_b = [c for c in model_cols if c != DROP_COL]

    fit_a = fit_hmm("Model A (control, all 18 features incl. ofi_raw)", df, model_cols)
    fit_b = fit_hmm(f"Model B (treatment, 17 features, {DROP_COL} dropped)", df, model_cols_b)

    # --- Align on the SHARED valid-bar index (dropping ofi_raw could in
    # principle change which rows are NaN-valid; check and restrict to the
    # intersection so every comparison below is on identical bars). ---
    idx_a, idx_b = set(fit_a["valid_index"]), set(fit_b["valid_index"])
    only_a, only_b = idx_a - idx_b, idx_b - idx_a
    log(f"\nValid-index reconciliation: A={len(idx_a):,} B={len(idx_b):,} "
        f"only_in_A={len(only_a):,} only_in_B={len(only_b):,}")

    states_a, post_a, labels_a, per_bar_label_a, means_a = state_labels_for(fit_a)
    states_b, post_b, labels_b, per_bar_label_b, means_b = state_labels_for(fit_b)

    df_a = pd.DataFrame({"state": states_a, "label": per_bar_label_a}, index=fit_a["valid_index"])
    df_b = pd.DataFrame({"state": states_b, "label": per_bar_label_b}, index=fit_b["valid_index"])
    common_index = df_a.index.intersection(df_b.index)
    log(f"Common index for comparison: {len(common_index):,} bars "
        f"({len(common_index)/max(len(idx_a),len(idx_b)):.2%} of the larger valid set)")

    da = df_a.loc[common_index]
    db = df_b.loc[common_index]

    # 1. Label agreement + ARI (permutation-invariant on raw state index)
    label_agree = float((da["label"].values == db["label"].values).mean())
    ari = float(adjusted_rand_score(da["state"].values, db["state"].values))
    confusion = pd.crosstab(da["label"], db["label"], rownames=["A_label"], colnames=["B_label"])
    log(f"\nLabel agreement (A vs B, same bar): {label_agree:.4%}")
    log(f"Adjusted Rand Index (raw state index, permutation-invariant): {ari:.4f}")
    log(f"\nConfusion matrix (state LABELS, not raw indices):\n{confusion.to_string()}")

    # 2. Per-state means on the 17 shared features
    shared_cols = [c for c in model_cols if c != DROP_COL]
    means_a_shared = means_a[shared_cols].copy()
    means_a_shared["label"] = labels_a
    means_b_shared = means_b[shared_cols].copy()
    means_b_shared["label"] = labels_b
    log(f"\nModel A per-state means (shared 17 features) + label:\n{means_a_shared.to_string()}")
    log(f"\nModel B per-state means (shared 17 features) + label:\n{means_b_shared.to_string()}")
    label_set_a, label_set_b = sorted(set(labels_a)), sorted(set(labels_b))
    log(f"\nDistinct state labels -- A: {label_set_a}  B: {label_set_b}  "
        f"(state-space collision check: {len(label_set_a)}/{N_STATES} and {len(label_set_b)}/{N_STATES} distinct)")

    # 3. Persistence-feature correlation on the common index
    def persistence_feats(fit_result, states, posteriors, lengths, valid_index):
        conf = persistence_common.compute_confidence(posteriors, states)
        stay = persistence_common.compute_stay_prob(states, fit_result["model"].transmat_)
        dur = persistence_common.compute_running_duration(states, lengths)
        marg = persistence_common.compute_margin(posteriors)
        return pd.DataFrame({
            "confidence": conf, "stay_prob": stay,
            "log1p_duration": persistence_common.duration_feature(dur), "margin": marg,
        }, index=valid_index)

    pf_a = persistence_feats(fit_a, states_a, post_a, fit_a["lengths"], fit_a["valid_index"]).loc[common_index]
    pf_b = persistence_feats(fit_b, states_b, post_b, fit_b["lengths"], fit_b["valid_index"]).loc[common_index]

    corrs = {}
    for col in ["confidence", "stay_prob", "log1p_duration", "margin"]:
        r = float(np.corrcoef(pf_a[col].values, pf_b[col].values)[0, 1])
        corrs[col] = r
        log(f"  corr(A.{col}, B.{col}) = {r:.4f}")

    # LL comparison caveat: different feature dimensionality (18 vs 17) makes
    # raw log-likelihood not directly comparable -- report BIC per sample
    # instead (penalizes A's extra dimension's parameters).
    bic_a = -2 * fit_a["train_ll"] + fit_a["n_params"] * np.log(len(fit_a["valid_index"]))
    bic_b = -2 * fit_b["train_ll"] + fit_b["n_params"] * np.log(len(fit_b["valid_index"]))

    runtime = time.time() - t0
    summary = {
        "drop_col": DROP_COL,
        "n_valid_a": len(idx_a), "n_valid_b": len(idx_b), "n_common": len(common_index),
        "train_ll_a": fit_a["train_ll"], "train_ll_b": fit_b["train_ll"],
        "train_ll_per_sample_a": fit_a["train_ll_per_sample"], "train_ll_per_sample_b": fit_b["train_ll_per_sample"],
        "bic_a": float(bic_a), "bic_b": float(bic_b),
        "label_agreement": label_agree, "adjusted_rand_index": ari,
        "distinct_labels_a": label_set_a, "distinct_labels_b": label_set_b,
        "persistence_feature_correlations": corrs,
        "runtime_s": runtime,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    confusion.to_csv(os.path.join(OUT_DIR, "state_label_confusion.csv"))
    means_a_shared.to_csv(os.path.join(OUT_DIR, "means_a_shared_features.csv"))
    means_b_shared.to_csv(os.path.join(OUT_DIR, "means_b_shared_features.csv"))

    log(f"\nDone in {runtime:.1f}s. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
