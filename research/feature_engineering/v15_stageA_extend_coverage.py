"""
v15_stageA_extend_coverage.py -- extends the 11-fold pilot (fold_000,
fold_010, ..., fold_100 -- fit by v15_stageA_expanded_hmm.py under
FOLD_SUBSET_STEP=10) to FULL 101-month OOS coverage, WITHOUT any further
expensive HMM refitting.

REFIT-CADENCE METHODOLOGY (disclosed explicitly, not hidden): each of the
90 not-yet-decoded months is causally decoded using whichever of the 11
already-fit HMMs is the MOST RECENTLY AVAILABLE one as of that month (the
largest-fold_id anchor whose OWN train_end is <= this month's test_start).
Still strictly causal (that HMM never saw data past its own train_end,
always before the month it decodes) -- a refit-CADENCE deviation from the
original "retrain every month" design (refits only every ~10 months
instead), not a leakage shortcut.

DECODE-SPEED METHOD (validated, see v15_validate_fast_decode.py -- 0/300
mismatches against the original trusted per-row windowed method on real
data): instead of recomputing forward-backward/Viterbi from scratch in an
overlapping ~500-bar window for every single test row (the original
O(rows x 500) design), this computes ONE continuous forward pass per anchor
group (covering every month that anchor is responsible for), and reads the
causal posterior/Viterbi state off that single pass at every position --
mathematically identical at each position (both depend only on data up to
that position, never after -- see v15_fast_causal_decode.py's docstring for
why this is NOT the same as the unsafe shortcut of calling
predict()/predict_proba() once and reading every row, which WOULD be
non-causal at intermediate rows).

Run: `python v15_stageA_extend_coverage.py` (from research_archive/src/).
Writes fold_XXX/{causal_output.csv,metadata.json,DONE} for the 90 skipped
folds into the SAME expanded_hmm_cache/ directory the pilot used, so
Stage B can read all 101 fold directories uniformly.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v2_common  # noqa: E402
from gap_aware import segment_lengths, valid_values_and_lengths  # noqa: E402
from v15_fast_causal_decode import forward_only_pass, causal_reads_from_pass  # noqa: E402
import persistence_common  # noqa: E402

FEATURES_PATH = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "features_expanded_33.csv")
CACHE_DIR = os.path.join(_RESEARCH_ROOT, "Data", "hmm_lr_abc_experiment", "expanded_hmm_cache")
LOG_PATH = os.path.join(CACHE_DIR, "stage_a_extend_log.jsonl")


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_global_valid_data():
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    model_cols = [c for c in df.columns if c != "valid"]
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    seg_id, _ = v2_common.segment_ids_for_index(valid_index)
    return valid_df, valid_index, seg_id, model_cols


def process_anchor_group(anchor_fold_id, group_folds, valid_df, valid_index, seg_id, model_cols):
    """One forward pass covers every fold assigned to this anchor -- avoids
    recomputing the same (growing) prefix once per month."""
    pending = [f for f in group_folds if not os.path.exists(
        os.path.join(CACHE_DIR, f"fold_{f['fold_id']:03d}", "DONE"))]
    if not pending:
        return {"done": 0, "skipped": len(group_folds), "empty": 0}

    anchor_dir = os.path.join(CACHE_DIR, f"fold_{anchor_fold_id:03d}")
    with open(os.path.join(anchor_dir, "hmm.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(anchor_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    means_df = pd.DataFrame(model.means_, columns=model_cols)
    trend_score_per_state = persistence_common.state_trend_scores(means_df)
    trend_sign_per_state = np.sign(trend_score_per_state)
    is_trending_per_state = np.abs(trend_score_per_state) > persistence_common.TREND_Z_THRESHOLD

    fold_test_positions = {}
    max_pos = 0
    for f in pending:
        mask = np.array((valid_index >= f["test_start"]) & (valid_index < f["test_end"]))
        pos = np.where(mask)[0]
        fold_test_positions[f["fold_id"]] = pos
        if len(pos):
            max_pos = max(max_pos, int(pos.max()))

    counts = {"done": 0, "skipped": len(group_folds) - len(pending), "empty": 0}
    non_empty = [f for f in pending if len(fold_test_positions[f["fold_id"]]) > 0]
    for f in pending:
        if len(fold_test_positions[f["fold_id"]]) == 0:
            fold_dir = os.path.join(CACHE_DIR, f"fold_{f['fold_id']:03d}")
            os.makedirs(fold_dir, exist_ok=True)
            json.dump({**{k: str(v) for k, v in f.items()}, "n_test": 0, "anchor_fold": anchor_fold_id},
                       open(os.path.join(fold_dir, "metadata.json"), "w"), indent=2)
            open(os.path.join(fold_dir, "DONE"), "w").close()
            counts["empty"] += 1
    if not non_empty:
        return counts

    t0 = time.time()
    decode_upper_pos = max_pos + 1
    log(f"anchor {anchor_fold_id:03d}: computing one forward pass over {decode_upper_pos:,} rows "
        f"for {len(non_empty)} month(s) [{[f['fold_id'] for f in non_empty]}]...")
    scaled_prefix = scaler.transform(valid_df.values[:decode_upper_pos])
    forward_ret_full = valid_df[v2_common.FORWARD_RET_COL].values

    log_alpha, log_delta = forward_only_pass(model, scaled_prefix, seg_id[:decode_upper_pos])
    viterbi_states, posteriors, confidence, margin = causal_reads_from_pass(log_alpha, log_delta)
    duration_full = persistence_common.compute_running_duration(
        viterbi_states, segment_lengths(valid_index[:decode_upper_pos]))
    stay_prob_full = model.transmat_[viterbi_states, viterbi_states]
    pass_time = time.time() - t0
    log(f"anchor {anchor_fold_id:03d}: forward pass done in {pass_time:.1f}s, extracting per-month outputs...")

    for f in non_empty:
        fold_id = f["fold_id"]
        test_positions = fold_test_positions[fold_id]
        fold_dir = os.path.join(CACHE_DIR, f"fold_{fold_id:03d}")
        os.makedirs(fold_dir, exist_ok=True)

        rows = []
        for p in test_positions:
            j = p + v2_common.LABEL_HORIZON_BARS
            forward_valid = (j < len(seg_id)) and (seg_id[min(j, len(seg_id) - 1)] == seg_id[p])
            forward_ret = float(forward_ret_full[j]) if forward_valid else np.nan
            state = int(viterbi_states[p])
            post = posteriors[p]
            rows.append({
                "timestamp": valid_index[p], "fold": fold_id, "hmm_state": state,
                "posterior_state_0": post[0] if len(post) > 0 else np.nan,
                "posterior_state_1": post[1] if len(post) > 1 else np.nan,
                "posterior_state_2": post[2] if len(post) > 2 else np.nan,
                "posterior_state_3": post[3] if len(post) > 3 else np.nan,
                "confidence": float(confidence[p]), "margin": float(margin[p]),
                "duration": int(duration_full[p]), "duration_left_censored": None,
                "stay_prob": float(stay_prob_full[p]),
                "trend_score": float(trend_score_per_state[state]), "trend_sign": float(trend_sign_per_state[state]),
                "is_trending": bool(is_trending_per_state[state]), "segment_id": int(seg_id[p]),
                "forward_ret_24": forward_ret, "forward_valid": forward_valid,
            })
        pd.DataFrame(rows).to_csv(os.path.join(fold_dir, "causal_output.csv"), index=False)

        metadata = {
            "fold_id": fold_id, "train_start": str(f["train_start"]), "train_end": str(f["train_end"]),
            "test_start": str(f["test_start"]), "test_end": str(f["test_end"]),
            "anchor_fold": anchor_fold_id, "anchor_note": "decoded using anchor fold's HMM, not a fresh fit",
            "n_test": len(test_positions), "feature_columns": model_cols, "context_bars": v2_common.CONTEXT_BARS,
            "decode_method": "fast_single_pass_forward_filter, grouped-by-anchor (validated 0/300 mismatches vs windowed)",
            "code_version": "v15_stageA_extend_coverage.py:3",
        }
        json.dump(metadata, open(os.path.join(fold_dir, "metadata.json"), "w"), indent=2, default=str)
        open(os.path.join(fold_dir, "DONE"), "w").close()
        counts["done"] += 1

    log(f"anchor {anchor_fold_id:03d}: {len(non_empty)} month(s) extracted in "
        f"{time.time()-t0:.1f}s total (pass={pass_time:.1f}s)")
    return counts


def main():
    log("=== Extending pilot coverage to all 101 months (fast single-pass decode, grouped by anchor) ===")
    valid_df, valid_index, seg_id, model_cols = load_global_valid_data()
    last_ts = valid_index.max()
    all_folds = v2_common.generate_folds(last_ts)

    anchor_ids = sorted(int(d.split("_")[1]) for d in os.listdir(CACHE_DIR)
                         if d.startswith("fold_") and os.path.exists(os.path.join(CACHE_DIR, d, "hmm.pkl")))
    log(f"Anchor (already-fit) folds: {anchor_ids}")

    def most_recent_anchor(fold_id):
        candidates = [a for a in anchor_ids if a <= fold_id]
        return max(candidates) if candidates else min(anchor_ids)

    groups = {}
    for fold in all_folds:
        anchor = most_recent_anchor(fold["fold_id"])
        groups.setdefault(anchor, []).append(fold)

    t0 = time.time()
    totals = {"done": 0, "skipped": 0, "empty": 0}
    for anchor_fold_id in sorted(groups):
        counts = process_anchor_group(anchor_fold_id, groups[anchor_fold_id], valid_df, valid_index, seg_id, model_cols)
        for k in totals:
            totals[k] += counts.get(k, 0)

    log(f"=== Extension complete: {totals} in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
