"""
v21_outlier_audit.py -- EXPERIMENTAL, DIAGNOSTIC ONLY.

Exact outlier timestamp + OHLCV audit of the existing, already-completed
A-HMM4 model. Read-only: reuses results/model_A_HMM4.pkl (already-fit model
+ scaler) and re-decodes (cheap, no refit) -- does not retrain the HMM,
refit the scaler, change features, winsorize, remove outliers, smooth
states, or touch the MTF B/C/D experiment in any way.

Z-SCORES: reuses the ALREADY-FIT StandardScaler from the A-HMM4 artifact --
scaler.transform(X) IS the z-score by construction (mean 0, std 1 on the
population the scaler was fit on, which for A-HMM4 is the full valid
dataset). This is "the actual feature distributions already used by the
project's diagnostic methodology", not a new ad-hoc z-score definition.

OUTLIER EVENT GROUPING RULE (documented, not invented ad hoc): consecutive
VALID bars (positionally adjacent in the causal, gap-aware valid_index --
never crossing a genuine data-gap segment boundary) that each independently
have >4sigma on at least one feature are merged into one event. A single
non-extreme bar breaks the run. This is the same "cumsum(state changed)"
adjacency-run logic already used throughout this project's segment-building
code (persistence_common / build_regime_chart_data.py), applied to
"is_outlier" instead of "state".

Run: `python v21_outlier_audit.py` (from research_archive/src/).
Writes to results/ (CSVs) and reports/ (outlier_audit_report.md).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

_RESEARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_RESEARCH_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import persistence_common  # noqa: E402
from gap_aware import valid_values_and_lengths  # noqa: E402
from save_production_model import load_masked_features  # noqa: E402

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports")
RAW_OHLCV_PATH = r"C:\Users\MohammedkhezerK\Downloads\New System\ICT_STRUCTURE_RESEARCH\data\BTCUSDT_5M.csv"
MODEL_PATH = os.path.join(RESULTS_DIR, "model_A_HMM4.pkl")

SIGMA_THRESHOLDS = [4, 5, 6]
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    import pickle
    log("Loading existing A-HMM4 model (no refit)...")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model, scaler, feat_cols = artifact["model"], artifact["scaler"], artifact["feature_columns"]

    df, model_cols = load_masked_features()
    assert model_cols == feat_cols, "feature schema mismatch vs saved A-HMM4 artifact"
    valid_df, lengths, valid_index = valid_values_and_lengths(df, model_cols)
    log(f"Total rows scanned (raw feature matrix): {len(df):,}")
    log(f"Total valid rows (existing gap-aware mask, unchanged): {len(valid_df):,}")

    scaled = scaler.transform(valid_df.values)  # TRANSFORM ONLY -- this IS the z-score, no refit
    log("Re-decoding with existing A-HMM4 (no refit)...")
    states = model.predict(scaled, lengths=lengths)
    posteriors = model.predict_proba(scaled, lengths=lengths)
    n_states = model.n_components

    means_df = pd.DataFrame(model.means_, columns=feat_cols)
    from infer_regime import characterize_state
    regime_labels = [characterize_state(means_df.iloc[s]) for s in range(n_states)]
    per_bar_regime = np.array([regime_labels[s] for s in states])
    seg_id = persistence_common.segment_ids_from_lengths(lengths)

    # raw OHLCV, aligned by timestamp (never assume row order)
    raw = pd.read_csv(RAW_OHLCV_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume"],
                       parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    ohlcv = raw.reindex(valid_index)
    assert ohlcv.notna().all(axis=None), "OHLCV alignment produced NaN -- timestamp mismatch, investigate"

    n = len(valid_index)
    log(f"Decoded {n:,} bars, {n_states} states")

    # ===========================================================================
    # PART 2/3: outlier identification (z-scores = scaler-transformed values)
    # ===========================================================================
    log("Computing per-bar extreme-feature flags (vectorized)...")
    abs_z = np.abs(scaled)  # (n, n_features)
    max_abs_z = abs_z.max(axis=1)
    max_z_feat_idx = abs_z.argmax(axis=1)

    counts_by_threshold = {}
    for thr in SIGMA_THRESHOLDS:
        counts_by_threshold[thr] = int((max_abs_z > thr).sum())
        log(f"  observations with >{thr}sigma on at least one feature: {counts_by_threshold[thr]:,}")

    is_outlier_4s = max_abs_z > 4
    extreme_feature_count = (abs_z > 4).sum(axis=1)

    ret_cols_present = [c for c in ["ret_5m", "ret_15m", "ret_30m", "ret_1h", "ret_2h", "ret_4h",
                                     "vol_15m", "vol_1h", "vol_2h"] if c in feat_cols]

    log("Building main outlier table (every >4sigma timestamp) -- vectorized extraction, per-row only for the "
        "variable-length extreme_features string...")
    outlier_positions = np.where(is_outlier_4s)[0]
    # pre-extract everything to plain numpy arrays ONCE -- avoids slow repeated .iloc[]/label lookups
    o_open = ohlcv["open"].values
    o_high = ohlcv["high"].values
    o_low = ohlcv["low"].values
    o_close = ohlcv["close"].values
    o_volume = ohlcv["volume"].values
    valid_vals = valid_df.values  # (n, n_features), column order == feat_cols
    ret_col_idx = {rc: feat_cols.index(rc) for rc in ret_cols_present}
    ts_arr = np.asarray(valid_index)

    n_out = len(outlier_positions)
    feat_cols_arr = np.array(feat_cols)
    out_max_zfeat = feat_cols_arr[max_z_feat_idx[outlier_positions]]
    out_max_zfeat_value = valid_vals[outlier_positions, max_z_feat_idx[outlier_positions]]

    rows = []
    for k in tqdm(range(n_out), desc="Building outlier rows"):
        p = outlier_positions[k]
        feats_over_4 = feat_cols_arr[abs_z[p] > 4]
        row = {
            "timestamp_utc": ts_arr[p], "open": o_open[p], "high": o_high[p], "low": o_low[p],
            "close": o_close[p], "volume": o_volume[p],
            "hmm_state": int(states[p]), "regime_label": per_bar_regime[p],
            "extreme_feature_count": int(extreme_feature_count[p]),
            "extreme_features": ";".join(feats_over_4.tolist()),
            "max_abs_zscore": float(max_abs_z[p]), "max_zscore_feature": out_max_zfeat[k],
            "max_zscore_feature_value": float(out_max_zfeat_value[k]),
            "max_zscore": float(scaled[p, max_z_feat_idx[p]]),
        }
        for rc, ci in ret_col_idx.items():
            row[rc] = float(valid_vals[p, ci])
        rows.append(row)
    outlier_df = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    outlier_df.to_csv(os.path.join(RESULTS_DIR, "outlier_events_all_4sigma.csv"), index=False)
    log(f"  saved {len(outlier_df):,} rows -> outlier_events_all_4sigma.csv")

    # ===========================================================================
    # PART 4: top 100 |ret_5m| extreme moves
    # ===========================================================================
    log("Building top-100 |ret_5m| table...")
    ret5m = valid_df["ret_5m"].values
    ret5m_z = scaled[:, feat_cols.index("ret_5m")]
    top_idx = np.argsort(-np.abs(ret5m))[:100]
    top_rows = []
    for p in top_idx:
        top_rows.append({
            "timestamp_utc": valid_index[p], "open": ohlcv["open"].iloc[p], "high": ohlcv["high"].iloc[p],
            "low": ohlcv["low"].iloc[p], "close": ohlcv["close"].iloc[p], "volume": ohlcv["volume"].iloc[p],
            "ret_5m": float(ret5m[p]), "ret_5m_pct": float(ret5m[p]) * 100, "ret_5m_zscore": float(ret5m_z[p]),
            "hmm_state": int(states[p]), "regime_label": per_bar_regime[p],
            "max_abs_zscore": float(max_abs_z[p]), "max_zscore_feature": feat_cols[max_z_feat_idx[p]],
        })
    top100_df = pd.DataFrame(top_rows).sort_values("ret_5m_pct", key=lambda s: s.abs(), ascending=False)
    top100_df.to_csv(os.path.join(RESULTS_DIR, "outlier_ret5m_top100.csv"), index=False)
    log(f"  saved top 100 -> outlier_ret5m_top100.csv")

    # ===========================================================================
    # PART 5: State 3 verification (recomputed from scratch, not reused)
    # ===========================================================================
    log("Recomputing State-3 verification (from actual files, not prior run)...")
    state3_verification = []
    state_dist_verification = []
    abs_ret5m = np.abs(ret5m)
    for pct in [0.1, 0.5, 1.0]:
        thresh = np.percentile(abs_ret5m, 100 - pct)
        mask = abs_ret5m >= thresh
        n_obs = int(mask.sum())
        state3_count = int((states[mask] == 3).sum())
        state3_verification.append({"threshold_pct": pct, "number_of_observations": n_obs,
                                     "state_3_count": state3_count, "state_3_percentage": state3_count / n_obs * 100})
        dist = pd.Series(states[mask]).value_counts(normalize=True).sort_index() * 100
        for s in range(n_states):
            state_dist_verification.append({"threshold_pct": pct, "state": s, "pct_of_extreme_obs": dist.get(s, 0.0)})
    pd.DataFrame(state3_verification).to_csv(os.path.join(RESULTS_DIR, "outlier_state3_verification.csv"), index=False)
    pd.DataFrame(state_dist_verification).to_csv(os.path.join(RESULTS_DIR, "outlier_state_distribution.csv"), index=False)
    for row in state3_verification:
        log(f"  top {row['threshold_pct']}%: n={row['number_of_observations']}, "
            f"state_3={row['state_3_count']} ({row['state_3_percentage']:.2f}%)")

    # ===========================================================================
    # PART 6: outlier event grouping (documented rule -- see module docstring)
    # ===========================================================================
    log("Grouping consecutive >4sigma bars into outlier events...")
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = (is_outlier_4s[1:] != is_outlier_4s[:-1]) | (seg_id[1:] != seg_id[:-1])
    run_id = np.cumsum(changed)
    tmp = pd.DataFrame({"pos": np.arange(n), "is_outlier": is_outlier_4s, "run_id": run_id,
                        "timestamp": valid_index, "max_abs_z": max_abs_z})
    event_runs = tmp[tmp["is_outlier"]].groupby("run_id").agg(
        start_pos=("pos", "first"), end_pos=("pos", "last"), start_timestamp=("timestamp", "first"),
        end_timestamp=("timestamp", "last"), n_bars=("pos", "size"), peak_max_abs_z=("max_abs_z", "max"),
    ).reset_index(drop=True)
    event_runs.insert(0, "event_id", np.arange(len(event_runs)))
    log(f"  {len(event_runs):,} outlier events from {int(is_outlier_4s.sum()):,} outlier bars")

    # ===========================================================================
    # PART 7/8: before/after analysis for "major" events (top 30 by peak severity,
    # matching Section 11's "largest 20" reference with a small safety margin)
    # ===========================================================================
    log("Computing before/after state/volatility/return analysis for major events...")
    major_events = event_runs.nlargest(30, "peak_max_abs_z").sort_values("event_id").reset_index(drop=True)
    offsets_before = {"15m": 3, "30m": 6, "1h": 12}
    offsets_after = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}
    close_vals = ohlcv["close"].values
    log_close = np.log(close_vals)

    major_rows = []
    for _, ev in major_events.iterrows():
        sp, ep = int(ev["start_pos"]), int(ev["end_pos"])
        row = {"event_id": int(ev["event_id"]), "start_timestamp": ev["start_timestamp"],
               "end_timestamp": ev["end_timestamp"], "n_bars": int(ev["n_bars"]),
               "peak_max_abs_z": float(ev["peak_max_abs_z"]), "state_at_event": int(states[sp])}
        for label, off in offsets_before.items():
            bp = sp - off
            row[f"state_before_{label}"] = int(states[bp]) if bp >= 0 and seg_id[bp] == seg_id[sp] else None
        for label, off in offsets_after.items():
            ap = ep + off
            row[f"state_after_{label}"] = int(states[ap]) if ap < n and seg_id[ap] == seg_id[ep] else None
            if ap < n and seg_id[ap] == seg_id[ep]:
                row[f"return_after_{label}"] = float(log_close[ap] - log_close[ep])
            else:
                row[f"return_after_{label}"] = None
        prev_state = states[sp - 1] if sp > 0 and seg_id[sp - 1] == seg_id[sp] else None
        row["state_3_started_at_event"] = bool(states[sp] == 3 and prev_state != 3)
        if states[sp] == 3:
            dur = 1
            k = sp + 1
            while k < n and seg_id[k] == seg_id[sp] and states[k] == 3:
                dur += 1
                k += 1
            row["state_3_duration_after_event_bars"] = dur
        else:
            row["state_3_duration_after_event_bars"] = 0
        vol_before_pos = max(0, sp - 12)
        vol_after_pos = min(n, ep + 12)
        row["volatility_before_1h"] = float(np.std(ret5m[vol_before_pos:sp])) if sp > vol_before_pos else None
        row["volatility_after_1h"] = float(np.std(ret5m[ep + 1:vol_after_pos])) if vol_after_pos > ep + 1 else None
        major_rows.append(row)
    major_df = pd.DataFrame(major_rows)
    major_df.to_csv(os.path.join(RESULTS_DIR, "outlier_event_summary.csv"), index=False)
    log(f"  saved {len(major_df):,} major-event rows -> outlier_event_summary.csv")

    # case classification: isolated vs sustained
    def classify_case(row):
        after_states = [row.get(f"state_after_{h}") for h in ["15m", "30m", "1h"]]
        after_states = [s for s in after_states if s is not None]
        if row["state_3_duration_after_event_bars"] >= 12:  # stayed in state 3 for >= 1h
            return "B: sustained high-vol regime"
        elif after_states.count(3) == 0:
            return "A: isolated extreme event"
        else:
            return "mixed / short-lived"
    major_df["case_classification"] = major_df.apply(classify_case, axis=1)

    # ===========================================================================
    # PART 9: State 3 analysis
    # ===========================================================================
    log("State 3 analysis...")
    state3_mask = states == 3
    n_state3 = int(state3_mask.sum())
    state3_over4 = int((extreme_feature_count[state3_mask] > 0).sum())
    state3_over5 = int((abs_z[state3_mask] > 5).any(axis=1).sum())
    state3_over6 = int((abs_z[state3_mask] > 6).any(axis=1).sum())

    # State 3 segments: how many begin at/near (within 1 bar) an outlier event start
    state3_changed = np.empty(n, dtype=bool)
    state3_changed[0] = True
    state3_changed[1:] = (states[1:] != states[:-1]) | (seg_id[1:] != seg_id[:-1])
    state3_run_id = np.cumsum(state3_changed)
    state3_segs = pd.DataFrame({"pos": np.arange(n), "state": states, "run_id": state3_run_id})
    state3_seg_starts = state3_segs[state3_segs["state"] == 3].groupby("run_id")["pos"].first().values
    outlier_positions_set = set(outlier_positions.tolist())
    near_outlier = sum(1 for sp in state3_seg_starts if any((sp + d) in outlier_positions_set for d in range(-1, 2)))
    n_state3_segments = len(state3_seg_starts)

    # ===========================================================================
    # PART 10: HMM transition check around outliers vs normal bars
    # ===========================================================================
    log("HMM transition check (outlier vs normal timestamps)...")
    state_changed_at = np.zeros(n, dtype=bool)
    state_changed_at[1:] = (states[1:] != states[:-1]) & (seg_id[1:] == seg_id[:-1])

    def transition_window_pct(mask):
        # fully vectorized: shift state_changed_at by +-1 position (numpy roll-with-edge-fill, not
        # a Python loop -- important since "normal bars" is ~800K+ elements)
        idxs = np.where(mask)[0]
        total = len(idxs)
        if total == 0:
            return {"n": 0, "pct_changed_at": None, "pct_changed_before": None, "pct_changed_after": None}
        changed_at = state_changed_at[idxs]
        prev_idx = np.clip(idxs - 1, 0, n - 1)
        next_idx = np.clip(idxs + 1, 0, n - 1)
        changed_before = state_changed_at[prev_idx] & (idxs > 0)
        changed_after = state_changed_at[next_idx] & (idxs + 1 < n)
        return {"n": total, "pct_changed_at": float(changed_at.mean() * 100),
                "pct_changed_before": float(changed_before.mean() * 100),
                "pct_changed_after": float(changed_after.mean() * 100)}

    outlier_trans = transition_window_pct(is_outlier_4s)
    normal_trans = transition_window_pct(~is_outlier_4s)
    log(f"  outlier bars: {outlier_trans}")
    log(f"  normal bars:  {normal_trans}")

    runtime = time.time() - t0

    # ===========================================================================
    # Final counts + report
    # ===========================================================================
    total_outlier_events = len(event_runs)
    counts_summary = {
        "total_rows_scanned": len(df), "total_valid_rows": len(valid_df),
        "total_4sigma_obs": counts_by_threshold[4], "total_5sigma_obs": counts_by_threshold[5],
        "total_6sigma_obs": counts_by_threshold[6], "total_outlier_events": total_outlier_events,
        "total_top100_ret5m_events": len(top100_df), "total_state3_obs": n_state3,
        "total_state3_pct_of_dataset": n_state3 / n * 100,
        "total_state3_outlier_obs_over4s": state3_over4,
        "total_state3_outlier_pct": state3_over4 / n_state3 * 100 if n_state3 else None,
        "total_state3_over5s": state3_over5, "total_state3_over6s": state3_over6,
        "n_state3_segments": n_state3_segments, "n_state3_segments_near_outlier": near_outlier,
        "pct_state3_segments_near_outlier": near_outlier / n_state3_segments * 100 if n_state3_segments else None,
        "outlier_transition_pct_at": outlier_trans["pct_changed_at"],
        "normal_transition_pct_at": normal_trans["pct_changed_at"],
        "runtime_s": runtime,
    }
    for k, v in counts_summary.items():
        log(f"  {k}: {v}")

    write_report(counts_summary, state3_verification, major_df, outlier_trans, normal_trans,
                  top100_df, outlier_df)
    log(f"\nTotal runtime: {runtime:.1f}s ({runtime/60:.1f} min)")


def write_report(counts, state3_verif, major_df, outlier_trans, normal_trans, top100_df, outlier_df):
    lines = []
    lines.append("# Outlier Timestamp + OHLCV Audit -- A-HMM4\n\n")
    lines.append("DIAGNOSTIC ONLY. No production files, features, scaler, or HMM were modified. "
                 "Reuses the existing, already-fit A-HMM4 model (results/model_A_HMM4.pkl), re-decoded "
                 "(not refit) against the existing gap-aware feature matrix.\n\n")

    lines.append("## Summary counts\n\n")
    for k, v in counts.items():
        lines.append(f"- **{k}**: {v}\n")

    lines.append("\n## Q1-3: Extreme observation counts\n\n")
    lines.append(f"- >4sigma: {counts['total_4sigma_obs']:,}\n- >5sigma: {counts['total_5sigma_obs']:,}\n"
                 f"- >6sigma: {counts['total_6sigma_obs']:,}\n")

    lines.append("\n## Q4: Which features produce the most extreme observations\n\n")
    feat_counts = outlier_df["max_zscore_feature"].value_counts()
    lines.append(feat_counts.to_string() + "\n")

    lines.append("\n## Q5-6: Top 20 |ret_5m| events (exact timestamps + OHLCV)\n\n")
    top20 = top100_df.head(20)[["timestamp_utc", "open", "high", "low", "close", "volume", "ret_5m_pct",
                                  "hmm_state", "regime_label"]]
    lines.append(top20.to_string(index=False) + "\n")

    lines.append("\n## Q7: What % of extreme ret_5m observations belong to State 3\n\n")
    lines.append(pd.DataFrame(state3_verif).to_string(index=False) + "\n")

    lines.append("\n## Q8: Isolated vs sustained volatility (major events)\n\n")
    lines.append(major_df["case_classification"].value_counts().to_string() + "\n\n")
    lines.append(major_df[["event_id", "start_timestamp", "peak_max_abs_z", "state_at_event",
                            "state_3_duration_after_event_bars", "case_classification"]].to_string(index=False) + "\n")

    lines.append("\n## Q9: State 3 duration after major events (bars, 1 bar = 5 min)\n\n")
    lines.append(major_df["state_3_duration_after_event_bars"].describe().to_string() + "\n")

    lines.append("\n## Q10: Outlier vs normal timestamps -- HMM transition coincidence\n\n")
    lines.append(f"Outlier bars: {outlier_trans}\n")
    lines.append(f"Normal bars:  {normal_trans}\n")
    if outlier_trans["pct_changed_at"] is not None and normal_trans["pct_changed_at"] is not None:
        lines.append(f"\nOutlier bars are {outlier_trans['pct_changed_at']/max(normal_trans['pct_changed_at'],1e-9):.2f}x "
                     f"more likely to coincide with a state transition than normal bars.\n")

    lines.append("\n## Q11: What does State 3 represent?\n\n")
    lines.append(f"- Total State 3 observations: {counts['total_state3_obs']:,} "
                 f"({counts['total_state3_pct_of_dataset']:.2f}% of dataset)\n")
    lines.append(f"- State 3 observations containing a >4sigma feature: {counts['total_state3_outlier_obs_over4s']:,} "
                 f"({counts['total_state3_outlier_pct']:.2f}% of State 3)\n")
    lines.append(f"- State 3 segments beginning at/near (+/-1 bar) an outlier event: "
                 f"{counts['n_state3_segments_near_outlier']:,} / {counts['n_state3_segments']:,} "
                 f"({counts['pct_state3_segments_near_outlier']:.2f}%)\n\n")
    if counts["total_state3_outlier_pct"] and counts["total_state3_outlier_pct"] < 50 and \
       counts["pct_state3_segments_near_outlier"] and counts["pct_state3_segments_near_outlier"] < 50:
        verdict = "MIXTURE: most State 3 observations/segments are NOT directly outlier-triggered, " \
                  "suggesting State 3 is a genuine (if noisy) high-volatility regime that ALSO reliably " \
                  "absorbs extreme events when they occur -- not purely an outlier bucket."
    elif counts["total_state3_outlier_pct"] and counts["total_state3_outlier_pct"] > 50:
        verdict = "Majority of State 3 bars co-occur with a >4sigma feature -- leans toward 'extreme-event bucket'."
    else:
        verdict = "Evidence is mixed -- see the detailed percentages above rather than a single label."
    lines.append(f"**Verdict (from measured data only): {verdict}**\n")

    lines.append("\n## Q12: Does fat-tail behavior plausibly contribute to rapid state switching?\n\n")
    lines.append("This must be judged from the Q10 transition-coincidence evidence above, not assumed. ")
    if outlier_trans["pct_changed_at"] is not None and normal_trans["pct_changed_at"] is not None and \
       outlier_trans["pct_changed_at"] > normal_trans["pct_changed_at"] * 1.5:
        lines.append("The data DOES support this: outlier bars coincide with state transitions at a "
                     "meaningfully higher rate than normal bars (see ratio above). This is consistent with "
                     "(not proof of) fat-tailed observations contributing to switching, alongside whatever "
                     "other structural drivers exist.\n")
    else:
        lines.append("The data does NOT show a strong disproportionate link between outlier bars and "
                     "state transitions relative to normal bars -- so this analysis does not support "
                     "claiming fat tails are a primary driver of the excessive switching seen elsewhere; "
                     "the switching appears to happen at a broadly similar rate regardless of whether the "
                     "bar is an outlier.\n")

    with open(os.path.join(REPORTS_DIR, "outlier_audit_report.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
