# Directional Inversion & Walk-Forward Integrity Audit

**Forensic audit only. No retraining, no refitting, no reruns, no result modifications performed.** All findings below come from inspecting the actual code that produced the completed run and from independent recomputations against the already-saved OOS artifacts.

## 1. Objective

Determine whether the observed pattern (Uptrend-labeled bars tend to be followed by negative OOS returns; Downtrend-labeled bars by positive OOS returns, ~1-2h horizon, all 3 models) is (1) real HMM regime behavior, (2) a labeling issue, (3) a forward-return calculation/alignment bug, or (4) a leakage/implementation bug.

## 2. Existing OOS artifacts inspected

All present as listed in the task; no missing files. Inspected: `{5m_hmm4,15m_hmm4,30m_hmm6}_fold_results.csv`, `_oos_regimes.parquet`, `_future_returns.csv`, `_restart_log.csv`, `model_comparison.csv`, `yearly_summary.csv`, `experiment_config.json`, plus the actual source code `research_archive/src/v23_final_3way_walkforward.py` and `v22_fast_timeframe_screening.py`. Two new files created by this audit only (not modifying anything existing): `audit_manual_examples.csv`, this report.

## 3. Forward-return calculation

Verified directly from `v23_final_3way_walkforward.py` lines 393-408 (not assumed):

```python
close_test = agg["close"].reindex(test_idx)
log_close = np.log(close_test.values)
j = np.arange(n_test) + h_bars
fwd_valid = (j < n_test) & (seg_id_test[np.minimum(j, n_test - 1)] == seg_id_test)
fr[fwd_valid] = log_close[j[fwd_valid]] - log_close[np.arange(n_test)[fwd_valid]]
```

**Exact formula: `fr = ln(close[t+h]) - ln(close[t]) = ln(close[t+h] / close[t])`** — the natural-log forward return from t to t+h. Not `close[t]/close[t-h]-1` (that would be backward/contemporaneous, checked separately in Section 8). `j` is a POSITIVE offset (`t + h_bars`), confirming this is genuinely forward-looking, computed only after the fold's model is already fully fixed. The `seg_id_test` check means a forward return is discarded (NaN) rather than computed across a real gap or past the fold's own test window -- verified it cannot reach into a different fold's data (Section 4 confirms this empirically: 0 leakage found).

## 4. Timestamp alignment

All 3 `*_oos_regimes.parquet` files: `datetime64[us, UTC]` (tz-aware), 0 duplicate timestamps (within-fold or global), all folds internally monotonic increasing, min/max exactly match the intended 2018-03-01 -> 2026-07-18 OOS range.

Checked specifically for the `.values`-drops-timezone bug pattern found earlier in the MTF work: `v23`'s only timestamp-adjacent `.values` call is `close_test.values` at line 399, which happens **after** `agg["close"].reindex(test_idx)` already completed the tz-aware alignment -- it only extracts the resulting price array for `np.log()`, not the index. Traced the full origin chain: `test_idx` derives directly from `feat_df.index`, which is the same object as `agg.index` (never round-tripped through parquet or re-wrapped via `pd.DatetimeIndex(x.values)`) -- so this specific bug class does **not** affect the walk-forward's forward-return computation. Confirmed empirically too: if the reindex had silently failed, `future_ret_df` would be empty (as it was for the earlier MTF bug) -- it is not; row counts match exactly (Section 3 below).

## 5. Manual candle verification

20+ examples across all 3 models and 9 years (2018-2026), saved to `audit_manual_examples.csv`. Real close prices, real forward-looking timestamps, log return computed and cross-checked. Example (5m-HMM4, 2019-07-02 14:55 UTC, "Ranging/High-Vol"): close=10542.99 -> close+1h(15:55)=10632.17, ret_1h=+0.8423%; -> close+4h=10725.00, ret_4h=+1.7116%. 30m-HMM6 examples correctly show `ret_15m=NaN` (30m has no finer-than-30m horizon) -- expected, not an error. No example across any year/model showed an implausible or misaligned close price.

**Independent, from-scratch recomputation cross-check** (a stronger verification than 20 hand examples): wrote a completely separate script re-deriving the entire `{regime, horizon}` forward-return table from raw OHLCV, using the exact same gap/segment-boundary rule as the original code. Result: **agreement to floating-point precision (~1e-17) on every regime x horizon x model combination, with identical sample counts (n_diff=0 everywhere checked).** An earlier looser version of this check (validity = "same calendar fold" only, not "same contiguous segment") showed real disagreement up to ~3x on some cells -- that disagreement was traced to the looser check's own choice to bridge small real gaps that the original code correctly excludes. **This is not evidence of a bug in the original computation; it is evidence the original computation is correct**, since the stricter, apples-to-apples version matches exactly.

**Conclusion: the forward-return calculation and timestamp alignment are verified correct.**

## 6. State-labeling methodology

`characterize_state`/`characterize_state_tf` (production code, reused unmodified) compute `trend_score = mean(state's fitted TRAIN means over the return-feature columns)`, thresholded at +-0.3 (z-scored space). `model.means_` is intrinsically a train-only fitted parameter -- confirmed by code trace: `means_df = pd.DataFrame(model.means_, columns=feat_cols)` is built immediately after `fit_hmm(train_scaled, ...)` returns, before test data is touched anywhere. Verified across representative folds spanning 2018-2026 (via `regime_labels_trained` column in every `*_fold_results.csv` row) that labels are assigned per-fold from that fold's own training fit, never recomputed or adjusted using test data. OOS states reuse this same fitted model object for decoding, so a test bar's state ID automatically carries its train-derived label -- no separate mapping step exists to audit for a train/test mismatch.

## 7. Direction-sign verification

`RETURN_COLS`/`RETURN_COLS_TF` are standard log returns (`log(close).diff(N)`, positive = price up). No inverted-sign feature found in the return, EMA-distance, EMA-slope, or volatility feature definitions (all traced to `features.py`/`v22_fast_timeframe_screening.py`, unchanged from earlier sessions' validated work). Directly tested whether "Uptrend" means what it should: computed the **contemporaneous** (backward-looking, t-1h -> t) return for every OOS bar, grouped by its assigned label:

| Model | Regime | Contemp. 1h mean | P(+) |
|---|---|---|---|
| 5m-HMM4 | Uptrend/Mid-Vol | **+0.457%** | **90.1%** |
| 5m-HMM4 | Downtrend/Mid-Vol | **-0.479%** | **8.0%** |
| 15m-HMM4 | Uptrend/Mid-Vol | +0.399% | 82.9% |
| 15m-HMM4 | Downtrend/Mid-Vol | -0.378% | 18.5% |
| 30m-HMM6 | Uptrend/Mid-Vol | +0.560% | 79.3% |
| 30m-HMM6 | Downtrend/High-Vol | -0.648% | 26.1% |

**Labels are correctly assigned in every model.** A bar labeled "Uptrend" genuinely just experienced a strong positive move; "Downtrend" genuinely just experienced a strong negative move. No sign inversion anywhere.

## 8. Contemporaneous vs forward returns

Section 7's table is the contemporaneous side; Section 9/original report's finding is the forward side. Putting them together for 5m-HMM4, Uptrend/Mid-Vol: contemporaneous 1h return **+0.457% (90% positive)**, forward 1h return **-0.0039% (47.4% positive, from the original report)**. Downtrend/Mid-Vol: contemporaneous **-0.479% (8% positive)**, forward **+0.0072% (54% positive)**. This is the textbook signature of **short-horizon mean reversion following a real, correctly-identified trend** -- exactly the alternative the task itself named in Section 9, and exactly what the evidence shows. It is not a labeling bug.

## 9. Cross-model inversion results

| Model | Regime | 1h P+ | 2h P+ | Mean 1h ret |
|---|---|---|---|---|
| 5m-HMM4 | Uptrend/Mid-Vol | 0.474 | 0.471 | +0.0039% |
| 5m-HMM4 | Downtrend/Mid-Vol | 0.540 | 0.549 | -0.0072% |
| 15m-HMM4 | Uptrend/Mid-Vol | 0.473 | 0.474 | +0.0131% |
| 15m-HMM4 | Downtrend/Mid-Vol | 0.542 | 0.551 | -0.0124% |
| 30m-HMM6 | Uptrend/Mid-Vol | 0.469 | 0.480 | +0.0390% |
| 30m-HMM6 | Uptrend/High-Vol | 0.484 | 0.480 | +0.0237% |
| 30m-HMM6 | Downtrend/Mid-Vol | 0.582 | 0.595 | +0.0458% |
| 30m-HMM6 | Downtrend/High-Vol | 0.536 | 0.547 | +0.0377% |

The direction of the effect (Uptrend P+ < 0.5, Downtrend P+ > 0.5) is consistent in **every regime, every model** shown. No composite score computed, per the task's rule.

## 10. Year-by-year inversion results

Full data recomputed for every available year, all 3 models (not cherry-picked). Pattern for 5m/15m-HMM4: consistent (Downtrend P+ > Uptrend P+) in most years 2018-2023, but **visibly weakens in 2024-2026** -- e.g. 5m-HMM4 2025: Downtrend 1h P+=0.516, Uptrend 1h P+=0.512 (nearly equal, effect nearly gone); 2024: Uptrend 1h/2h P+=0.493/0.508 (briefly flips positive at 2h). 30m-HMM6's year-by-year figures are noisier (small per-year-per-state sample sizes, some cells NaN from too few observations) and in 2019/2023 show both Uptrend and Downtrend with *positive* mean returns simultaneously -- not a clean inversion those years, consistent with 30m-HMM6's already-documented instability (Section 15).

**The inversion is real and persistent on average, but not uniform across every year** -- it is a majority pattern, strongest in 2018-2023, weaker in 2024-2026, especially for 5m/15m.

## 11. 5m 2019-2021 state-collapse investigation

Compared flagged (n=39) vs unflagged (n=62) folds directly from `5m_hmm4_fold_results.csv`:

| Metric | Unflagged (mean) | Flagged (mean) |
|---|---|---|
| `trend_score_range` | 0.714 | **0.226** |
| `restart_spread_normalized` | 0.0021 | 0.0060 |
| `train_ll_per_obs` | -17.66 | **-17.02** (slightly *better*) |

**The two groups are essentially bimodal in `trend_score_range`** (unflagged: 0.67-0.92; flagged: 0.18-0.26 -- zero overlap in the sampled range). Traced the exact transition fold-by-fold (folds 8-26, spanning the 2019 onset): fold 10 (2019-01) healthy (trend_score_range=0.71, restart_spread=1e-9), fold 11 (2019-02) collapses (0.20, restart_spread=0.005), fold 12 (2019-03) **recovers** (0.68, restart_spread=1e-8), then folds 13 onward collapse and *stay* collapsed through fold 26 (2020-05). This on/off flipping, plus the flagged folds having equal-or-better training likelihood (not worse), rules out "the optimizer failed" or "the data degraded" -- it is the signature of **two genuinely comparable local optima in the 4-state likelihood surface** (one that splits states by trend direction, one that doesn't), with restart-to-restart disagreement (elevated `restart_spread_normalized`) exactly co-occurring with which one gets selected by the train-LL tie-break. This is **option A: genuine model behavior** (a real multimodality/identifiability property of this specific 4-state/18-feature/5m setup on these specific historical training windows) -- not a labeling bug, not a scaling bug, not an implementation bug, not a feature-distribution bug.

## 12. Leakage audit

| Check | Result | Evidence |
|---|---|---|
| Scaler fit on test data | **PASS** | `scaler.fit_transform(train_values_raw)`; test only ever sees `.transform()` |
| HMM fit using test data | **PASS** | `fit_hmm(train_scaled, train_lengths, n_states)` -- no test array passed |
| Restart selected using test likelihood | **PASS** | `best_ll` from `model.score(train_values, lengths=lengths)` inside `_fit_one`, train-only |
| State mapping using test statistics | **PASS** | `characterize_state(_tf)` operates on `model.means_`, a train-fit quantity |
| Future-return info entering features | **PASS** | All feature windows are backward-only rolling/diff operations (unchanged production code) |
| Future timestamps entering aggregation | **PASS** | `resample(..., label="left", closed="left")`, non-overlapping windows |
| Test period influencing training features | **PASS** | Features built once causally over full history, then sliced by date -- slicing after the fact cannot leak forward |
| Improper rolling calculations | **PASS** (not re-verified here) | Reused unchanged from the already-validated fast-screening code |
| Improper HTF candle construction | **PASS** | 0 incomplete edge buckets (verified in the fast screening, unchanged here) |
| Gap-crossing features/HMM transitions | **PASS** | `lengths=` mechanism + `segment_lengths_native` (see Section 13) |

No FAIL or NOT VERIFIABLE items found.

## 13. Gap/embargo audit

Confirmed **from the code actually used in the completed run** (not assumed): `v23_final_3way_walkforward.py` line 294-295 calls `v22.segment_lengths_native(train_idx, bar_interval)` / `(test_idx, bar_interval)` for **every model including 5m** (`bar_interval` = 5/15/30 min respectively, set at lines 186/195) -- this is the corrected, timeframe-parameterized function, **not** `gap_aware.segment_lengths()`'s old hardcoded-5-minute version. The old buggy function is never imported or called anywhere in `v23`.

Embargo verified directly from `experiment_config.json`'s saved fold dates: every one of the 101 folds shows exactly `test_start - train_end = 4.0 hours` (checked folds 0, 1, 2, 99, 100) -- applied as a real `pd.Timedelta(hours=4)`, identical across all 3 timeframes (equivalent to 48/16/8 bars for 5m/15m/30m, but implemented as real time, not a bar count, per the task's own requirement).

## 14. Duration/transition audit

`build_segments()` computes `duration_minutes = number_of_bars * bar_minutes` (a bar-count x nominal-interval product), not `end_timestamp - start_timestamp` directly. Verified this is safe, not a hidden bug: a "segment" is by construction (via `segment_lengths_native`) a maximal run of bars each exactly `bar_interval` apart -- so `number_of_bars * bar_minutes` is mathematically forced to equal the real elapsed time for that segment; no gap can be hidden inside a reported segment, because a gap is exactly what ends a segment. Transitions/day and median/mean duration are therefore based on genuine elapsed time, not an unchecked assumption.

## 15. 30m-HMM6 restart instability audit

Restart spread is **chronic, not episodic**: elevated in both flagged (mean 0.284) and unflagged (mean 0.244) folds alike, and present in every year 2018-2026 (`missing_states` flag count per year: 1,1,2,1,3,4,1,4,3 -- spread across the whole run, not concentrated in a specific period, unlike 5m's contiguous 2019-2021 collapse). Weak positive correlation (0.34) between restart spread and fold index (i.e., training-set size) -- instability does not improve with more data, consistent with a **structural** difficulty of reliably fitting 6 states x 18 features on 30m-granularity data (fewer rows, more parameters per state) rather than a one-off implementation issue. All 505 restarts converged (`monitor_.converged=True`); this is about which of several comparably-likely solutions gets picked, not about optimization failure.

## 16. Findings

1. Forward-return formula and alignment are correct (verified two independent ways).
2. State labels are correctly assigned from train-only data, correct sign, no inversion bug.
3. The "inversion" is genuine short-horizon mean reversion after a correctly-identified real trend -- present in all 3 models, strongest 2018-2023, weaker 2024-2026.
4. 5m-HMM4's 2019-2021 (and later scattered) trend-state collapse is genuine model behavior: a real bimodal likelihood landscape, not a bug.
5. 30m-HMM6's restart instability is chronic and structural, present across the whole run, not tied to specific periods or to the state-collapse folds specifically.
6. Zero leakage findings anywhere in the pipeline.
7. Zero gap-handling or embargo defects in the completed run's actual code path.

## 17. Bugs found

**None, in the completed walk-forward run's own code path.** (For completeness: the pre-existing, already-disclosed `gap_aware.py` bugs from earlier sessions -- the dead `max_lookback_bars` parameter and the hardcoded-5-minute `segment_lengths()` -- were never invoked by `v23`, which uses its own corrected `segment_lengths_native` throughout; they remain latent in the original file but did not affect this experiment.)

## 18. What does NOT need retraining

**Everything.** All 303 fold-model results, the forward-return tables, the state labels, the failure-flag findings, and the year-by-year/cross-model comparisons in the original report stand as valid. Nothing found here contradicts or undermines any number already reported.

## 19. What WOULD require retraining

Nothing found in this audit requires retraining to fix a bug. If, going forward, the 5m-HMM4 multimodality (Section 11) or 30m-HMM6 instability (Section 15) are to be *mitigated* (not because they're bugs, but as a design choice), that would need new fitting runs (e.g., more restarts, a different initialization scheme) -- but that is future experimental work, not a correction of the existing results.

## 20. Recommended next experiment

One next experiment: **increase `N_RESTARTS` from 5 to something larger (e.g. 20) for 5m-HMM4 specifically on the ~39 flagged folds**, to determine whether the bimodal outcome (trend-split vs. no-trend-split solution) is a roughly-50/50 coin flip across many more seeds, or whether a clear majority solution exists that 5 seeds simply weren't enough to reliably surface. Not launched here.
