# Fast Timeframe HMM Screening Report

**These are full-history development results and are NOT OOS validation.**

Script: `research_archive/src/v22_fast_timeframe_screening.py`. Runtime: 825s (13.8 min) for the 4 new fits (15m/30m x HMM4/HMM6) + decode-only 5m baseline + D construction. Does not modify any production file.

## 1. Objective

Determine, cheaply, whether moving the HMM's base timeframe from 5m to 15m or 30m (or combining both via a weighted score) produces more coherent/stable regimes than the existing 5m production model -- to decide which configurations (if any) are worth the much more expensive walk-forward validation later. This is a screening filter, not a final answer.

## 2. Exact methodology

- **A (5m)**: existing `results/model_A_HMM4.pkl` / `model_A_HMM6.pkl` (full-history fits from the prior MTF experiment's config "A", identical to the 18-feature production design), **decoded only, not refit**.
- **B (15m)** / **C (30m)**: genuine candles built directly from the raw 5m OHLCV file via clock-aligned resampling (open=first, high=max, low=min, close=last, volume=sum; verified the raw 5m series is perfectly contiguous end-to-end -- 933,924 rows, zero missing timestamps -- so every 15m/30m bucket contains the expected number of real 5m bars; none fabricated). 18 features per timeframe (Section 3 below), single full-history `StandardScaler` fit + `GaussianHMM(covariance_type="diag", n_iter=100)`, 5 restarts (seeds 42-46), best selected by training log-likelihood -- identical architecture/hyperparameters to `save_production_model.py`, only the underlying candles change.
- **D (15m+30m weighted score)**: **no new HMM fit.** Combines B's and C's own state outputs into a continuous score (Section 13 below); not a state-count model, so it is compared separately, not folded into the same table rows as A/B/C.

## 3. Feature mapping (production concept -> 15m/30m equivalent)

| Family | Production (5m) | 15m/30m equivalent | Concept source |
|---|---|---|---|
| Returns (5) | ret_5m/15m/30m/1h/2h/4h | N-bar log(close) diffs at economic 15m/30m/1h/2h/4h (15m-base) or 30m/1h/2h/4h/8h (30m-base) | `features.add_returns` |
| Volatility (4) | vol_15m/1h/2h | rolling std of 1-bar log return, same horizons minus the timeframe's own native step | `features.add_volatility` |
| Trend (3) | *(none in production)* | ema_dist_fast, ema_dist_slow, ema_slope_fast (fast=8 bars, slow=24 bars, slope lag=3 bars) | reused `v20_mtf_hmm_experiment.py`'s own causal EMA design -- the closest already-vetted concept in this codebase, since production itself has no EMA family |
| Volume (2) | vol_change, vol_zscore | same formulas | `features.add_volume_orderflow` (buy_sell_ratio/vwap_dist intentionally excluded -- not in this task's Section 5 feature list) |
| Order flow (2) | ofi_raw, ofi_zscore | **recomputed** as sum(buy_vol)-sum(sell_vol) at the aggregated timeframe | see Section 7 below |
| Distributional (2) | skew_1h, updown_asymmetry | skew_2h (15m) / skew_4h (30m), updown_asymmetry over the same window | `features.add_skew`, window widened -- see Section 7 |

Section 5's `momentum_4`/`momentum_8` are mathematically identical to N-bar log(close) diffs at bars 4/8 -- i.e. identical to columns already present in the returns family (ret_1h/ret_2h at 15m-base, ret_2h/ret_4h at 30m-base). Rather than feed a diagonal-covariance Gaussian HMM two literally-duplicate columns, they are treated as aliases of the existing return columns and were not added a second time.

## 4. Runtime

| Step | Time |
|---|---|
| Benchmark (1 restart each, 15m-HMM6 / 30m-HMM6) | 147.6s / 32.8s |
| Estimated total (printed before proceeding) | ~26.3 min |
| **Actual total** | **13.8 min** (825s) |
| 15m-HMM4 fit | 116.0s |
| 15m-HMM6 fit | 246.0s |
| 30m-HMM4 fit | 42.6s |
| 30m-HMM6 fit | 94.5s |
| 5m baseline (decode only, both N) | ~1 min |
| D construction (both N) | ~10s |

## 5-8. Results by timeframe

### 5m baseline (existing, decode-only)
923,766 valid bars. HMM-4: 88,738 transitions (96.06/1k bars, **27.37/day**), median segment 30min, 4/4 distinct labels. HMM-6: 114,888 transitions (124.37/1k bars, **35.44/day**), median segment 25min, 4/6 distinct labels (2 collision pairs).

### 15m results
309,721 valid bars (from 311,308 genuine candles). HMM-4: 45,493 transitions (146.88/1k bars, **14.03/day**), median segment 60min, mean 102.1min, 4/4 distinct labels, LL/obs=-15.52. HMM-6: 58,361 transitions (188.43/1k bars, **18.00/day**), median segment 60min, 4/6 distinct labels, LL/obs=-14.00.

### 30m results
154,657 valid bars (from 155,654 genuine candles). HMM-4: 80,589 transitions (521.08/1k bars, **24.86/day**), median segment 30min, mean 57.6min, 4/4 distinct labels, LL/obs=-11.94. HMM-6: 74,366 transitions (480.84/1k bars, **22.94/day**), median segment 30min, **5/6 distinct labels** (only 1 collision pair -- the cleanest label separation of any HMM-6 config tested), LL/obs=-10.91.

### 15m+30m 60/40 results (D)
309,702 valid rows (15m grid). No fitted HMM -- `direction_score = 0.6*trend_score_15m[state] + 0.4*trend_score_30m[state]`, binned Bullish/Ranging/Bearish at +-0.3 (same threshold convention used everywhere else in this project). D-HMM4 (built from the HMM4 pair): 12,026 transitions (38.83/1k bars, **3.71/day**), median segment 120min. D-HMM6 (from the HMM6 pair): 28,312 transitions (91.42/1k bars, **8.73/day**), median segment 75min. Empirical (not fitted) transition matrix -- see Section 13.

**Caution**: D's much lower transition rate is partly mechanical -- 3 coarse bins with a wide "Ranging" middle bucket naturally absorb more noise than 4-6 full states. Not directly comparable to A/B/C on transition count alone (see Section 17/Section 12 in the code's Section-17 checklist).

## 9. HMM-4 comparison

| Config | TF | States | Trans/1k | **Trans/day** | Median dur | Mean dur | >1h | >2h | >4h | Labels | LL/obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 5m | 4 | 96.06 | 27.37 | 30min | 52.0min | 22.1% | 7.3% | 2.2% | 4/4 | n/a (existing) |
| B | 15m | 4 | 146.88 | **14.03** | 60min | 102.1min | 45.6% | 22.3% | 7.0% | 4/4 | -15.52 |
| C | 30m | 4 | 521.08 | 24.86 | 30min | 57.6min | 18.1% | 4.1% | 1.3% | 4/4 | -11.94 |
| D | 15m+30m | 3-bin | 38.83 | 3.71 | 120min | 385.2min | 66.6% | 46.9% | 28.8% | 3/3 | n/a (no fit) |

## 10. HMM-6 comparison

| Config | TF | States | Trans/1k | **Trans/day** | Median dur | Mean dur | >1h | >2h | >4h | Labels | LL/obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 5m | 6 | 124.37 | 35.44 | 25min | 40.2min | 16.8% | 4.0% | 0.8% | 4/6 | n/a (existing) |
| B | 15m | 6 | 188.43 | **18.00** | 60min | 79.6min | 39.9% | 16.1% | 3.4% | 4/6 | -14.00 |
| C | 30m | 6 | 480.84 | 22.94 | 30min | 62.4min | 20.2% | 6.1% | 2.1% | **5/6** | -10.91 |
| D | 15m+30m | 3-bin | 91.42 | 8.73 | 75min | 164.0min | 50.4% | 30.3% | 14.7% | 3/3 | n/a (no fit) |

**Key clock-time finding** (Section 17's own warning against comparing raw transition counts across timeframes is exactly why this column matters): by transitions/day, **15m is the most stable of A/B/C at both state counts** -- more stable than 30m, not just more stable than 5m. This is not monotonic with coarseness: 30m switches *more* often per day than 15m does, at both HMM-4 and HMM-6.

## 11. State characteristics (trend-score separation)

Using each config's own z-scored trend_score (same RETURN_COLS-mean quantity `characterize_state`/`characterize_state_tf` use):

| Config | Directional states' \|trend_score\| |
|---|---|
| 5m-HMM4 | 0.343 (Up), 0.344 (Down) |
| 15m-HMM4 | 0.373 (Up), 0.359 (Down) |
| 30m-HMM4 | **0.963 (Up), 0.749 (Down)** |

30m's directional states are substantially more cleanly separated (nearly 2-2.5x the magnitude) than 5m's or 15m's -- 30m's Uptrend/Downtrend states are "purer" trend states, even though 30m switches more often per day than 15m. This is a genuine trade-off, not a strict ranking: **15m wins on stability, 30m wins on state purity.**

## 12. Transition/duration analysis

5m-HMM4's `pct_gt1h` (22.1%) sits between 15m-HMM4 (45.6%) and 30m-HMM4 (18.1%) -- 15m produces roughly double the share of hour-plus regimes that either 5m or 30m does. 30m-HMM4's very high `pct_1bar` (55.6% of segments last exactly 1 bar = 30min) is notable: more than half of all 30m regime segments are a single bar, meaning a large share of 30m's nominal "stability" (in transitions/1k-bars terms) is actually short, choppy single-bar flips -- consistent with 30m's higher transitions/day than 15m despite 30m using fewer, coarser bars.

## 13. MTF alignment methodology (D)

`DIRECTION_SCORE_t = 0.6 * trend_score_15m[state_15m_t] + 0.4 * trend_score_30m[state_30m_t]` (unweighted by confidence, thresholded at +-0.3 into Bullish/Ranging/Bearish). `MTF_SCORE_t` is the same weighted combination but with each timeframe's score additionally multiplied by its own posterior confidence, used only for the future-return coherence check (Section 14).

**Causal alignment**: a 30m candle at timestamp t0 (left-labeled) covers [t0, t0+30m) and is only fully known at t0+30m. Its score is stamped "available at" t0+30m and joined onto the 15m grid via `merge_asof(..., direction="backward")` -- for every 15m timestamp, the 30m score used is from the most recently **completed** 30m bar, never the one currently forming. Verified directly on a held-out slice (not assumed): a 15m bar was confirmed to only ever see a 30m score whose `available_at <= timestamp`, and the still-forming next 30m bar's score was confirmed absent from that row.

Empirical (observed, not fitted) bin-to-bin transition matrix, D-HMM4:
| from\to | Bearish | Ranging | Bullish |
|---|---|---|---|
| Bearish | 0.915 | 0.067 | 0.018 |
| Ranging | 0.010 | 0.978 | 0.012 |
| Bullish | 0.020 | 0.100 | 0.879 |

**Future-return coherence (Section 14, descriptive only)**: Pearson correlation of the confidence-weighted `mtf_score` against forward returns is **near zero at every horizon** (15m: 0.015, 30m: 0.011, 1h: 0.006, 2h: 0.0005, 4h: 0.004 for the HMM4-built D; similarly small for HMM6-built D). More strikingly, the discrete bin's mean forward return runs **opposite to its label** at the longer horizons: the "Bearish" bin's mean 1h/2h/4h forward return is *positive* (+0.010%, +0.024%, +0.019%), and its `p_positive` is 0.53-0.54 (more likely to go up than down afterward); the "Bullish" bin's mean 2h/4h forward return is smaller/negative and its `p_positive` is 0.48-0.49. This looks like weak mean-reversion, not momentum -- the opposite of what a "Bullish=continue up" label would predict. This is a real, measured finding, not a forced negative: **D's continuous score does not show the coherence a "positive=bullish, negative=bearish" directional signal would need**, at least not via this simple weighted-trend-score construction.

## 14. Limitations

- Single seeded 5-restart fit per config, no cross-validation of the screening result itself -- exactly why this is a *screening* step, not a final answer.
- 30m's `pct_1bar`=55.6% (HMM-4) suggests a meaningful share of its apparent segments are single-bar noise, which the transitions/day metric alone doesn't fully separate from genuine short regimes.
- D's construction is a simple linear combination of two independently-fit models' outputs, not a jointly-optimized model; its weak future-return coherence may reflect that simplicity rather than ruling out 15m/30m combination entirely.
- Two real, pre-existing hardcoded-assumption bugs were found while building this and are NOT fixed in the original files (per "do not modify existing methodology"), only worked around locally:
  1. `gap_aware.build_validity_mask()` accepts `max_lookback_bars` but hardcodes `.rolling(48, ...)` internally (already disclosed in the prior MTF report).
  2. **New**: `gap_aware.segment_lengths()` hardcodes `BAR_INTERVAL = pd.Timedelta(minutes=5)` at module level -- calling it on 15m/30m data would flag every single row as a new segment, since real 15m/30m spacing never equals 5 minutes. A parameterized local equivalent (`segment_lengths_native`) was used instead.
- Two implementation bugs were found and fixed during this task (not pre-existing, introduced and caught while building it):
  3. The literal "economic 1h" window is only 2 bars at 30m-base, which is structurally too small for `pandas.Series.rolling().skew()` (undefined by the skew formula's own `(n-2)` denominator for n<3) -- `skew_1h` was NaN for 100% of rows in an initial smoke test, zeroing out all 30m validity. Fixed by using the shortest window in each timeframe's own horizon ladder that is >=8 bars (15m's "2h", 30m's "4h") -- named accordingly (`skew_2h`/`skew_4h`), not mislabeled as "1h".
  4. `analyze_mtf()`'s `pd.DatetimeIndex(merged["timestamp"].values)` (calling `.values` first) silently dropped the UTC timezone, producing a tz-naive index; reindexing the tz-aware `agg_15m["close"]` against it matched zero rows (not an error -- silent all-NaN), which zeroed out the entire Section 14 future-return/correlation analysis for D on the first run. Caught because the output CSVs were unexpectedly empty, root-caused, and fixed (`pd.DatetimeIndex(merged["timestamp"])`, no `.values`) -- Section 13's numbers above are from the corrected recomputation.

## 15. Candidate configurations for the later walk-forward test

Per Section 17, no config is selected on transition count alone. Weighing stability + state separation + persistence + label distinctness + absence of pathological collapse + future-return signal together:

- **15m-HMM4: strongest overall candidate.** Most stable by clock-time transition rate of any config tested (14.03/day, beating even 5m and 30m), clean 4/4 label separation, no pathological single-bar collapse (`pct_1bar`=13.9% vs 30m's 55.6%), reasonable state purity (trend_score ~0.37).
- **30m-HMM6: interesting secondary candidate**, on different grounds than 15m -- cleanest label separation of any HMM-6 config (5/6 distinct, vs 4/6 everywhere else) and by far the strongest directional state purity (trend_score up to 0.96), at the cost of a higher single-bar-segment share and no improvement in transitions/day over 15m.
- **D (15m+30m weighted score) as currently constructed: not recommended to carry forward.** Its low transition rate is largely mechanical (coarse 3-bin design), and its own future-return coherence check shows near-zero correlation and a possible mean-reversion (not momentum) pattern -- opposite of what its "Bullish"/"Bearish" labeling implies. If MTF combination is still of interest, a jointly-fit or differently-weighted construction would need to be screened again before this one is trusted.
- **30m-HMM4**: not recommended on its own -- no better on transitions/day than 5m, and >55% single-bar segments raise real doubt about whether its nominal "fewer, bigger" candles are actually cleaner regimes rather than just fewer, choppier ones.

This is a screening result only. Nothing here is a trading recommendation, and the walk-forward test itself was **not run** as part of this task.
