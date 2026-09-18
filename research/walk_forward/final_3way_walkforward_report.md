# Final 3-Way Walk-Forward Validation

**These are genuine out-of-sample (OOS) walk-forward results.** Every fold's model was fit only on data strictly before its own test period (with a 4-hour embargo), and decoded on unseen future data. This is not a full-history development fit.

## 1. Objective

Determine whether the fast-screening advantages found for 15m-HMM4 and 30m-HMM6 (full-history development fits, `research_archive/src/v22_fast_timeframe_screening.py`) survive genuine walk-forward validation against the existing 5m-HMM4 baseline.

## 2. Models

- **A — 5m-HMM4** (existing baseline, same 18 production features)
- **B — 15m-HMM4** (genuine 15m candles, 18 features per the fast-screening design)
- **C — 30m-HMM6** (genuine 30m candles, 18 features, 6 states)

No other models, features, or hyperparameters were tested. D/MTF was not included.

## 3. Dataset

Raw 5m OHLCV, 2017-09-01 → 2026-07-18 (933,924 bars, verified perfectly contiguous). 15m/30m candles built via clock-aligned resampling of the same source. No raw data modified.

## 4. Walk-Forward Methodology

**Expanding window**, reproduced from `research_archive/src/v2_common.py`'s `generate_folds()` — the only actual prior walk-forward implementation in this codebase, and the variant this project's own `Data/rolling_vs_expanding_final_report.md` validated (expanding beat rolling OOS in 91-96% of folds) and `save_production_model.py` adopted for production.

- `WARMUP_MONTHS=6` (train start fixed at 2017-09-01, train end grows every fold)
- `TEST_MONTHS=1`, `STEP_MONTHS=1`
- `EMBARGO=4 hours` (applied as a real time delta, not a bar count — equivalent to 48 5m-bars / 16 15m-bars / 8 30m-bars)
- **101 folds**, first test 2018-03-01, last test 2026-07-01 (through 2026-07-18)

**Note on a real conflict in the task spec**: the task's own illustrative example ("Fold1: Train=Jan-Jun; Fold2: Train=Feb-Jul") describes a *rolling* 6-month window, which conflicts with the actual established/validated/production methodology (*expanding*). Per the task's own higher-priority instruction ("derive boundaries from the existing implementation, do not invent new ones"), expanding was used, since it's the only real prior implementation and the one this project already validated. Disclosed here rather than silently picked.

Per-fold procedure (identical for all 3 models): slice each timeframe's own precomputed feature table by train/test dates → restrict to valid rows, recompute per-slice contiguous segments → fit `StandardScaler` on TRAIN only → fit `GaussianHMM` (diag covariance, n_iter=100) on TRAIN only, 5 restarts (seeds 42-46), best selected by TRAINING log-likelihood only → characterize states from TRAIN-fitted means only → decode the entire test fold in one pass.

## 5. Leakage Controls

- No full-history scaler used — verified in code: `StandardScaler().fit_transform(train_values_raw)` then `.transform(test_values_raw)`, never `.fit()` on test.
- No test data influenced training, restart selection (`best_ll` computed via `model.score(train_scaled, ...)` only), or state labeling (`characterize_state`/`characterize_state_tf` called on `model.means_`, a train-only quantity).
- Test log-likelihood (`test_ll`) is computed and reported, but never used to select anything — confirmed by code inspection: it's computed *after* `best_ll`/`best_seed` are already fixed.
- 4-hour embargo applied as `test_start - pd.Timedelta(hours=4)`, identical real time across all 3 timeframes.
- HMM state IDs are never compared across folds directly — each fold's own `regime_labels` (train-derived) is what's reported, and the primary comparison uses aggregated regime *names*, not raw state indices.
- 15m/30m feature horizons are the same economically-scaled ones validated in the fast screening (not re-derived here).

## 6. Gap Handling

Genuine 15m/30m candles built only from complete windows of real 5m bars (verified 0 incomplete edge buckets). Feature validity separately excludes any bar whose lookback touches a zero-volume/gap bar (via a locally-fixed, timeframe-parameterized equivalent of `gap_aware`'s logic — the two known `gap_aware.py` bugs, the dead `max_lookback_bars` parameter and the hardcoded 5-minute `BAR_INTERVAL` in `segment_lengths()`, are both disclosed but not fixed in the shared file, per "do not modify production code unless required"). `hmmlearn`'s `lengths=` mechanism prevents any transition from being fit across a gap, in every fold, for every model. No raw data modified, no blanket `dropna`.

## 7. 5m-HMM4 Results

879,408 OOS observations across 101 folds. Mean 21.14 transitions/day (median 19.24, std 7.19). Mean duration 77.6 min (median 30-45 min typically). `test_ll_per_obs` mean -14.89.

**Major finding**: in **39/101 folds (38.6%)**, TRAIN characterization produced **zero Uptrend and zero Downtrend labels** — all 4 states came out as "Ranging / {Low,Mid,High}-Vol" variants, with no directional state discovered at all. Concentrated almost entirely in **2019-2021** (10/12, 12/12, 10/12 folds respectively — three consecutive years), plus scattered 2023-2024. This is the dominant failure mode of the baseline model in this experiment.

## 8. 15m-HMM4 Results

292,950 OOS observations. Mean 13.01 transitions/day (median 13.55, std 3.42) — the lowest of the three, consistent with the fast screening. Mean duration 123.1 min. `test_ll_per_obs` mean -12.57.

**Zero failure flags across all 101 folds.** Every fold's training characterization discovered all 4 distinct regime labels (mean `distinct_labels_observed` = 4.0 exactly, every single year 2018-2026).

## 9. 30m-HMM6 Results

146,180 OOS observations. Mean 25.06 transitions/day (median 25.25, std 3.51) — highest of the three. Mean duration 58.6 min. `test_ll_per_obs` mean -8.27 (best raw value, but not comparable across models with different feature/timeframe scale).

**20/101 folds (19.8%) flagged `missing_states`** — one or more of the 6 trained states never appeared in that fold's OOS decode. Mean distinct labels observed = 4.32/6 (consistent with the fast screening's partial label-collision finding). **Restart-to-restart instability is a serious concern**: mean normalized restart spread = 0.252 (25%), vs 5m's 0.0036 and 15m's 0.0006 — 30m-HMM6's 5 random restarts land on meaningfully different local optima far more often than either other model.

## 10. Fold-by-Fold Comparison

Full detail in `{model}_fold_results.csv` (101 rows each). Distributional summary:

| Metric | Model | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| Transitions/day | 5m-HMM4 | 21.14 | 7.19 | 4.94 | 16.55 | 19.24 | 26.67 | 45.01 |
| | 15m-HMM4 | 13.01 | 3.42 | 2.97 | 11.30 | 13.55 | 15.57 | 20.49 |
| | 30m-HMM6 | 25.06 | 3.51 | 11.43 | 23.47 | 25.25 | 27.60 | 30.98 |
| Mean duration (min) | 5m-HMM4 | 77.59 | 35.50 | 31.98 | 53.93 | 73.95 | 86.65 | 285.13 |
| | 15m-HMM4 | 123.05 | 57.02 | 70.19 | 91.50 | 106.03 | 127.06 | 468.99 |
| | 30m-HMM6 | 58.62 | 11.42 | 46.45 | 51.91 | 56.38 | 60.08 | 125.75 |
| OOS LL/obs | 5m-HMM4 | -14.89 | 3.18 | -25.71 | -16.70 | -14.43 | -12.59 | -7.48 |
| | 15m-HMM4 | -12.57 | 3.91 | -26.76 | -14.73 | -12.15 | -9.88 | -4.22 |
| | 30m-HMM6 | -8.27 | 4.05 | -23.78 | -9.95 | -7.95 | -5.28 | -0.73 |
| Max state occupancy % | 5m-HMM4 | 55.02 | 15.42 | 27.66 | 44.39 | 54.39 | 63.72 | 93.13 |
| | 15m-HMM4 | 52.08 | 16.88 | 29.51 | 35.66 | 50.76 | 64.86 | 92.28 |
| | 30m-HMM6 | 41.26 | 6.15 | 21.63 | 37.65 | 41.53 | 45.56 | 52.24 |
| Restart spread (normalized) | 5m-HMM4 | 0.0036 | 0.0037 | 0.0000 | 0.0000 | 0.0022 | 0.0072 | 0.0110 |
| | 15m-HMM4 | 0.0006 | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0204 |
| | 30m-HMM6 | 0.2520 | 0.1046 | 0.0000 | 0.2311 | 0.2794 | 0.3027 | 0.6176 |
| Trend-score range | 5m-HMM4 | 0.53 | 0.24 | 0.18 | 0.24 | 0.68 | 0.70 | 0.92 |
| | 15m-HMM4 | 0.75 | 0.05 | 0.70 | 0.73 | 0.73 | 0.76 | 0.94 |
| | 30m-HMM6 | 1.76 | 0.31 | 1.41 | 1.60 | 1.71 | 1.81 | 2.98 |

A model can look fine on average while failing badly in specific periods — see Section 15 for exactly where.

## 11. Year-by-Year Comparison

Full table in `yearly_summary.csv`. Transitions/day by year (mean_transitions_per_day):

| Year | 5m-HMM4 | 15m-HMM4 | 30m-HMM6 |
|---|---|---|---|
| 2018 | 20.25 | 10.26 | 25.95 |
| 2019 | 15.10 | 11.12 | 23.88 |
| 2020 | 16.99 | 13.69 | 24.84 |
| 2021 | 20.78 | 16.13 | 22.59 |
| 2022 | 27.43 | 14.04 | 25.21 |
| 2023 | 18.87 | 10.65 | 23.56 |
| 2024 | 23.92 | 14.46 | 26.66 |
| 2025 | 22.99 | 12.89 | 26.23 |
| 2026 (partial) | 25.74 | 13.61 | 27.93 |

**15m-HMM4 has the lowest transitions/day of the three in every single year, 2018-2026, with no exception.** Not cherry-picked — this is the full year-by-year record.

Distinct labels observed per year (`mean_n_distinct_labels_observed`): 15m-HMM4 = 4.0 in every year (perfect). 5m-HMM4 drops as low as 2.42 in 2020/2021 (matches Section 7's finding — most folds those years found no directional state). 30m-HMM6 ranges 3.92-4.63/6.

## 12. OOS Regime Stability

Covered in Sections 7-11. Summary: 15m-HMM4 is both the most stable (lowest transitions/day, every year) and the most structurally reliable (zero failure flags, always finds all 4 labels). 5m-HMM4 is moderately stable but structurally unreliable (frequently fails to find any trend state). 30m-HMM6 is the least stable by transitions/day and has real restart-instability and partial-label-collision issues, despite having the cleanest state *separation* when its states are fully formed.

## 13. OOS State Separation

`trend_score_range` (highest-state-mean-return minus lowest, in the model's own z-scored feature space) mean across folds: 30m-HMM6 = 1.76, 15m-HMM4 = 0.75, 5m-HMM4 = 0.53. 30m-HMM6's directional states, when they exist, are far more cleanly separated than either other model's — consistent with the fast-screening finding. This does not offset its stability/reliability problems (Sections 9, 14) — per the task's own interpretation rule, state separation alone does not make a model superior.

## 14. Restart Stability

All 505 restarts (101 folds × 5 seeds) converged for all 3 models — 0 fit failures, 0 non-convergence anywhere (the "Model is not converging" lines seen during the run are hmmlearn's own transient per-iteration EM-delta warnings, not failures; `monitor_.converged` was `True` at the end of every restart).

Restart **spread** (best-vs-worst training log-likelihood among the 5 restarts, normalized by `|best_ll|`) tells a different story: 15m-HMM4 is most consistent (mean 0.0006, i.e. restarts almost always agree), 5m-HMM4 is mildly inconsistent (mean 0.0036), and **30m-HMM6 is seriously inconsistent (mean 0.252, max 0.618)** — its 5 restarts frequently converge to substantially different solutions, meaning the "best" one is more of a lottery than for the other two models.

## 15. Failure Cases

- **5m-HMM4**: 39/101 folds, `no_uptrend_state` + `no_downtrend_state` together (all 4 states characterized as Ranging variants). Concentrated in 2019-2021.
- **30m-HMM6**: 20/101 folds, `missing_states` (fewer than 6 distinct states observed OOS, i.e. 1+ trained states never occupied during that fold's test month).
- **15m-HMM4**: 0/101 folds flagged.
- No covariance collapse, no NaN/Inf values, no impossible timestamps, and no gap-crossing sequences were detected in any fold for any model (all guarded by assertions during the run; none tripped).

## 16. Computational Cost

Benchmark-based pre-run estimate: ~4.0h. **Actual: 8.12 hours** (29,234s) — the estimate under-counted because it averaged only two sample folds (0 and 50) linearly, but the expanding-window design makes per-fold cost convex (later folds, with much larger training sets, cost substantially more than the early-fold-biased linear extrapolation implied). 5m-HMM4 (101 folds, up to ~880K-row training sets) dominated the cost.

## 17. Raw Measurements

All fold-level, restart-level, and future-return data preserved without aggregation in: `5m_hmm4_fold_results.csv`, `15m_hmm4_fold_results.csv`, `30m_hmm6_fold_results.csv`, `{model}_restart_log.csv`, `{model}_future_returns.csv`, `{model}_oos_regimes.parquet` (full per-bar OOS decode, 879K/293K/146K rows respectively), `model_comparison.csv`, `yearly_summary.csv`, `experiment_config.json` (full fold date table, package versions, all hyperparameters).

**OOS forward-return behavior by regime** (Section 15's requirement) — a genuinely striking, consistent-across-all-3-models finding, not present in any prior full-history analysis this session: at 1h-4h horizons, **"Uptrend"-labeled OOS bars show `p_positive` *below* 0.5** (more often followed by a down move than up) in all three models, while **"Downtrend"-labeled bars show `p_positive` *above* 0.5** (more often followed by an up move). Example (1h horizon): Uptrend/Mid-Vol `p_positive` = 0.474 (5m), 0.473 (15m), 0.469 (30m); Downtrend `p_positive` = 0.540 (5m), 0.542 (15m), 0.582-0.536 (30m's two downtrend states). "Ranging/High-Vol" consistently shows the largest positive mean forward return at longer horizons across all 3 models. These are descriptive statistics only, not a trading signal, and are reported exactly as measured — no threshold was tuned and no model was changed based on them.

## 18. Interpretation

Per the task's explicit rule, no composite score or ranking is produced. What the *measurements* show:

- **Stability** (transitions/day): 15m-HMM4 is lowest in every single year tested. This is the fast-screening finding surviving OOS most cleanly of anything tested.
- **Structural reliability**: 15m-HMM4 is flawless (0 failure folds, always finds all 4 labels). 5m-HMM4's own baseline has a severe, previously-unmeasured reliability problem — it fails to discover any directional regime in nearly 40% of folds, concentrated in a 3-year stretch. 30m-HMM6 has a real, distinct reliability problem (restart instability, partial state loss) not previously visible in the full-history screening.
- **State separation**: 30m-HMM6 is best by a wide margin when its states fully form, but its own restart instability (Section 14) means that separation is not reliably reproducible fold-to-fold.
- **Forward-return coherence**: all three models show the same counter-intuitive pattern — trend labels correlate weakly *negatively* with the direction their name implies, OOS, at 1h+ horizons. This is a shared characteristic of the labeling methodology (not specific to any one timeframe) and was not visible in any prior in-sample analysis this session.

## 19. Next Experiment

Based strictly on the measured evidence above (not a recommendation to trade or deploy anything):

1. **The 5m-HMM4 no-trend-state failure mode (Section 7) merits direct investigation** before anything else — it's a real, currently-undiagnosed weakness in the *existing production baseline*, concentrated in 2019-2021, that the fast screening never surfaced (a full-history fit pools enough data to always find a trend state).
2. **15m-HMM4's stability and reliability advantages both survived OOS validation** and are now the most rigorously-evidenced findings in this whole research program — a reasonable next step is deciding whether to test it against a downstream signal (persistence gate / LR), the one thing this experiment deliberately did not touch.
3. **30m-HMM6's restart instability (Section 14) should be understood before its state-separation advantage is trusted** — e.g., checking whether more restarts, or a different initialization, stabilizes it, since 5 restarts currently behave more like a lottery than a converged answer for this specific model.
4. **The Uptrend/Downtrend forward-return inversion (Section 17) is model-independent** (seen in all 3) and worth its own targeted investigation — it suggests the `characterize_state` labeling convention itself may need reconsidering, separate from any timeframe question.
