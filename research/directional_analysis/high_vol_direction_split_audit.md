# High-Vol Directional Split Audit — 15m-HMM4 OOS Walk-Forward

**Forensic audit only. The HMM was not retrained, refit, or modified. State mapping, transition matrix, covariance, feature set, and the completed 101-fold OOS results are all unchanged.**

## 1. Objective

Test whether the completed 15m-HMM4 walk-forward's "Ranging / High-Vol" state is genuinely direction-neutral, or actually hides two directionally distinct populations (High-Vol-Up / High-Vol-Down) that the current 4-state characterization collapses into one label.

## 2. Exact hypotheses

- **H0**: High-Vol-Up and High-Vol-Down (after a causal split) show no meaningful, persistent forward-return difference.
- **H1**: They show opposite, economically meaningful, multi-year-persistent forward-return differences that survive statistical/control checks.

## 3. Data and OOS methodology

Source: `experiments/final_3way_walkforward/15m_hmm4_oos_regimes.parquet` (292,950 OOS observations, 101 expanding-window folds, 2018-03 → 2026-07-18, unchanged) and `15m_hmm4_fold_results.csv` (per-fold train-derived `regime_labels_trained`, used to map each fold's own arbitrary state IDs to "Ranging / High-Vol" correctly — never assumed constant across folds).

**One recomputation was required and is disclosed here**: the original run never persisted each fold's scaled feature values, only the final state/regime/confidence. To build a per-observation, causally-scaled `trend_score` in the same space `characterize_state_tf` uses, this audit refit each fold's `StandardScaler` on that **same fold's train-only** raw features (identical to what the original experiment did) and applied `.transform()` to that fold's test bars. **No `GaussianHMM.fit()` call occurs anywhere in this audit** — this is a deterministic preprocessing step, not HMM regeneration.

**Row-count integrity check**: an initial version of this recompute (before applying the gap-aware validity mask to the date-sliced feature table) produced 293,952 rows vs. 292,950 actual OOS rows — caught immediately, root-caused (gap-touched-but-non-NaN bars were slipping into the date-based slice), and fixed by applying the same `v22.build_validity_mask_native` mask used everywhere else in this project. After the fix: 292,951 rows (1-row residual, ~0.0003%, disclosed and not chased further).

- **n OOS observations**: 292,950
- **n High-Vol observations (label match)**: 21,112
- **n High-Vol observations with valid trend score**: 21,112 (100% — none dropped)
- **n folds represented**: 101 (all)
- **date range**: 2018-03-01 → 2026-07-18

## 4. Exact trend-score formula

Reused verbatim from the project's existing state-characterization methodology (`v22_fast_timeframe_screening.characterize_state_tf`), applied **per observation** instead of per-state-mean:

```
trend_score(t) = mean( scaled(ret_15m)[t], scaled(ret_30m)[t], scaled(ret_1h)[t],
                        scaled(ret_2h)[t], scaled(ret_4h)[t] )
```

where `scaled(x)` = that fold's own train-only-fit `StandardScaler` applied to feature `x`. These are the exact 5 columns `RETURN_COLS_TF["15m"]` already uses for state characterization — no new feature, no renamed feature, no invented methodology. All 5 inputs are backward-looking log-return features computed from data at or before `t` — no future information enters the score.

## 5. Test A — sign split results

`trend_score > 0` → High-Vol-Up, `< 0` → High-Vol-Down, `== 0` excluded (0 observations excluded in practice — continuous variable). n_up=9,982, n_down=11,130.

| Horizon | Mean Up | Mean Down | Mean Diff | P+ Up | P+ Down | P+ Diff | Boot 95% CI | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| 15m | 0.000144 | -0.0000004 | +0.000144 | 0.490 | 0.527 | -0.037 | [-0.00005, 0.00032] | 0.018 |
| 30m | 0.000164 | 0.000195 | -0.000031 | 0.490 | 0.532 | -0.042 | [-0.00037, 0.00028] | -0.003 |
| 1h | 0.000210 | 0.000439 | -0.000228 | 0.503 | 0.545 | -0.042 | [-0.00079, 0.00029] | -0.016 |
| 2h | 0.000112 | 0.000793 | -0.000680 | 0.505 | 0.549 | -0.044 | [-0.00159, 0.00025] | -0.037 |
| 4h | -0.000155 | 0.001462 | **-0.001617** | 0.506 | 0.561 | -0.055 | **[-0.00318, -0.000001]** | -0.066 |

## 6. Test B — ±0.3 threshold split results

Pre-specified threshold, not tuned on OOS data (the project's own existing ±0.3 characterization threshold). n_up=8,828, n_down=9,830 (11.62% fall in the excluded Range band — see Section 10).

| Horizon | Mean Up | Mean Down | Mean Diff | P+ Up | P+ Down | P+ Diff | Boot 95% CI | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| 15m | 0.000190 | 0.000035 | +0.000155 | 0.489 | 0.533 | -0.044 | [-0.00005, 0.00034] | 0.019 |
| 30m | 0.000230 | 0.000275 | -0.000045 | 0.489 | 0.536 | -0.048 | [-0.00040, 0.00028] | -0.004 |
| 1h | 0.000310 | 0.000646 | -0.000336 | 0.503 | 0.551 | -0.048 | [-0.00089, 0.00024] | -0.024 |
| 2h | 0.000213 | 0.001105 | -0.000892 | 0.506 | 0.554 | -0.049 | [-0.00181, 0.00008] | -0.049 |
| 4h | 0.000109 | 0.001733 | **-0.001624** | 0.508 | 0.564 | -0.057 | **[-0.00312, -0.000076]** | -0.067 |

**Both tests agree**: the bootstrap CI (block size 24 bars = 6h, accounting for autocorrelation) excludes zero **only at the 4h horizon**, and even there Cohen's d ≈ -0.066/-0.067 — a trivially small effect size by any standard convention (below the usual "small effect" threshold of 0.2). At every shorter horizon (15m-2h), the CI includes zero.

## 7. Forward-return results (summary)

The direction of the difference is consistent (Up < Down) at every horizon except 15m, but the magnitude is economically tiny throughout (≤0.16% mean difference at any horizon) and only reaches bootstrap-based statistical distinguishability at 4h, and even then barely (upper CI bound ≈ -0.00008 to -0.000001, essentially touching zero).

## 8. Contemporaneous direction sanity check

**PASS.** Both splits correctly identify opposite current directions:

| | ret_15m | ret_30m | ret_1h | ret_2h |
|---|---|---|---|---|
| High-Vol-Up (Test B) | +0.521% | +0.949% | +1.536% | +2.163% |
| High-Vol-Down (Test B) | -0.516% | -0.939% | -1.542% | -2.248% |

The split is measuring what it claims to measure.

## 9. Year-by-year results

4h mean-return difference (Up − Down) by year: 2018 **-0.00216**, 2019 -0.00113, 2020 -0.00169, 2021 -0.00393, 2022 -0.00139, 2023 **+0.00041**, 2024 -0.00051, 2025 **+0.00102**, 2026 -0.00163.

**The sign flips in 2 of 9 years (2023, 2025)** — the (already weak) effect is not perfectly persistent. Full year-by-year table for all 5 horizons saved in the CSV outputs.

## 10. Trend-score distribution

| | n | mean | median | std | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| ALL High-Vol | 21,112 | -0.093 | -0.132 | 1.634 | -2.40 | -1.24 | -0.13 | 1.09 | 2.24 |
| High-Vol-Up (sign) | 9,982 | 1.264 | 1.152 | 0.939 | 0.13 | 0.62 | 1.15 | 1.69 | 2.82 |
| High-Vol-Down (sign) | 11,130 | -1.311 | -1.182 | 1.066 | -2.97 | -1.71 | -1.18 | -0.63 | -0.13 |

**This is the single strongest finding in this audit.** The "ALL High-Vol" distribution has std=1.63 — far wider than the ±0.3 characterization threshold — and is **not** tightly clustered near zero. Breaking down by the ±0.3 threshold (Test B): **41.82% of High-Vol observations have trend_score > +0.3, 46.56% have trend_score < -0.3, and only 11.62% actually fall in the "genuinely near-zero" -0.3 to +0.3 band.** The vast majority of individual High-Vol bars are, by the project's own existing threshold, individually directional — the "Ranging" label is a property of the *state's pooled average* (opposing directional instances cancel out), not of most individual observations within it.

## 11. Transition-context analysis

| | Prev regime: Uptrend | Prev regime: Downtrend | Next regime: Uptrend | Next regime: Downtrend |
|---|---|---|---|---|
| High-Vol-Up (n=8,828) | **61.25%** | 32.69% | **80.02%** | 19.86% |
| High-Vol-Down (n=9,830) | 21.95% | **72.32%** | 28.01% | **71.81%** |

**Strong, clear asymmetry.** High-Vol-Up bars are overwhelmingly surrounded by Uptrend context both before (61%) and after (80%) they occur; High-Vol-Down bars are overwhelmingly surrounded by Downtrend context (72% before, 72% after). This is a purely descriptive, non-causal-to-the-split diagnostic (computed only after the split, using each bar's already-decoded neighbors) — but it's a substantial, unambiguous pattern.

## 12. Statistical tests / bootstrap confidence intervals

Block bootstrap (2,000 resamples, 24-bar/6h blocks to respect autocorrelation) results are embedded in Sections 5-6 above. Summary: the forward-return difference is **not** statistically distinguishable from zero at 15m/30m/1h/2h for either test; it is barely distinguishable at 4h, with a trivial effect size (Cohen's d ≈ -0.07). No p-hacking or IID-t-test overreach was used, per the task's explicit instruction.

## 13. Control analysis using existing Uptrend/Downtrend states

To confirm the scoring methodology itself works (isolating any High-Vol finding from a broken-methodology explanation): repeated the identical forward-return/bootstrap procedure on the two **already-known-directional** states.

| Horizon | Mean Uptrend | Mean Downtrend | Diff | Boot 95% CI |
|---|---|---|---|---|
| 15m | 0.000137 | -0.000123 | +0.000261 | **[0.000228, 0.000296]** |
| 30m | 0.000148 | -0.000138 | +0.000286 | **[0.000215, 0.000353]** |
| 1h | 0.000159 | -0.000150 | +0.000309 | **[0.000180, 0.000447]** |
| 2h | 0.000196 | -0.000096 | +0.000292 | **[0.000053, 0.000536]** |
| 4h | 0.000231 | -0.000067 | +0.000297 | [-0.000129, 0.000740] |

**The control passes**: the known-directional Uptrend/Downtrend split shows a clear, CI-excludes-zero difference at 4 of 5 horizons, in a magnitude comparable to or larger than what the High-Vol split produced. This confirms the methodology is capable of detecting real, robust directional structure when it exists — the High-Vol split's own weak result is not an artifact of a broken scoring method; it's a genuine measurement of a weaker effect.

## 14. Interpretation

The evidence is **genuinely mixed across different lines of inquiry**, not uniformly for or against H1:

- **Distributional evidence (Section 10) strongly supports H1's premise**: most individual High-Vol observations are, by the project's own threshold, clearly directional — the state's "Ranging" label reflects a pooled average, not most individual bars.
- **Transition-context evidence (Section 11) strongly supports H1's premise**: High-Vol-Up/Down occur in clearly different regime contexts (Up embedded in Uptrend periods, Down embedded in Downtrend periods).
- **Forward-return evidence (Sections 5-7, the primary hypothesis test) does NOT support H1**: the difference is directionally consistent but economically tiny (Cohen's d ~0.02-0.07), only marginally distinguishable from zero at the longest horizon, and flips sign in 2 of 9 years.
- **Control check (Section 13) confirms the methodology works** and, by comparison, highlights just how much weaker the High-Vol effect is relative to genuinely well-separated directional states.

## 15. Decision implications

Per the task's own decision framework: H1 requires the difference to be **economically meaningful, persistent across years, and survive statistical/control checks** — the forward-return evidence satisfies none of these three cleanly (tiny effect size, 2/9-year sign flips, barely-there significance only at 4h). **This is a weak/unstable result on the metric that matters for a splitting decision.**

However, the audit also surfaces a real, separate finding worth keeping: High-Vol *is* internally heterogeneous **descriptively** (contemporaneously and contextually) even though that heterogeneity doesn't convert into a useful **predictive** difference. Splitting the state would make its current-moment characterization more descriptively honest, but would not, on this evidence, buy a materially better forward-looking signal — which is the more decision-relevant question for whether the 4-state model should change.

---

## Final Answer

**Does the existing 15m-HMM4 Ranging / High-Vol state contain two causally identifiable directional populations (High-Vol Up and High-Vol Down) with meaningfully different future-return behavior?**

**Partially, and not in the way that would justify splitting the state.** The state unambiguously contains two directionally distinct populations *right now* (contemporaneously and contextually) — 88% of its observations are individually directional by the project's own threshold, and they cluster around opposite trend contexts before and after. But those two populations do **not** show a meaningfully different, persistent **future**-return profile: the effect is directionally consistent but economically negligible (Cohen's d ≈ -0.07 at best), reaches only marginal statistical significance at the longest horizon tested, and reverses sign in 2 of 9 years. Per the task's own interpretation rule, this is a weak/unstable result — **High-Vol should not be split on this evidence.**

## Outputs

- `high_vol_direction_split.csv` — 21,112 observation-level rows (timestamp, fold_id, hmm_state, original_regime, trend_score, split_A_label, split_B_label, prev/next_regime, contemporaneous returns, forward returns at all 5 horizons)
- This report
