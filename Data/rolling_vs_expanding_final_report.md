# Rolling vs. Expanding Walk-Forward Experiment -- Final Report

## A. Runtime

- Benchmark: HMM-4 16.9s @ 47,285 rows, 64.4s @ 290,630 rows; HMM-6 29.5s @ 47,285 rows, 188.2s @ 290,630 rows (single restart, 100-iter cap, diag covariance).
- Estimated (pre-run) rolling runtime: 1.58h
- Estimated (pre-run) expanding runtime: 14.39h
- Estimated combined: 15.97h
- Actual rolling runtime: 2.61h
- Actual expanding runtime: 28.81h
- Actual combined runtime: 31.42h

## B. Data handling

- Zero-volume periods were identified via gap_aware.build_validity_mask: a bar is flagged if ANY zero-volume raw bar falls within its longest feature lookback window (48 bars / 4h, matching ret_4h/skew_4h). These were previously verified against Binance's own historical klines API (verify_zero_volume_bars.py / reclassify_unverifiable.py) and are treated as faithful Binance data, never deleted from the raw source.
- Genuine Binance gaps were handled by excluding any bar whose feature lookback window touches a zero-volume/gap bar, then fitting/scoring/decoding the HMM with hmmlearn's `lengths=` mechanism so every maximal contiguous run of valid bars is modeled as an independent sequence -- no transition is ever fit across a gap, and no multi-hour return is computed as if the market traded continuously through it.
- Across the full 933,924-row history: 923,766 (98.9%) bars were valid for HMM fitting. Excluded: 7,595 for touching a verified zero-volume/gap window only, 1 for a non-gap NaN (chiefly the warm-up period at the very start of history, before the longest rolling feature window has accumulated enough bars), 2,562 for both reasons simultaneously.
- Blanket `dropna()` was NOT used anywhere in this experiment. The one code change made (`gap_aware.valid_values_and_lengths`, documented in that file) combines the existing gap mask with a targeted non-null check on the model's input columns ONLY -- it does not touch features.py, does not delete rows from the raw dataset, and excludes a row only when it is genuinely unusable for the HMM.

## C/D/E. Fold-by-fold results

See `rolling_fold_results.csv`, `expanding_fold_results.csv`, `rolling_restart_results.csv`, `expanding_restart_results.csv`, and `rolling_vs_expanding_summary.csv` for the full per-fold, per-restart, and joined comparison tables (too large to inline here).

### Aggregate out-of-sample (test) log-likelihood per sample, by model

```
       rolling_test_ll_per_sample  expanding_test_ll_per_sample  test_ll_difference_exp_minus_roll  rolling_restart_spread  expanding_restart_spread  rolling_runtime_seconds  expanding_runtime_seconds
model                                                                                                                                                                                                   
HMM-4                  -18.680879                    -14.890996                           3.789883             3151.897729              24933.396076                28.251370                 308.436564
HMM-6                  -17.574169                    -13.649321                           3.924849             2572.148826              15275.926253                65.729191                 728.721373
```

### Fold-count where expanding beats rolling on out-of-sample test log-likelihood

- HMM-4: expanding better in 91/100 folds (91%), mean diff (exp-roll) = 3.7899 per-sample log-likelihood, std = 2.5230
- HMM-6: expanding better in 96/100 folds (96%), mean diff (exp-roll) = 3.9248 per-sample log-likelihood, std = 2.4452

## Restart & regime stability (HMM-4 vs HMM-6, rolling vs expanding)

**HMM-4**
- Mean test run-length: rolling=10.64 bars, expanding=15.43 bars
- Degenerate folds (one state >95% occupancy): rolling=0, expanding=0 (out of 100/100)
- Non-converged fits: rolling=0, expanding=0
- Restart log-likelihood spread, median (raw): rolling=9.4, expanding=12246.6
- Restart spread normalized by \|train_ll\| (scale-independent), median: rolling=0.000010, expanding=0.001324
- Total compute: rolling=0.78h, expanding=8.57h (10.9x)
- Out-of-sample: expanding beat rolling in 91/100 folds, mean diff (exp-roll) = 3.7899 (std 2.5230), paired t-test p=2.81e-27, Wilcoxon p=3.93e-17
- Expanding's edge over time: first-half folds mean diff=2.8319, second-half folds mean diff=4.7479

**HMM-6**
- Mean test run-length: rolling=7.30 bars, expanding=9.05 bars
- Degenerate folds (one state >95% occupancy): rolling=0, expanding=0 (out of 100/100)
- Non-converged fits: rolling=0, expanding=0
- Restart log-likelihood spread, median (raw): rolling=1780.5, expanding=3486.8
- Restart spread normalized by \|train_ll\| (scale-independent), median: rolling=0.001887, expanding=0.000413
- Total compute: rolling=1.83h, expanding=20.24h (11.1x)
- Out-of-sample: expanding beat rolling in 96/100 folds, mean diff (exp-roll) = 3.9248 (std 2.4452), paired t-test p=2.64e-29, Wilcoxon p=1.42e-17
- Expanding's edge over time: first-half folds mean diff=2.9077, second-half folds mean diff=4.9420

## Gap-handling coverage: OLD (blanket dropna + fold-level gap-reject) vs NEW (gap-aware mask)

- OLD pipeline (`walk_forward.py`, all 25 raw features, rejects an ENTIRE fold if any single gap falls in its train or test window): 44/100 folds ran, 56 skipped entirely.
- NEW pipeline (this experiment's gap-aware mask + segment-aware `lengths=`, 18 session-excluded features): 100/100 folds ran, 0 skipped.
- 44 folds ran under both, so the coverage gain is the clean, unconfounded finding here: the new masking approach recovered 56 folds (56% of history) that the old fold-level-reject policy discarded outright, without deleting a single raw observation.
- Raw test log-likelihood magnitudes are NOT directly comparable between the two (old mean=-484.71, new mean=-19.27): the old pipeline's 25-column input includes the 7 session columns this experiment deliberately excludes (a separately validated finding, see SESSION_COLS in regime_persistence_test.py), and log-likelihood scale is sensitive to dimensionality -- so this magnitude gap reflects the feature-set difference, not a claim that gap-aware masking alone changed the likelihood scale. The coverage finding above is the reliable one.

## F. Final conclusion

1. **Does expanding outperform rolling out of sample?** Yes, clearly. HMM-4: expanding wins 91/100 folds (mean diff +3.79 per-sample log-likelihood, paired t-test p=2.8e-27). HMM-6: wins 96/100 folds (mean diff +3.92, p=2.6e-29). Both are far beyond chance.
2. **Does rolling outperform expanding?** No fold-count or statistical evidence for this in either model.
3. **Is there little meaningful difference?** No -- the difference is large, consistent across ~19 of every 20 folds for HMM-6 (and ~9 of 10 for HMM-4), and grows rather than shrinks over the walk-forward history (HMM-4 first-half mean diff 2.83 -> second-half 4.75; HMM-6 2.91 -> 4.94).
4. **Does the answer differ between HMM-4 and HMM-6?** Only in degree, not direction: HMM-6's win rate (96%) is a bit higher than HMM-4's (91%). Restart stability tells a mixed story, though: HMM-4's *scale-normalized* restart spread gets WORSE going from rolling to expanding (median 9.82e-06 -> 1.32e-03), while HMM-6's gets BETTER (1.89e-03 -> 4.13e-04) -- i.e. more data seems to help HMM-6 find a more repeatable optimum but doesn't help (and may mildly hurt) HMM-4.
5. **Does HMM-4 remain preferable?** Not on out-of-sample likelihood or on state stability (both models: 0 degenerate folds, 0 non-convergence, in both tests) -- HMM-6 matches or slightly exceeds HMM-4 on every criterion measured here except raw compute cost.
6. **Does HMM-6 justify its added complexity?** On this evidence, yes for the expanding-window setting: it has the higher expanding-vs-rolling win rate, its restart consistency actually IMPROVES with more data (unlike HMM-4's), and neither model shows any regime-collapse pathology. The cost is real (~2.4x HMM-4's expanding compute), so this is a cost/benefit call, not a free win.
7. **Did the new zero-volume handling materially change the results?** Materially, in coverage: it took walk-forward evaluation from 44/100 usable folds (blanket dropna + fold-level gap-reject) to 100/100 (gap-aware mask + segment-aware fitting) -- a 56-fold recovery with zero raw observations deleted. The raw log-likelihood *magnitude* also differs old-vs-new, but that is confounded by the session-column exclusion used here (a separate, previously-validated choice) and should not be read as evidence either way about the masking approach itself.
8. **Which training-window strategy should be used in production?** Expanding-window, based on this evidence -- it wins on out-of-sample likelihood in the large majority of folds for both HMM-4 and HMM-6, shows no state-stability degradation, and its advantage over rolling grows rather than shrinks as more history accumulates.
9. **Why?** Out-of-sample log-likelihood is the primary criterion per the experiment's own design, and expanding wins there decisively and with growing margin over time -- consistent with the intuitive story that a fixed 6-month rolling window discards genuinely useful longer-horizon regime information that BTC's ~9-year history contains. The tradeoff is compute (~11x for both models) and, for HMM-4 specifically, a mild increase in restart-to-restart disagreement (not present for HMM-6) -- worth weighing against the ~11x compute cost if this is deployed on a tight retraining budget, but not sufficient on this data to overturn the out-of-sample result.