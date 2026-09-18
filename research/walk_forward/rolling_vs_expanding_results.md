# Rolling vs. Expanding: Experiment Results

**Status: Foundational decision, carried forward.** This experiment established the expanding-window training strategy used by every walk-forward validation in this project since, including the final 15m-HMM4 selection.

**The question:** should the HMM regime detector retrain on a fixed rolling window (the original approach — always the most recent 6 months) or an expanding window (all available history, growing every retrain)?

**The method:** 100 walk-forward folds, identical test periods for both strategies, identical HMM architecture (4-state and 6-state, 5 random restarts each, best kept by training log-likelihood — never by test performance), identical gap-aware data handling (no blanket `dropna()` — zero-volume/gap bars are masked, not deleted, and verified against Binance's own historical API to be genuine, not data errors). The only thing that differs between the two tests is how much training history each fold gets.

## Expanding wins, decisively

<img src="../visualizations/win_rate.png" width="600" alt="Expanding-window beats rolling-window in 91/100 folds for HMM-4 and 96/100 folds for HMM-6">

Not a coin flip and not noise — HMM-4 wins 91 times out of 100, HMM-6 wins 96 times out of 100. A paired t-test on the log-likelihood differences puts this at p ≈ 3×10⁻²⁷ (HMM-4) and p ≈ 3×10⁻²⁹ (HMM-6).

## The fit is meaningfully better, not just "more often"

<img src="../visualizations/log_likelihood.png" width="600" alt="Mean out-of-sample log-likelihood per sample: Rolling vs Expanding, HMM-4 and HMM-6">

Expanding isn't squeaking by — it's a real, consistent gap in how well the model explains held-out data, for both architectures.

## And the advantage grows over time

<img src="../visualizations/advantage_over_time.png" width="600" alt="Expanding's advantage over rolling grows from the first 50 folds to the last 50 folds, for both HMM-4 and HMM-6">

Early in the walk-forward series, expanding-window has barely more data than rolling-window does (there isn't much history to expand into yet). By the back half, expanding is training on years of extra data rolling-window has already discarded — and the gap nearly doubles. This is exactly the pattern you'd expect if the fixed 6-month window is genuinely throwing away useful long-horizon signal.

## The honest tradeoff: compute cost

<img src="../visualizations/runtime_cost.png" width="600" alt="Expanding-window costs about 11x more compute than rolling-window for both HMM-4 and HMM-6">

Expanding-window trains on progressively more data every fold — by the last fold it's ~900K bars vs. rolling's fixed ~52K. That shows up directly in compute cost: roughly **11x** more for both models. In production terms this is a single retrain, not a 100-fold backtest — a one-off HMM-4 retrain on the full history takes on the order of minutes, not hours. Still worth knowing the shape of the tradeoff before choosing a retraining cadence.

## Bottom line (as of this experiment)

Expanding-window is the better choice — wins decisively out of sample, the win grows over time, and shows no stability degradation (zero degenerate/collapsed-state folds, zero convergence failures, in either test). At the time of this experiment, HMM-4 was chosen for production over HMM-6: it's ~2.4x cheaper to retrain and its states each map to a distinct human-readable label, whereas HMM-6's states collapse onto fewer distinct labels under the labeling scheme. **This HMM-4 vs. HMM-6 conclusion, and the 5m base timeframe used throughout this experiment, were both later superseded** by the timeframe research (see `../timeframe_comparison/`) and the final 3-way walk-forward (see `final_3way_walkforward_report.md` in this folder) — the expanding-window strategy itself, however, carried forward unchanged into the current 15m-HMM4 production model.

For the full methodology, restart-stability breakdown, gap-handling coverage analysis, and all nine conclusion questions answered against the data: [`rolling_vs_expanding_final_report.md`](rolling_vs_expanding_final_report.md).
