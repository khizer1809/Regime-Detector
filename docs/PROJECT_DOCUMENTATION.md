# Project Documentation — BTCUSDT Regime Classification System

Internal technical notebook. This is more detailed than `README.md` and is meant to let a new researcher understand not just *what* the production system is, but *why* every major decision was made, in the order it was made, with the actual numbers behind each one.

## 1. Original Objective

Build a system that classifies BTCUSDT's current market environment into a small number of statistically distinct, human-interpretable regimes, using only causal (no-lookahead) information, validated well enough out-of-sample to trust the classification in a live setting. A secondary line of research (later terminated) explored whether the HMM's own regime-confidence signals could predict how long a detected trend would persist.

## 2. Datasets

- Raw source: Binance BTCUSDT 5-minute klines, `2017-09-01 05:00 UTC` → `2026-07-18 23:55 UTC`, 933,924 bars.
- Columns: `open, high, low, close, volume, avg_price, buy_vol, sell_vol, ofi`. `avg_price/buy_vol/sell_vol/ofi` are derived from Binance's public kline fields (`quote_asset_volume`, `taker_buy_base_asset_volume`) — documented proxies, not guaranteed identical to whatever originally produced historical columns of the same name, but internally consistent and exact for `open/high/low/close/volume`.
- A raw-file quirk was found and disclosed (not fixed, since it doesn't affect production which recomputes OFI cleanly at aggregation time): the raw file's own `ofi` column carries an undocumented 1-bar lag relative to `buy_vol - sell_vol` in the same row.

## 3. Data Validation

- Verified programmatically that the raw 5m series has **zero missing timestamps** across its full 9-year span (a diff of consecutive timestamps has exactly one unique value: 5 minutes).
- 2,183 five-minute bars have exactly zero volume — real low-liquidity periods (Binance returns a flat OHLC row even with no trades), not missing data, cross-checked against Binance's historical API in earlier work.
- 15m/30m candle aggregation only accepts a bucket if all underlying 5m bars are present (3 for 15m, 6 for 30m) — a partial edge bucket is dropped rather than built from incomplete data.

## 4. Feature Engineering

See `README.md` §4 for the production feature table. Design principles, in the order they were enforced:

1. **Backward-looking only** — every rolling operation (`.diff()`, `.rolling()`, `.ewm()`) is a pandas backward window; no feature at bar *t* uses information from *t+1* or later.
2. **Economic-time-scaled windows** — a "1h" feature is always 4 bars at 15m (not the 12-bar count that meant 1h at the old 5m base). This was enforced explicitly after the multi-timeframe research made clear that copying bar counts across timeframes silently changes what a feature means.
3. **Map to the closest existing production concept, don't invent new methodology** — the EMA-trend family (not present in the original 5m feature set) was added during the multi-timeframe research and carried into the 15m production set; it reuses the exact same causal EMA-distance/slope design validated there, not a new formula.
4. **A real bug found and fixed during 30m feature-set testing**: `pandas.Series.rolling(window).skew()` is *structurally* undefined (not just noisy) for `window < 3` — the bias-corrected skew formula divides by `(n-2)`. At 30m base, the literal "economic 1h" window is only 2 bars, so `skew_1h` was NaN for 100% of test rows, zeroing out all validity. Fixed by using the shortest window in each timeframe's own horizon ladder that is ≥8 bars (15m's "2h", 30m's "4h"), named accordingly rather than mislabeled "1h".

## 5. Scaling

`sklearn.preprocessing.StandardScaler`, fit once per training window, applied via `.transform()` to test/live data. Every walk-forward fold in this project fits its own scaler on that fold's own training data only — global (full-history) scaling was deliberately avoided for any OOS evaluation, because it would let the test period's own mean/std leak into what counts as "normal," inflating apparent performance. The production model itself is a single full-history fit (there is nothing to hold out for a model about to be deployed forward), which is a distinct, intentional choice from the OOS validation methodology — documented explicitly so the two are never confused.

## 6. HMM Theory (as applied here)

`hmmlearn.hmm.GaussianHMM`, `covariance_type="diag"` (each state's features are assumed independent given the state — a simplification, not a claim that features are truly uncorrelated), `n_iter=100`, 5 random restarts per fit (seeds 42-46), best restart selected by **training** log-likelihood only. This exact hyperparameter set was fixed early (in the 5m production model) and never changed across any later experiment — every timeframe/state-count comparison in this project's history varies only the base timeframe and/or number of states, never the HMM's own architecture, to keep comparisons fair.

## 7. GMM Experiments

No standalone GMM implementation exists in this repository's history. GMM was discussed as the conceptual reference point for explaining why an HMM (which models state *persistence* via a transition matrix) was chosen over a stateless mixture model, not as a separately built and benchmarked system. Documented explicitly rather than implied.

## 8. LR Experiments

Logistic Regression was the original design for the persistence gate (predict whether a detected trend continues for the label horizon). Reused `sklearn.linear_model.LogisticRegression`, 4 features (`confidence, stay_prob, log1p_duration, margin`), chronological train/holdout split with an embargo matching the timeframe's own 4-hour convention (48 bars at 5m, 16 bars at 15m). Holdout AUC: **0.5203** (5m baseline, 4 features), **0.5262** (15m, same 4 features). LR was also directly tested against Random Forest on an expanded 19-feature set (`research/ml_comparisons/`) — LR held up better on a later recency check than RF did, even though RF's pooled AUC looked marginally higher.

## 9. RF Experiments

Tested as a nonlinear alternative to LR on the identical feature set and identical frozen HMM decode (no HMM refit). Base RF AUC 0.5138 (below base LR's 0.5203). With 15 added market-structure features (swing highs/lows, BOS/CHoCH, pullback ratio, etc.), struct-RF reached AUC 0.5232 — briefly the best pooled number in the project — but a stability check (`research/ml_comparisons/v18_struct_rf_stability_check.py`) found a **+0.178 train-vs-OOS AUC gap** and struct-RF losing to plain base-LR in both 2025 (0.476 vs 0.505) and 2026 (0.513 vs 0.522). A 7-configuration regularization sweep (`RF-1` through `RF-7`, varying `max_depth`/`min_samples_leaf`/`min_samples_split`/`max_features`) reduced the gap (best: RF-5, gap +0.123, AUC 0.5248) but **did not fix the recency reversal** — RF-5 still lost to base-LR in the same 2025/2026 test. **Final verdict: RF rejected.** Base LR (AUC 0.5203) remained the strongest, simplest result of that entire line of research.

## 10. XGB Experiments

Not tested. No XGBoost script, config, or result exists anywhere in this project's history. Stated explicitly per the project's "do not fabricate" rule — if this is wanted later, it would be new work, not a rediscovery of something already tried.

## 11. HMM-4

Production architecture. 4 states, each characterized post-fit by thresholding its own fitted mean feature vector (`trend_score`/`vol_score`, ±0.3 z-score convention) into a `{Uptrend, Downtrend, Ranging} × {High-Vol, Mid-Vol, Low-Vol}` label. In practice, across every configuration tested in this project (5m, 15m, 30m), HMM-4 always produces exactly 4 distinct labels when it succeeds — the interesting failure mode (discovered in the final walk-forward) is that it can *also* produce a degenerate solution where all 4 states are various "Ranging" labels (see §16).

## 12. HMM-6

Tested as a higher-capacity alternative at both the 5m and 30m base timeframes. Consistently produces better raw log-likelihood than HMM-4 at the same timeframe, but its extra states routinely collapse onto fewer distinct human-readable labels (5m-HMM6: 4/6 distinct; 30m-HMM6: 4-5/6 distinct across different fits) and, at 30m specifically, showed chronic restart-to-restart instability (normalized restart spread ~0.25, i.e. the 5 random seeds frequently converge to meaningfully different solutions) that was present in both flagged and unflagged folds alike — a structural property of fitting 6 states on 18 features with 30m-granularity row counts, not a one-off bug. **Not selected for production at any timeframe tested.**

## 13. 5m

The original production base timeframe. Walk-forward audited in the final 3-way comparison: OOS transitions/day 21.14 (mean), and — the most important finding — **39 of 101 folds (38.6%) failed to characterize any state as Uptrend or Downtrend at all**, with all 4 states reading as "Ranging / {Low,Mid,High}-Vol" variants. This is concentrated almost entirely in 2019-2021 (32 of 34 possible folds in that span), plus scattered 2023-2024 folds. Root-caused (not just observed): flagged folds are essentially bimodal in `trend_score_range` (0.18-0.26 vs. 0.67-0.92 for healthy folds, zero overlap), flip on/off fold-to-fold rather than degrading gradually, and have **equal-or-better** training likelihood than healthy folds — ruling out an optimizer failure. This is a genuine multimodal likelihood surface: two comparably-good 4-state solutions exist for 5m/18-feature data in these historical windows, and which one 5 random restarts happen to surface is seed-dependent. **This was never visible in any full-history screening**, because pooling all 9 years of data always finds *some* directional stretch somewhere.

## 14. 15m

The selected production base timeframe. Zero labeling-failure folds across all 101 OOS folds — the only one of the four configurations tested with this property. Lowest transitions/day of any configuration, every single year 2018-2026. See `README.md` §10 for the full comparison table.

## 15. 30m

Tested at both HMM-4 and HMM-6. 30m-HMM4: worst label-reliability of the four configurations (mean 3.27/4 distinct labels observed OOS, worse even than 5m's 3.46/4), sharing the same "no trend state found" failure mode as 5m (14/101 folds, vs 5m's 39/101). 30m-HMM6: best raw state separation (`trend_score_range` 1.76 vs 15m's 0.75) but the restart-instability problem described in §12. **Neither 30m configuration was selected for production.**

## 16. Rolling vs Expanding

The foundational training-window decision, made early (on the 5m baseline) and never revisited — every later walk-forward in this project's history (including the final 15m/30m validation) uses expanding, not rolling, windows. 100-fold comparison: expanding beat rolling in 91/100 folds for HMM-4 (paired t-test p ≈ 3×10⁻²⁷) and 96/100 for HMM-6 (p ≈ 3×10⁻²⁹), with the advantage *growing* over time (first-half mean diff 2.83 → second-half 4.75 for HMM-4). Cost: ~11x more compute for expanding, judged worth it for a system retrained periodically rather than continuously. Full report: `research/walk_forward/rolling_vs_expanding_final_report.md`.

## 17. Gap Handling

A bar is excluded from model fitting/decoding if its own feature lookback window touches a zero-volume bar (native to whatever timeframe is being used — 15m/30m have their own timeframe-native validity mask, not a reuse of 5m-granularity zero-volume flags). `hmmlearn`'s `lengths=` parameter is used everywhere to prevent the EM/Viterbi recursions from ever modeling a transition across an excluded gap — a "segment" is by construction a maximal run of contiguous, correctly-spaced valid bars.

**Two real, disclosed bugs found in the original `gap_aware.py` utility** (retired to `research/abandoned_models/` along with the rest of the legacy 5m pipeline, never fixed in place per this project's "don't modify without explicit authorization" convention, but never used by the current 15m production code either, since that code has its own corrected local equivalents):
1. `build_validity_mask(raw_df, feature_index, max_lookback_bars=48)` accepts `max_lookback_bars` as a parameter but its body hardcodes `.rolling(48, ...)` — the parameter is dead.
2. `segment_lengths(index)` hardcodes `BAR_INTERVAL = pd.Timedelta(minutes=5)` at module level — calling it on 15m/30m data would misdetect every row as starting a new segment, since real 15m/30m spacing never equals 5 minutes.

## 18. Zero-Volume Analysis

2,183 zero-volume 5m bars across the full history (0.23% of all bars) — cross-checked against Binance's own historical klines API in earlier work and confirmed genuine (periods of literally no trades), not a data pipeline defect. These are not deleted; they are the trigger for the gap-aware validity mask described in §17.

## 19. Outlier Analysis

See `README.md` §13 for the headline findings. Additional detail: within-state excess kurtosis was computed for every (state, feature) pair on the old 5m-HMM4 model; State 3 (Ranging/High-Vol) showed the largest deviation from the Gaussian reference (kurtosis 0) by a wide margin, and its fitted variance was measurably inflated relative to a robust (MAD-based) estimate — direct evidence the Gaussian emission assumption is most strained in exactly the state that also absorbs the most extreme price moves. A full timestamp-level audit (`research/outlier_analysis/`) independently reconfirmed the "100% of extreme moves land in State 3" finding and additionally established that most of State 3's own time is *not* spent in outlier conditions (75.6% of its bars are not themselves >4σ outliers), and that outlier bars do not show a dramatically elevated transition rate relative to normal bars (1.2x, not several-fold) — evidence against "fat tails are a primary driver of excessive regime switching."

## 20. Directional Inversion Audit

A forensic, no-retraining audit of the completed 15m-HMM4 walk-forward triggered by a surprising finding: across all three models tested (5m/15m/30m), OOS bars labeled "Uptrend" showed a *higher* probability of a subsequent *negative* return at 1-2h horizons than "Downtrend" bars did (P(+) 0.47-0.49 for Uptrend vs 0.53-0.58 for Downtrend). The audit verified, independently, that this was **not** a calculation bug: the forward-return formula (`ln(close[t+h]/close[t])`) was independently recomputed and matched the original to floating-point precision; contemporaneous returns confirmed "Uptrend"/"Downtrend" labels are correctly signed (Uptrend bars really did just move up, on average +0.46% over the prior hour; Downtrend really did just move down, -0.48%). The conclusion: this is genuine short-horizon **mean reversion** after a real, correctly-identified trend — not a labeling defect. No code was changed as a result of this finding; it's reported as a real property of the data.

## 21. High-Vol Directional Split Audit

See `README.md` §12 for the full summary. This audit used the same no-retraining, causal-scaling-reproduction methodology as §20, applied specifically to the "Ranging / High-Vol" state, to test whether it hides two directional sub-populations. The result was genuinely mixed: strong descriptive/contextual evidence for internal heterogeneity, weak and unstable predictive evidence. The project's own decision framework (economically meaningful + persistent across years + survives a control check) was not satisfied, so the state was kept unsplit.

## 22. Every Major Experiment — Full List

See `research/README.md` for the complete folder-by-folder index with status labels, and `README.md` §17 for the condensed hypothesis → result → decision table. This document's §8-21 above cover the substance of each.

## 23. Rejected Approaches

- Random Forest (and its 7-config regularized variants) as the persistence-gate model — rejected for a recency reversal not fixed by regularization (§9).
- HMM-6 at every base timeframe tested — rejected for label collapse and/or restart instability (§12).
- 30-minute base timeframe (both HMM-4 and HMM-6) — rejected for worse OOS reliability than 15m (§15).
- 5-minute base timeframe (the original production choice) — rejected after its own 39%-of-folds trend-state-collapse problem was discovered (§13).
- Splitting "Ranging / High-Vol" into directional sub-states — rejected for weak, unstable forward-return evidence (§21).
- The persistence gate itself (LR and GBT variants both) — not rejected on the merits (both showed a real, if weak, signal), but explicitly **terminated** as out of scope for the current production release.

## 24. Final Architecture

```
production/hmm_15m/
    features_15m.py            15m candle aggregation + 18-feature engineering
    regime_labels.py           state -> label characterization (±0.3 threshold convention)
    train_production_model.py  full-history HMM-4 fit -> model/production_model_15m_hmm4.pkl
    predict.py                 live inference: fetch -> features -> scale -> decode -> label
    binance_fetch.py           public Binance klines fetch (no API key required)
    model/
        production_model_15m_hmm4.pkl
```

No file in `production/` imports from `research/` or from the old `abandoned_models/` pipeline — verified by direct import testing (see §28).

## 25. Production Files

| File | Role |
|---|---|
| `production/hmm_15m/features_15m.py` | Feature engineering, extracted verbatim from the validated research pipeline |
| `production/hmm_15m/regime_labels.py` | State characterization |
| `production/hmm_15m/train_production_model.py` | Retraining entry point |
| `production/hmm_15m/predict.py` | Live inference entry point |
| `production/hmm_15m/model/production_model_15m_hmm4.pkl` | The fitted artifact (scaler + HMM + metadata) |

## 26. Known Limitations

1. **Gaussian emission assumption is measurably imperfect** in the high-volatility state (§19) — bounded, not fixed, not blocking.
2. **The 4-state characterization can land on a degenerate (all-Ranging) solution** in some historical training windows due to genuine likelihood-surface multimodality (§13, discovered for 5m but the underlying mechanism — 5 restarts on a genuinely multimodal surface — is architecture-level, not timeframe-specific, and hasn't been separately stress-tested for 15m beyond the fact that it didn't occur in any of the 101 observed folds).
3. **No non-stationarity correction for volatility's own long-run baseline.** Log returns handle price-level non-stationarity; two features (`vol_zscore`, `ofi_zscore`) have local rolling-window adaptivity; the raw volatility features (`vol_30m/1h/2h/4h`) and the model's overall scaler do not adapt to a changing long-run volatility regime between retrains — the primary mitigation is periodic retraining, not a continuous statistical correction.
4. **The persistence signal, where it was measured, was weak** (AUC ~0.52-0.53) — even if revisited later, this is not close to a strong predictive edge on its own.
5. **High-Vol's internal directional heterogeneity is real but not currently actionable** — descriptive, not predictive, per §21.

## 27. Future Research Ideas (not started)

- Re-run the 5m no-trend-state collapse investigation with a much larger restart count (e.g. 20 seeds) to determine whether the degenerate solution is a near-50/50 coin flip or a clear minority outcome that more restarts would reliably avoid.
- Investigate a Student-t (or other fat-tailed) emission distribution for the high-vol state specifically, given the confirmed kurtosis/variance-inflation finding.
- Revisit the persistence gate with 15m as the base (a first cut exists: `research/persistence/` includes a 15m-adapted dataset/training script) if the terminated status is ever reconsidered.
- A volatility-regime-aware or periodically-recalibrated scaler, to address limitation #3.

## 28. Verification Performed Before This Documentation Was Written

- `production/hmm_15m/features_15m.py` and `regime_labels.py` import cleanly with no dependency on `research/` or `research_archive/`.
- `production/hmm_15m/model/production_model_15m_hmm4.pkl` loads, and its `feature_columns` match exactly what `features_15m.build_feature_table()` produces (schema assertion, not just "should match").
- Full offline decode (309,721 valid 15m bars) runs against the loaded model and reproduces the exact 4-state label mapping documented in `README.md` §11.
- Live end-to-end `predict.py` run against real, current Binance data succeeded (fetch → aggregate → feature → scale → decode → label), confirming the production path works outside of a cached/offline context.
