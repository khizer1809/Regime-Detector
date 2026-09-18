# Research Archive — Index

This is the complete research history behind the production 15m-HMM4 regime classifier. It exists so that every production decision can be traced back to the experiment that motivated it — nothing here was deleted for being "unsuccessful"; failed and rejected approaches are preserved and clearly labeled.

For the full narrative (what we did, why, what happened), see [`../docs/PROJECT_DOCUMENTATION.md`](../docs/PROJECT_DOCUMENTATION.md). This file is a directory index.

| Folder | Purpose | Status |
|---|---|---|
| `data_validation/` | (see `outlier_analysis/`, `abandoned_models/gap_aware.py`) — zero-volume/gap verification | Findings folded into gap-aware masking used throughout |
| `feature_engineering/` | Early feature-family experiments (returns, OFI stationarity, expanded/structure features, trend features) — v7-v10, v13, v15 | Superseded by the 18-feature production set |
| `scaling/` | (see `abandoned_models/scaling.py`) — train-only fold scaling utility | Superseded by direct `StandardScaler` in production |
| `model_selection/` | HMM-4 vs HMM-6 selection, rolling-vs-expanding window strategy | Expanding window: **adopted**. HMM-4 vs HMM-6: superseded by timeframe research |
| `hmm/` | (see `walk_forward/`, `timeframe_comparison/`) | — |
| `walk_forward/` | V2 walk-forward persistence framework, the final 3-way (5m/15m/30m) and 30m-HMM4 101-fold OOS walk-forward validations | **Production decision source** |
| `regime_analysis/` | State-transition and segment-breakdown diagnostics | Descriptive, informed labeling/QA |
| `timeframe_comparison/` | Fast full-history screening (5m/15m/30m × HMM4/HMM6) and the multi-timeframe (MTF) EMA-context experiment | 15m identified as strongest candidate here, confirmed in `walk_forward/` |
| `outlier_analysis/` | Fat-tail/kurtosis investigation, exact outlier-timestamp audit | State 3 (High-Vol) absorbs 100% of extreme moves; not a labeling bug |
| `directional_analysis/` | High-Vol causal directional-split hypothesis test | **REJECTED FOR PRODUCTION** — see below |
| `ml_comparisons/` | Random Forest vs Logistic Regression (+ regularization), nonlinear capacity test, raw-vs-HMM-features test | **RF REJECTED**, LR-based persistence retained only as a research artifact |
| `persistence/` | P(trend persists) gate: dataset construction, LR/GBT training, threshold-calibration experiments (v3, v4, v11) | **TERMINATED / NOT PART OF CURRENT SYSTEM** |
| `abandoned_models/` | Retired 5m production pipeline (features, HMM, scaler, live inference) | Superseded by 15m-HMM4; kept for reference |
| `reports/` | Session transcripts and narrative write-ups | Historical record |
| `visualizations/` | All chart/plot outputs referenced from the README | Real experiment outputs, not fabricated |

## Key findings by area

### `walk_forward/` — the production decision
101-fold expanding-window walk-forward, 2018-03 → 2026-07-18, 4-hour embargo, train-only scaler/HMM/labeling throughout. Compared 5m-HMM4, 15m-HMM4, 30m-HMM4, 30m-HMM6.

- **15m-HMM4**: lowest transitions/day of any config, every single year 2018-2026. Zero labeling-failure folds (101/101 always found all 4 regimes).
- **5m-HMM4** (previous production): failed to find any directional (Uptrend/Downtrend) state in 39/101 folds, concentrated in 2019-2021 — a real, previously-unmeasured reliability gap in the old baseline.
- **30m-HMM6**: best raw state separation but chronic restart instability (normalized restart spread ~0.25, vs 15m's ~0.0006) and 20/101 folds losing states.
- **30m-HMM4**: worst label reliability of all four (3.27/4 distinct labels on average).

→ **15m-HMM4 selected for production.**

### `directional_analysis/` — High-Vol split investigation
Tested whether "Ranging / High-Vol" hides two directional populations. Result: yes, descriptively (88% of bars are individually directional by the existing ±0.3 threshold; strong asymmetric transition context) — but the forward-return separation between the two halves was economically tiny (Cohen's d ≈ -0.07), barely significant only at the 4h horizon, and flipped sign in 2 of 9 years. **Decision: keep "Ranging / High-Vol" as one regime.**

### `ml_comparisons/` — RF vs LR
Random Forest tested as an alternative to Logistic Regression for the (now-terminated) persistence gate, on identical HMM-derived features. Base LR AUC 0.5203 vs base RF AUC 0.5138; with 15 added market-structure features, struct-RF reached 0.5232 — but a stability check showed RF's train-vs-OOS gap of +0.178 and losses to LR in 2025/2026 (recency reversal). A 7-config regularization sweep (`rf_regularization/`) improved the gap somewhat (best: RF-5, AUC 0.5248) but did not fix the recency-reversal problem. **RF rejected**; LR remained the stronger, simpler choice throughout — moot now that persistence itself is terminated.

### `outlier_analysis/` — fat tails
100% of the top 0.1%/0.5%/1.0% most extreme 5m returns fall in State 3 (Ranging/High-Vol) of the old 5m-HMM4. Deeper audit: only 24.4% of State 3's own bars are themselves outliers, and outlier bars show only a modest 1.2x higher transition-coincidence rate than normal bars — State 3 is a genuine (if noisy) high-vol regime that also absorbs extremes, not purely an outlier bucket, and fat tails do not appear to be a primary driver of excessive switching.

### `persistence/` — TERMINATED
A downstream P(trend persists for ~2h) gate built on top of the HMM's own confidence/stay-probability/duration/margin. Reached holdout AUC ≈ 0.525 (5m, GBT) and ≈ 0.532 (15m, GBT) — a real but weak signal. **Explicitly terminated for the current system** per project direction; not part of the active production architecture, not imported by it, and not a hidden dependency. Kept here as a complete, reproducible research trail should it be revisited.
