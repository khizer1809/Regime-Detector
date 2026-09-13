# Regime Detector

A Gaussian HMM regime detector for BTCUSDT (5-minute bars): classifies the market into a small number of hidden "regimes" (e.g. uptrend, downtrend, ranging / low-vol, ranging / high-vol) from returns, volatility, volume/order-flow, and skew features.

## Key finding

A 100-fold walk-forward experiment compared two training-window strategies — a fixed rolling 6-month window vs. an expanding window that trains on all available history — across both a 4-state and 6-state HMM. **Expanding-window wins decisively out-of-sample** (HMM-4: 91/100 folds, HMM-6: 96/100 folds, both far beyond chance), so that's what this repo's production model uses. Full write-up, methodology, and all nine conclusion questions answered against the data: [`Data/rolling_vs_expanding_final_report.md`](Data/rolling_vs_expanding_final_report.md).

## Pipeline

```
Raw Binance 5m candles
        |
        v
features.py            -- 18 features: returns, volatility, volume/order-flow, skew
        |
        v
gap_aware.py            -- masks zero-volume/gap bars (no blanket dropna(); verified
        |                   against Binance's own historical API that these gaps are
        |                   genuine, not data errors -- see the final report)
        v
save_production_model.py -- fits a fresh GaussianHMM on ALL history (expanding window),
        |                    5 random restarts, best kept by training log-likelihood
        v
Data/production_model_hmm{4,6}.pkl
        |
        v
infer_regime.py          -- fetches live data via binance_fetch.py, decodes the
                             current regime with a human-readable label + confidence
```

`refresh_features.py` incrementally updates the local feature file with new bars from Binance without rebuilding history from scratch.

## Usage

All commands run from `src/`.

**Get the current regime right now** (fetches live data from Binance):
```
python infer_regime.py [N_STATES] [CONTEXT_BARS]
# e.g. python infer_regime.py 4 500
```

**Retrain the production model** on all available history:
```
python save_production_model.py [N_STATES]
# e.g. python save_production_model.py 6
```

**Refresh the local feature file** with new bars since it was last built:
```
python refresh_features.py
```

## Files

| File | Purpose |
|---|---|
| `features.py` | Raw OHLCV+order-flow → 18 engineered features |
| `src/gap_aware.py` | Zero-volume/gap validity mask + segment-aware fitting (no `dropna()`) |
| `src/scaling.py` | `StandardScaler` wrapper (fit on train only) |
| `src/save_production_model.py` | Trains and pickles the production HMM (expanding window) |
| `src/binance_fetch.py` | Live 5m candle fetch from Binance's public API |
| `src/refresh_features.py` | Incrementally appends new bars to the feature file |
| `src/infer_regime.py` | Loads a saved model + live data → current regime |
| `Data/production_model_hmm4.pkl` / `hmm6.pkl` | Trained production models (4-state / 6-state) |
| `Data/rolling_vs_expanding_final_report.md` | Full experiment write-up behind the training-strategy choice |

`Data/features_out.csv` and `Data/features_out_masked.csv` (the full historical feature matrices, ~785MB combined) are gitignored — regenerate locally via `refresh_features.py` starting from an existing copy, or the original feature-building pipeline.

## Known gaps / not yet production-hardened

- No scheduler — `refresh_features.py` / `infer_regime.py` are run manually, not on a cron/service.
- No crash-safe (atomic) write for the feature-file append.
- No automated tests currently in the repo.
- `binance_fetch.py`'s `avg_price`/`buy_vol`/`sell_vol`/`ofi` are proxies derived from Binance's public kline fields (the original raw data's exact methodology for those columns is undocumented/unavailable) — `open`/`high`/`low`/`close`/`volume` are exact.
- Gap detection is zero-volume-based; a genuinely *missing* (not zero-volume) candle from the live feed isn't separately reindexed/caught yet.
