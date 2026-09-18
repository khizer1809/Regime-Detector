"""
Live regime prediction: fetch recent BTCUSDT 5m data -> build 15m candles/
features -> transform with the production scaler -> decode with the
production HMM-4 -> characterize the current state.

Run: `python predict.py [context_bars]` (from production/hmm_15m/).
"""

import os
import pickle
import sys

import pandas as pd

from binance_fetch import fetch_recent
from features_15m import (MAX_LOOKBACK_BARS, TF_MINUTES, build_15m_candles, build_features,
                           build_validity_mask, segment_lengths)
from regime_labels import characterize_state

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model", "production_model_15m_hmm4.pkl")

CONTEXT_BARS = int(sys.argv[1]) if len(sys.argv) > 1 else 500  # 15m bars of decode context


def load_artifact() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No production model at {MODEL_PATH} -- run train_production_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_current_regime(context_bars: int = CONTEXT_BARS):
    artifact = load_artifact()
    model, scaler, feature_cols = artifact["model"], artifact["scaler"], artifact["feature_columns"]

    n_5m_bars = context_bars * (TF_MINUTES // 5) + MAX_LOOKBACK_BARS * (TF_MINUTES // 5) + 50
    raw5m = fetch_recent(n_5m_bars)

    agg = build_15m_candles(raw5m)
    feats = build_features(agg)[feature_cols]
    valid = build_validity_mask(agg, feats.index) & feats.notna().all(axis=1)

    context = feats.iloc[-context_bars:] if len(feats) > context_bars else feats
    context_valid_flag = valid.reindex(context.index)

    last_bar_ts = context.index[-1]
    if not bool(context_valid_flag.iloc[-1]):
        print(f"[UNAVAILABLE] Most recent 15m bar ({last_bar_ts}) is invalid "
              f"(gap/zero-volume window or NaN feature) -- no regime can be reported.")
        return None

    valid_context = context[context_valid_flag]
    lengths = segment_lengths(valid_context.index)
    scaled = scaler.transform(valid_context.values)

    states = model.predict(scaled, lengths=lengths)
    posteriors = model.predict_proba(scaled, lengths=lengths)
    current_state = int(states[-1])
    confidence = float(posteriors[-1, current_state])

    means_df = pd.DataFrame(model.means_, columns=feature_cols)
    label = characterize_state(means_df.iloc[current_state])

    return {
        "timestamp": last_bar_ts, "state": current_state, "n_states": model.n_components,
        "regime": label, "confidence": confidence, "means_df": means_df,
    }


def main():
    result = predict_current_regime()
    if result is None:
        return
    print(f"\nCurrent 15m regime as of {result['timestamp']}")
    print(f"  State {result['state']} of {result['n_states']}: {result['regime']}")
    print(f"  Posterior confidence: {result['confidence']:.1%}")
    print("\nAll states, for reference:")
    for s in range(result["n_states"]):
        marker = " <- current" if s == result["state"] else ""
        print(f"  state {s}: {characterize_state(result['means_df'].iloc[s])}{marker}")


if __name__ == "__main__":
    main()
