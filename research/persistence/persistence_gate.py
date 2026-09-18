"""
persistence_gate.py -- live inference for the persistence module: loads
Data/persistence_gate.pkl (trained by train_persistence_gate.py) and
Data/production_model_hmm4.pkl (via infer_regime.decode_current_regime),
computes the same 4 features used at training time (confidence,
stay-probability, duration, posterior margin) for the CURRENT live bar, and
reports P(trend persists for the next ~2h / 24 bars).

Only meaningful for a genuinely trending state (Uptrend/Downtrend) -- the
gate was trained exclusively on those (Ranging states were excluded, see
persistence_dataset.py), so a live call during a Ranging regime returns
p_persist=None with an explicit reason rather than an out-of-distribution
guess.

Reuses infer_regime.decode_current_regime() for the fetch->features->mask->
scale->predict/predict_proba pipeline -- no duplicated plumbing here.

Run: `python persistence_gate.py [N_STATES] [CONTEXT_BARS]` (from src/).
Defaults match infer_regime.py: N_STATES=4, CONTEXT_BARS=2000.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import infer_regime
import persistence_common

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_PATH = os.path.join(_ROOT, "Data", "persistence_gate.pkl")

N_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CONTEXT_BARS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000


def load_gate_artifact() -> dict:
    if not os.path.exists(GATE_PATH):
        raise FileNotFoundError(f"No persistence gate found at {GATE_PATH} -- run train_persistence_gate.py first")
    with open(GATE_PATH, "rb") as f:
        return pickle.load(f)


def compute_live_gate_features(decode: dict) -> dict:
    state = decode["current_state"]
    stay_prob = float(decode["model"].transmat_[state, state])
    duration_raw = int(persistence_common.compute_running_duration(decode["states"], decode["lengths"])[-1])
    margin = float(persistence_common.compute_margin(decode["posteriors"])[-1])
    left_censored = duration_raw == decode["lengths"][-1]
    return {
        "confidence": decode["confidence"],
        "stay_prob": stay_prob,
        "duration_raw": duration_raw,
        "log1p_duration": float(persistence_common.duration_feature(np.array([duration_raw]))[0]),
        "margin": margin,
        "duration_left_censored": left_censored,
    }


def predict_persistence(n_states: int = N_STATES, context_bars: int = CONTEXT_BARS):
    """Returns None if the current bar is unavailable (same condition as
    infer_regime.decode_current_regime). Otherwise returns a dict; if the
    current state is Ranging (not trending), p_persist is None and
    `reason` explains why -- the gate was never trained on Ranging bars."""
    decode = infer_regime.decode_current_regime(n_states, context_bars)
    if decode is None:
        return None

    is_trending = bool(persistence_common.state_is_trending(decode["means_df"])[decode["current_state"]])
    direction_sign = int(persistence_common.state_trend_signs(decode["means_df"])[decode["current_state"]])
    feats = compute_live_gate_features(decode)

    result = {
        "last_bar_ts": decode["last_bar_ts"], "label": decode["label"],
        "is_trending": is_trending, "direction_sign": direction_sign,
        "p_persist": None, "reason": None, **feats,
    }

    if not is_trending:
        result["reason"] = "current regime is Ranging -- gate was trained only on Uptrend/Downtrend bars, no directional call to make"
        return result

    gate = load_gate_artifact()
    row = pd.DataFrame([[feats["confidence"], feats["stay_prob"], feats["log1p_duration"], feats["margin"]]],
                        columns=gate["feature_columns"])
    scaled = gate["scaler"].transform(row.values)
    result["p_persist"] = float(gate["model"].predict_proba(scaled)[0, 1])
    result["gate_holdout_auc"] = gate.get("holdout_auc")
    return result


def main():
    result = predict_persistence(N_STATES, CONTEXT_BARS)
    if result is None:
        return  # decode_current_regime already printed [UNAVAILABLE]

    print(f"\n{'='*60}")
    print(f"PERSISTENCE GATE as of {result['last_bar_ts']}")
    print(f"{'='*60}")
    print(f"  Regime: {result['label']}")
    if result["p_persist"] is None:
        print(f"  P(trend persists): N/A -- {result['reason']}")
    else:
        print(f"  P(trend persists ~2h): {result['p_persist']:.1%}  "
              f"(gate holdout AUC={result['gate_holdout_auc']:.3f} -- a soft signal, not a strong filter)")
    print(f"  Raw features: confidence={result['confidence']:.3f}  stay_prob={result['stay_prob']:.3f}  "
          f"duration={result['duration_raw']}  margin={result['margin']:.3f}")
    if result["duration_left_censored"]:
        print(f"  NOTE: duration is left-censored (no state change observed anywhere in the "
              f"{CONTEXT_BARS}-bar context window) -- the true run may have started earlier.")


if __name__ == "__main__":
    main()
