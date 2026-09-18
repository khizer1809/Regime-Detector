"""
State -> regime label characterization for the 15m-HMM4 production model.

A state's label is derived from its FITTED (training-only) mean feature
vector, never from live/test data -- see production/hmm_15m/README.md for
why this matters. Thresholds are the same +/-0.3 z-score convention used
throughout this project's research.
"""

import pandas as pd

from features_15m import RETURN_COLS, VOL_COLS

TREND_Z_THRESHOLD = 0.3
VOL_Z_THRESHOLD = 0.3


def characterize_state(state_mean: pd.Series) -> str:
    trend_score = float(state_mean[RETURN_COLS].mean())
    vol_score = float(state_mean[VOL_COLS].mean())

    if trend_score > TREND_Z_THRESHOLD:
        trend = "Uptrend"
    elif trend_score < -TREND_Z_THRESHOLD:
        trend = "Downtrend"
    else:
        trend = "Ranging"

    if vol_score > VOL_Z_THRESHOLD:
        vol = "High-Vol"
    elif vol_score < -VOL_Z_THRESHOLD:
        vol = "Low-Vol"
    else:
        vol = "Mid-Vol"

    return f"{trend} / {vol}"
