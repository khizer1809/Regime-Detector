
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_regime import RETURN_COLS, TREND_Z_THRESHOLD

LABEL_HORIZON_BARS = 24  # ~2h, matches ret_2h's own trailing window exactly


##This Function helps to get the segement length count
def segment_ids_from_lengths(lengths: list) -> np.ndarray:
    """Which contiguous gap_aware segment each row (in valid-only,
    chronological order) belongs to, e.g. lengths=[3,2] -> [0,0,0,1,1]."""
    return np.repeat(np.arange(len(lengths)), lengths)

# Goal : This function helps to compute the duration of the current regime, which is the number of consecutive bars in the same decoded state, backward-looking. It resets on a state change or a segment/gap boundary.
def compute_running_duration(states, lengths):
    seg_id = segment_ids_from_lengths(lengths)

    duration = []
    count = 0

    for i in range(len(states)):#here states meanns total candles decoded till yet through the HMM in all the 4 states 
        # Start a new run if:
        # 1. first candle
        # 2. HMM state changed
        # 3. segment changed (gap)
        if i == 0  or states[i] != states[i - 1] or seg_id[i] != seg_id[i - 1]: # If this is the first candle, OR the HMM state changed, OR we crossed a data-gap boundary → start a new duration at 1. Otherwise → keep counting
            count = 1
        else:
            count += 1

        duration.append(count)

    return np.array(duration)

# Goal : → Gets the probability that the current HMM state will remain the same on the next candle.
def compute_stay_prob(states: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    return transmat[states, states]

# Here we get the prob of the candle belonging to the state 
def compute_confidence(posteriors: np.ndarray, states: np.ndarray) -> np.ndarray:
    idx = np.arange(len(states))
    return posteriors[idx, states]


def compute_margin(posteriors: np.ndarray) -> np.ndarray:
    sorted_desc = -np.sort(-posteriors, axis=1)
    return sorted_desc[:, 0] - sorted_desc[:, 1]

# it returns the avg of each state in respect to the features of the state 
def state_trend_scores(means_df: pd.DataFrame) -> np.ndarray:
    return_cols = [c for c in RETURN_COLS if c in means_df.columns]
    return means_df[return_cols].mean(axis=1).values

# We get the trend sign : +1 for uptrend , -1 for downtrend, 0 for ranging
def state_trend_signs(means_df: pd.DataFrame) -> np.ndarray:
    return np.sign(state_trend_scores(means_df))

#A function taht helps to calculate the trend strength
def state_is_trending(means_df: pd.DataFrame) -> np.ndarray:
    return np.abs(state_trend_scores(means_df)) > TREND_Z_THRESHOLD

#→ Converts the raw state duration using log(1 + duration) to reduce the effect of very large durations.
def duration_feature(duration_raw: np.ndarray) -> np.ndarray:
    return np.log1p(duration_raw)
