"""
v15_fast_causal_decode.py -- EXPERIMENTAL speed optimization for the A/B/C
extension pass. NOT used for the 11-fold pilot itself (already running with
the original, trusted per-row windowed method).

WHY THIS IS SAFE: the project's own existing leakage audit already
establishes that a causal read at position t (via forward-backward over a
window ENDING at t) is exactly determined by observations up to and
including t, never anything after. The SAME property holds for the Viterbi
recursion's forward (delta) pass: delta_t(s) = max_s'[delta_{t-1}(s') *
A[s',s]] * B_s(o_t) depends ONLY on o_1..o_t. This means BOTH the causal
posterior AND the causal Viterbi state at every position t can be read off
a SINGLE continuous forward pass over a whole stretch, instead of
recomputing forward-backward/Viterbi from scratch in an overlapping window
for every single row (the current O(rows x 500) design).

CRITICAL DISTINCTION (do not confuse with an unsafe shortcut): this is NOT
the same as calling model.predict()/predict_proba() ONCE on a big chunk and
reading every row of the result -- that WOULD be non-causal at intermediate
rows (hmmlearn's own predict()/predict_proba() do full backward smoothing /
global backtracking using the WHOLE given chunk, so only the chunk's own
LAST row is safe -- exactly why the existing per-row windowed design calls
them separately for every row). This module instead implements the
forward-only recursion itself (reusing hmmlearn's own tested
model._compute_log_likelihood for emissions, model.startprob_/transmat_ for
the recursion -- not reimplementing the Gaussian math), and reads
argmax_s(delta_t) / normalized alpha_t at EVERY t from that single forward
pass -- mathematically the terminal value of what a fresh windowed call
ending at t would have produced.

MUST BE VALIDATED against the existing trusted per-row method before use
-- see validate() below, run standalone first.
"""

import numpy as np
from scipy.special import logsumexp


def forward_only_pass(model, scaled_X: np.ndarray, seg_id: np.ndarray):
    """One continuous pass. Resets the recursion (fresh startprob_) at each
    gap-aware segment boundary, mirroring the windowed version's own
    segment-respecting behavior. Returns:
      log_alpha (n, k): forward (sum) lattice -- normalize row t for the
        causal posterior at t (identical to forward-backward's terminal row
        for a window ending at t).
      log_delta (n, k): Viterbi forward (max) lattice -- argmax of row t is
        the causal Viterbi state at t (terminal state of the best path
        ending at t, NOT a full-sequence backtrack).
    """
    n, k = scaled_X.shape[0], model.n_components
    log_frameprob = model._compute_log_likelihood(scaled_X)  # (n, k) -- hmmlearn's own emission calc
    log_startprob = np.log(model.startprob_)
    log_transmat = np.log(model.transmat_)

    log_alpha = np.empty((n, k))
    log_delta = np.empty((n, k))

    for t in range(n):
        if t == 0 or seg_id[t] != seg_id[t - 1]:
            log_alpha[t] = log_startprob + log_frameprob[t]
            log_delta[t] = log_startprob + log_frameprob[t]
        else:
            m = log_alpha[t - 1][:, None] + log_transmat        # (k, k): from-state x to-state
            log_alpha[t] = logsumexp(m, axis=0) + log_frameprob[t]
            mv = log_delta[t - 1][:, None] + log_transmat
            log_delta[t] = mv.max(axis=0) + log_frameprob[t]

    return log_alpha, log_delta


def causal_reads_from_pass(log_alpha, log_delta):
    """Vectorized extraction of state/confidence/margin/posteriors for every
    row from the full forward pass -- no further loop needed."""
    viterbi_states = np.argmax(log_delta, axis=1)
    row_max = log_alpha.max(axis=1, keepdims=True)
    posteriors = np.exp(log_alpha - row_max)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
    confidence = posteriors[np.arange(len(viterbi_states)), viterbi_states]
    sorted_desc = -np.sort(-posteriors, axis=1)
    margin = sorted_desc[:, 0] - sorted_desc[:, 1]
    return viterbi_states, posteriors, confidence, margin


def validate(model, scaler, valid_df, seg_id, valid_index, sample_positions, context_bars, causal_decode_at_fn):
    """Runs BOTH methods on a small sample of real positions and reports
    agreement -- must be checked before trusting the fast path for the
    real experiment."""
    decode_upper_pos = int(max(sample_positions)) + 1
    scaled_prefix = scaler.transform(valid_df.values[:decode_upper_pos])

    log_alpha, log_delta = forward_only_pass(model, scaled_prefix[:decode_upper_pos], seg_id[:decode_upper_pos])
    fast_states, fast_post, fast_conf, fast_margin = causal_reads_from_pass(log_alpha, log_delta)

    mismatches = []
    for p in sample_positions:
        old = causal_decode_at_fn(scaled_prefix, seg_id, p, model, context_bars)
        state_match = int(old["state"]) == int(fast_states[p])
        conf_diff = abs(old["confidence"] - fast_conf[p])
        margin_diff = abs(old["margin"] - fast_margin[p])
        if not state_match or conf_diff > 1e-3 or margin_diff > 1e-3:
            mismatches.append({"pos": int(p), "old_state": int(old["state"]), "fast_state": int(fast_states[p]),
                                "old_conf": old["confidence"], "fast_conf": float(fast_conf[p]),
                                "conf_diff": float(conf_diff), "margin_diff": float(margin_diff)})
    return mismatches, len(sample_positions)
