import pandas as pd
from typing import Dict, Tuple, List, Optional
from config import OUTCOMES, LIVE_BLEND_THRESHOLD, SMOOTH_ALPHA, MIN_MATCHES

TARGET_OUTCOMES = [o for o in OUTCOMES if o != "unknown"]

# -----------------------------
# Helpers
# -----------------------------
def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = "UNK"
        out[c] = out[c].fillna("UNK")
    if "hurry_up" in cols:
        out["hurry_up"] = out["hurry_up"].fillna(False).astype(bool)
    return out

def _laplace_probs(counts: Dict[str, int], alpha: float) -> Dict[str, float]:
    total = 0.0
    out = {}
    for o in TARGET_OUTCOMES:
        total += counts.get(o, 0) + alpha
    for o in TARGET_OUTCOMES:
        out[o] = (counts.get(o, 0) + alpha) / total if total > 0 else 1.0 / len(TARGET_OUTCOMES)
    return out

def _counts_in_slice(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty:
        return {o: 0 for o in TARGET_OUTCOMES}
    vc = df["outcome"].value_counts()
    return {o: int(vc.get(o, 0)) for o in TARGET_OUTCOMES}

def _blend_probs(hist_probs: Dict[str, float],
                 live_probs: Dict[str, float],
                 live_n: int,
                 threshold: int) -> Dict[str, float]:
    # weight increases as we see more live examples
    w_live = min(1.0, float(live_n) / float(threshold)) if threshold > 0 else 1.0
    w_hist = 1.0 - w_live
    return {o: w_hist * hist_probs.get(o, 0.0) + w_live * live_probs.get(o, 0.0) for o in TARGET_OUTCOMES}

def _filter(df: pd.DataFrame, cond: Dict[str, object]) -> pd.DataFrame:
    out = df
    for k, v in cond.items():
        out = out[out[k] == v]
    return out

# -----------------------------
# Core: one-condition blended probabilities
# -----------------------------
def blended_probs_for_condition(
    cond: Dict[str, object],
    df_hist: pd.DataFrame,
    df_live: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """
    Compute blended empirical probabilities for a condition dict.
    Uses backoff from strict->loose, and blends historical + live with threshold.
    """

    # Labeled only drive empirical outcomes
    df_hist = df_hist[df_hist["outcome"].notna()].copy() if df_hist is not None and not df_hist.empty else pd.DataFrame()
    df_live = df_live[df_live["outcome"].notna()].copy() if df_live is not None and not df_live.empty else pd.DataFrame()

    # Backoff levels (strict -> loose)
    levels = [
        ["pv_possession", "quarter", "clock_bucket", "hurry_up", "down", "dist_bucket", "field_zone",
         "opp_personnel", "opp_formation", "def_shell", "pressure"],
        ["pv_possession", "quarter", "clock_bucket", "hurry_up", "down", "dist_bucket", "field_zone",
         "opp_personnel", "def_shell", "pressure"],
        ["pv_possession", "quarter", "clock_bucket", "hurry_up", "down", "dist_bucket", "field_zone",
         "def_shell", "pressure"],
        ["pv_possession", "down", "dist_bucket", "field_zone"],
    ]

    needed = sorted(set(sum(levels, [])) | {"outcome"})
    df_hist = _ensure_cols(df_hist, needed)
    df_live = _ensure_cols(df_live, needed)

    # normalize condition values
    norm = {}
    for k in needed:
        if k == "outcome":
            continue
        if k == "hurry_up":
            norm[k] = bool(cond.get(k, False))
        else:
            v = cond.get(k, "UNK")
            norm[k] = "UNK" if v is None else v

    used_level = None
    hist_n = live_n = 0
    hist_probs = {o: 1.0 / len(TARGET_OUTCOMES) for o in TARGET_OUTCOMES}
    live_probs = {o: 1.0 / len(TARGET_OUTCOMES) for o in TARGET_OUTCOMES}

    for i, cols in enumerate(levels):
        cond_i = {c: norm.get(c, "UNK") for c in cols}

        hist_slice = _filter(df_hist, cond_i) if not df_hist.empty else pd.DataFrame()
        live_slice = _filter(df_live, cond_i) if not df_live.empty else pd.DataFrame()

        hist_n = len(hist_slice)
        live_n = len(live_slice)

        min_req = MIN_MATCHES[min(i, len(MIN_MATCHES) - 1)]
        if (hist_n + live_n) >= min_req:
            used_level = i
            hist_probs = _laplace_probs(_counts_in_slice(hist_slice), SMOOTH_ALPHA)
            live_probs = _laplace_probs(_counts_in_slice(live_slice), SMOOTH_ALPHA)
            break

    # if none hit, fall back to global priors
    if used_level is None:
        used_level = len(levels)
        hist_n = len(df_hist)
        live_n = len(df_live)
        hist_probs = _laplace_probs(_counts_in_slice(df_hist), SMOOTH_ALPHA)
        live_probs = _laplace_probs(_counts_in_slice(df_live), SMOOTH_ALPHA)

    blended = _blend_probs(hist_probs, live_probs, live_n, LIVE_BLEND_THRESHOLD)

    debug = {
        "used_backoff_level": used_level,
        "hist_matches": hist_n,
        "live_matches": live_n,
        "live_blend_threshold": LIVE_BLEND_THRESHOLD,
    }
    return blended, debug

# -----------------------------
# Convenience: current play (row -> condition)
# -----------------------------
def blended_probs_for_latest_row(
    latest_row: pd.Series,
    df_hist: pd.DataFrame,
    df_live: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, object]]:
    cond = latest_row.to_dict()
    return blended_probs_for_condition(cond, df_hist, df_live)

# -----------------------------
# Build tables by bucket
# -----------------------------
def table_by_clock_bucket(
    base_cond: Dict[str, object],
    df_hist: pd.DataFrame,
    df_live: pd.DataFrame,
    clock_buckets: List[str]
) -> pd.DataFrame:
    """
    Returns a dataframe with one row per clock_bucket, showing blended probs
    that update as live labeled outcomes accumulate.
    """
    rows = []
    for cb in clock_buckets:
        cond = dict(base_cond)
        cond["clock_bucket"] = cb
        probs, dbg = blended_probs_for_condition(cond, df_hist, df_live)
        row = {"clock_bucket": cb, **{f"p_{k}": probs[k] for k in probs},
               "hist_n": dbg["hist_matches"], "live_n": dbg["live_matches"], "backoff": dbg["used_backoff_level"]}
        rows.append(row)
    return pd.DataFrame(rows)

def table_for_current_situation_variants(
    base_cond: Dict[str, object],
    variants: List[Dict[str, object]],
    df_hist: pd.DataFrame,
    df_live: pd.DataFrame,
    label_col: str = "label"
) -> pd.DataFrame:
    """
    Build a table for multiple variant conditions (e.g. different zones, distances).
    Each variant dict can include label_col for display.
    """
    rows = []
    for v in variants:
        cond = dict(base_cond)
        cond.update({k: val for k, val in v.items() if k != label_col})
        probs, dbg = blended_probs_for_condition(cond, df_hist, df_live)
        label = v.get(label_col, "VAR")
        row = {label_col: label, **{f"p_{k}": probs[k] for k in probs},
               "hist_n": dbg["hist_matches"], "live_n": dbg["live_matches"], "backoff": dbg["used_backoff_level"]}
        rows.append(row)
    return pd.DataFrame(rows)

