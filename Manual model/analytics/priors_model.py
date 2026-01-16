from typing import Dict
from config import (
    CALL_TYPES,
    get_base_alpha,
    ZONE_MULT, CLOCK_MULT, HURRY_MULT,
    GOAL_TO_GO_MULT, AFTER_FIRST_DOWN_MULT,
    PRESSURE_PRIOR, TIMEOUT_PRIOR,
    FOURTH_TRI_CFB, FOURTH_TRI_NFL,
)

OFFENSE_KEYS = ["RUN", "PASS_QUICK", "PASS_DROPBACK", "PLAY_ACTION", "SCREEN", "SHOT", "SACK", "PENALTY"]

def _apply_mult(alpha: Dict[str, float], mult: Dict[str, float]) -> Dict[str, float]:
    out = dict(alpha)
    for k, m in (mult or {}).items():
        if k in out:
            out[k] = out[k] * float(m)
    return out

def posterior_mean(prior_alpha: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    denom = 0.0
    num = {}
    for k in prior_alpha.keys():
        num[k] = float(prior_alpha.get(k, 0.0)) + float(counts.get(k, 0))
        denom += num[k]
    if denom <= 0:
        n = len(prior_alpha) if len(prior_alpha) else 1
        return {k: 1.0 / n for k in prior_alpha}
    return {k: num[k] / denom for k in prior_alpha}

def counts_from_live(df_labeled, cond: Dict[str, object], label_col: str) -> Dict[str, int]:
    if df_labeled is None or df_labeled.empty:
        return {}
    sub = df_labeled
    for k, v in cond.items():
        if k in sub.columns:
            sub = sub[sub[k] == v]
    vc = sub[label_col].value_counts()
    return {str(k): int(v) for k, v in vc.items()}

def call_prior_alpha(
    down: int,
    dist_bucket: str,
    field_zone: str,
    clock_bucket: str,
    hurry_up: bool,
    league_mix_cfb: float,
    prior_strength: float,
    goal_to_go: bool = False,
    after_first_down: bool = False,
    # NEW: make special teams context aware
    fg_in_range: bool = False,
) -> Dict[str, float]:
    base = get_base_alpha(down, dist_bucket, league_mix_cfb)
    base = _apply_mult(base, ZONE_MULT.get(field_zone, {}))
    base = _apply_mult(base, CLOCK_MULT.get(clock_bucket, {}))
    if hurry_up:
        base = _apply_mult(base, HURRY_MULT)
    if goal_to_go:
        base = _apply_mult(base, GOAL_TO_GO_MULT)
    if after_first_down:
        base = _apply_mult(base, AFTER_FIRST_DOWN_MULT)

    alpha = {k: max(0.0, float(base.get(k, 0.0)) * float(prior_strength)) for k in OFFENSE_KEYS}

    # IMPORTANT: These are NOT always valid next-play calls in your usage.
    # We set them to 0 unless context says they’re possible.
    alpha["KICKOFF"] = 0.0
    alpha["PAT_KICK"] = 0.0
    alpha["TWO_POINT"] = 0.0

    # Punt/FG only become non-zero on 4th down;
    # FG only if in range.
    if int(down) == 4:
        alpha["PUNT"] = 0.7 * float(prior_strength)
        alpha["FIELD_GOAL"] = (0.7 * float(prior_strength)) if fg_in_range else 0.0
    else:
        alpha["PUNT"] = 0.0
        alpha["FIELD_GOAL"] = 0.0

    # Fill missing call types with 0.0
    out = {k: float(alpha.get(k, 0.0)) for k in CALL_TYPES}
    return out

def derived_pass_conditionals(call_probs: Dict[str, float]) -> Dict[str, float]:
    p_run = call_probs.get("RUN", 0.0)
    pass_keys = ["PASS_QUICK", "PASS_DROPBACK", "PLAY_ACTION", "SCREEN", "SHOT"]
    p_pass = sum(call_probs.get(k, 0.0) for k in pass_keys)

    def cond(k: str) -> float:
        return (call_probs.get(k, 0.0) / p_pass) if p_pass > 1e-9 else 0.0

    return {
        "p_run": p_run,
        "p_pass": p_pass,
        "p_shot_given_pass": cond("SHOT"),
        "p_screen_given_pass": cond("SCREEN"),
        "p_pa_given_pass": cond("PLAY_ACTION"),
        "p_quick_given_pass": cond("PASS_QUICK"),
        "p_dropback_given_pass": cond("PASS_DROPBACK"),
    }

# -----------------------------
# Pressure prior alpha
# -----------------------------
def pressure_prior_alpha(down: int, dist_bucket: str, strength: float) -> Dict[str, float]:
    base = PRESSURE_PRIOR.get((int(down), str(dist_bucket)), {"4": 30, "5+": 10})
    return {k: max(0.0, float(v) * float(strength)) for k, v in base.items()}

# -----------------------------
# Timeout prior alpha
# -----------------------------
def timeout_prior_alpha(quarter: int, clock_bucket: str, hurry_up: bool, strength: float) -> Dict[str, float]:
    base = TIMEOUT_PRIOR.get((int(quarter), str(clock_bucket), bool(hurry_up)), {"NO": 36, "YES": 4})
    return {k: max(0.0, float(v) * float(strength)) for k, v in base.items()}

# -----------------------------
# NEW: 4th-down decision prior (GO vs PUNT vs FIELD_GOAL)
# -----------------------------
def fourth_tri_prior(dist_bucket: str, field_zone: str, league_mix_cfb: float, strength: float, fg_in_range: bool) -> Dict[str, float]:
    key = (str(dist_bucket), str(field_zone))
    cfb = FOURTH_TRI_CFB.get(key, {"GO": 6, "FIELD_GOAL": 6, "PUNT": 28})
    nfl = FOURTH_TRI_NFL.get(key, {"GO": 6, "FIELD_GOAL": 8, "PUNT": 26})

    out = {}
    for k in ["GO", "FIELD_GOAL", "PUNT"]:
        out[k] = float(league_mix_cfb) * float(cfb.get(k, 0.0)) + (1.0 - float(league_mix_cfb)) * float(nfl.get(k, 0.0))

    # If not in range, force FG to ~0 (but not negative)
    if not fg_in_range:
        out["FIELD_GOAL"] = 0.0

    return {k: max(0.0, float(v) * float(strength)) for k, v in out.items()}
