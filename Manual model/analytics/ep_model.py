from typing import Dict, Any, Optional

# Base EP by field zone (rough but consistent)
# Interpreted as offense expected points from that zone, roughly “next-drive points”
EP_ZONE_CFB = {
    "BACKED_UP": 0.6,
    "OWN_SIDE": 1.2,
    "MIDFIELD": 2.0,
    "HIGH_RED": 3.4,
    "LOW_RED": 4.8,
    "UNK": 2.0,
}
EP_ZONE_NFL = {
    "BACKED_UP": 0.4,
    "OWN_SIDE": 1.0,
    "MIDFIELD": 1.8,
    "HIGH_RED": 3.2,
    "LOW_RED": 4.6,
    "UNK": 1.8,
}

# Down/dist adjustments (subtract EP as you get behind the sticks)
DIST_ADJ = {
    "SHORT": 0.00,
    "MEDIUM": -0.25,
    "LONG": -0.55,
    "X_LONG": -0.80,
    "UNK": -0.35,
}
DOWN_ADJ = {
    1: 0.00,
    2: -0.15,
    3: -0.45,
    4: -0.80,
}

# Clock bucket “compression” (less time => fewer points)
CLOCK_MULT = {
    "15-10": 1.00,
    "10-7": 1.00,
    "7-6": 0.98,
    "5-3": 0.95,
    "3-2": 0.92,
    "2-0": 0.88,
    "SCRIPT_START": 1.00,
    "OTHER": 1.00,
}

# Zone progression ladder for approximate state transitions based on yards_bucket
ZONE_LADDER = ["BACKED_UP", "OWN_SIDE", "MIDFIELD", "HIGH_RED", "LOW_RED"]

def _blend(a: float, b: float, w_cfb: float) -> float:
    return float(w_cfb) * float(a) + (1.0 - float(w_cfb)) * float(b)

def ep_pre(state: Dict[str, Any], league_mix_cfb: float) -> float:
    z = str(state.get("field_zone", "UNK"))
    down = int(state.get("down", 1))
    dist = str(state.get("dist_bucket", "UNK"))
    clock = str(state.get("clock_bucket", "OTHER"))
    gtg = bool(state.get("goal_to_go", False))

    base = _blend(EP_ZONE_CFB.get(z, 2.0), EP_ZONE_NFL.get(z, 1.8), league_mix_cfb)
    base += DOWN_ADJ.get(down, -0.2) + DIST_ADJ.get(dist, -0.35)
    base *= CLOCK_MULT.get(clock, 1.0)

    # goal-to-go slightly higher EP in red zone
    if gtg and z in ("LOW_RED", "HIGH_RED"):
        base += 0.35

    # clamp
    return max(-1.5, min(6.8, base))

def _shift_zone(zone: str, step: int) -> str:
    if zone not in ZONE_LADDER:
        return "UNK"
    i = ZONE_LADDER.index(zone)
    j = max(0, min(len(ZONE_LADDER) - 1, i + step))
    return ZONE_LADDER[j]

def next_state_from_result(state: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Approximate next state after a play based on buckets.
    We model:
    - TD ends drive (handled in EP_after)
    - turnovers flip possession (handled in EP_after)
    - first down resets to 1st/medium and might advance zone
    - otherwise down increments and dist tends to worsen/improve depending on yards_bucket
    """
    z = str(state.get("field_zone", "UNK"))
    down = int(state.get("down", 1))
    dist = str(state.get("dist_bucket", "UNK"))
    clock = str(state.get("clock_bucket", "OTHER"))
    gtg = bool(state.get("goal_to_go", False))

    fd = bool(result.get("first_down", False))
    yards_b = str(result.get("yards_bucket", "NA"))
    td = bool(result.get("td", False))
    turnover = str(result.get("turnover", "NONE"))

    # if scoring/turnover, state irrelevant (handled elsewhere)
    if td or turnover in ("INT", "FUMBLE", "PICK6", "SCOOP6"):
        return dict(state)

    # Estimate zone movement by yards bucket
    zone_step = 0
    if yards_b == "21+":
        zone_step = 2
    elif yards_b == "11-20":
        zone_step = 1
    elif yards_b == "7-10":
        zone_step = 1 if z in ("BACKED_UP", "OWN_SIDE") else 0
    elif yards_b == "NEG":
        zone_step = -1
    else:
        zone_step = 0

    z2 = _shift_zone(z, zone_step)

    if fd:
        # new series
        return {
            "field_zone": z2,
            "down": 1,
            "dist_bucket": "MEDIUM",
            "clock_bucket": clock,
            "goal_to_go": gtg if z2 in ("LOW_RED", "HIGH_RED") else False,
        }

    # no first down: increment down
    down2 = min(4, down + 1)

    # crude dist update: good gain tends to shorten, bad gain lengthens
    if yards_b in ("11-20", "21+"):
        dist2 = "SHORT"
    elif yards_b in ("7-10", "3-6"):
        dist2 = "MEDIUM"
    elif yards_b in ("0-2", "NA"):
        dist2 = "LONG"
    elif yards_b == "NEG":
        dist2 = "X_LONG"
    else:
        dist2 = dist

    return {
        "field_zone": z2,
        "down": down2,
        "dist_bucket": dist2,
        "clock_bucket": clock,
        "goal_to_go": gtg,
    }

def ep_after(state_pre: Dict[str, Any], result: Dict[str, Any], league_mix_cfb: float) -> float:
    """
    Compute EP after the play.
    - TD => +7 (approx; ignores XP variability but we model 2pt separately elsewhere)
    - PICK6/SCOOP6 => -7
    - other turnovers => negative EP of same state (possession flips)
    - otherwise EP of next state
    """
    td = bool(result.get("td", False))
    turnover = str(result.get("turnover", "NONE"))

    if turnover in ("PICK6", "SCOOP6"):
        return -7.0

    if td:
        return 7.0

    if turnover in ("INT", "FUMBLE"):
        # possession flips; opponent now has the “mirror” value — approximate by negating EP
        return -ep_pre(state_pre, league_mix_cfb)

    # normal transition
    st2 = next_state_from_result(state_pre, result)
    return ep_pre(st2, league_mix_cfb)

def epa_for_row(row: Dict[str, Any], league_mix_cfb: float) -> Optional[float]:
    """
    Requires at least: down/dist/zone/clock and result fields (td/turnover/first_down/yards_bucket).
    If result not labeled, returns None.
    """
    if row.get("td") is None and row.get("turnover") is None and row.get("first_down") is None and row.get("yards_bucket") is None:
        return None

    state = {
        "down": row.get("down", 1),
        "dist_bucket": row.get("dist_bucket", "UNK"),
        "field_zone": row.get("field_zone", "UNK"),
        "clock_bucket": row.get("clock_bucket", "OTHER"),
        "goal_to_go": row.get("goal_to_go", False),
    }
    result = {
        "first_down": row.get("first_down", False),
        "td": row.get("td", False),
        "yards_bucket": row.get("yards_bucket", "NA"),
        "turnover": row.get("turnover", "NONE") if row.get("turnover") is not None else "NONE",
    }

    pre = ep_pre(state, league_mix_cfb)
    post = ep_after(state, result, league_mix_cfb)
    return post - pre
