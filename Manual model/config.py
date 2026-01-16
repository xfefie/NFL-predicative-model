from pathlib import Path

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "events.parquet"

# =====================================================
# Buckets
# =====================================================
CLOCK_BUCKETS = [
    "15-10",
    "10-7",
    "7-6",
    "5-3",
    "3-2",
    "2-0",
    "SCRIPT_START",
    "OTHER",
]
DIST_BUCKETS = ["SHORT", "MEDIUM", "LONG", "X_LONG", "UNK"]
FIELD_ZONES = ["LOW_RED", "HIGH_RED", "MIDFIELD", "OWN_SIDE", "BACKED_UP", "UNK"]

# Simple “in-range” defs (bucket-world)
# CFB: mostly red zone range
FG_RANGE_ZONES_CFB = {"LOW_RED", "HIGH_RED"}
# NFL: red zone + fringe (midfield sometimes)
FG_RANGE_ZONES_NFL = {"LOW_RED", "HIGH_RED", "MIDFIELD"}

# =====================================================
# Taxonomy
# =====================================================
PERSONNEL = ["UNK", "10", "11", "12", "13", "20", "21", "22"]
FORMATION = ["UNK", "2x2", "3x1", "trips", "bunch", "empty", "compressed"]
SHELL = ["UNK", "0", "1", "2"]
PRESSURE = ["UNK", "4", "5+"]

# =====================================================
# Call types (what the play IS)
# =====================================================
CALL_TYPES = [
    "RUN",
    "PASS_QUICK",
    "PASS_DROPBACK",
    "PLAY_ACTION",
    "SCREEN",
    "SHOT",
    "PUNT",
    "FIELD_GOAL",
    "KICKOFF",
    "PAT_KICK",
    "TWO_POINT",
    "SACK",
    "PENALTY",
]

# =====================================================
# Results / outcomes
# =====================================================
PASS_RESULT = ["NA", "COMPLETE", "INCOMPLETE"]
TURNOVER_RESULT = ["NONE", "INT", "FUMBLE", "PICK6", "SCOOP6"]
YARDS_BUCKETS = ["NA", "NEG", "0-2", "3-6", "7-10", "11-20", "21+"]

# 4th down decision + 2pt decision
GO_NO_GO = ["GO", "NO_GO"]
TWO_PT_CHOICE = ["KICK", "TWO"]

# =====================================================
# Priors for offensive call family (CFB + NFL)
# =====================================================
_PRIOR_CFB = {
    (1, "SHORT"):   {"RUN": 40, "PASS_QUICK": 16, "PASS_DROPBACK": 12, "PLAY_ACTION": 10, "SCREEN": 6, "SHOT": 6, "SACK": 1, "PENALTY": 1},
    (1, "MEDIUM"):  {"RUN": 28, "PASS_QUICK": 18, "PASS_DROPBACK": 18, "PLAY_ACTION": 12, "SCREEN": 8, "SHOT": 8, "SACK": 1, "PENALTY": 1},
    (1, "LONG"):    {"RUN": 16, "PASS_QUICK": 14, "PASS_DROPBACK": 28, "PLAY_ACTION": 10, "SCREEN": 10, "SHOT": 10, "SACK": 1, "PENALTY": 1},
    (1, "X_LONG"):  {"RUN": 10, "PASS_QUICK": 12, "PASS_DROPBACK": 34, "PLAY_ACTION": 8,  "SCREEN": 12, "SHOT": 12, "SACK": 1, "PENALTY": 1},

    (2, "SHORT"):   {"RUN": 36, "PASS_QUICK": 18, "PASS_DROPBACK": 12, "PLAY_ACTION": 10, "SCREEN": 6, "SHOT": 6, "SACK": 1, "PENALTY": 1},
    (2, "MEDIUM"):  {"RUN": 24, "PASS_QUICK": 18, "PASS_DROPBACK": 22, "PLAY_ACTION": 12, "SCREEN": 8, "SHOT": 8, "SACK": 1, "PENALTY": 1},
    (2, "LONG"):    {"RUN": 12, "PASS_QUICK": 14, "PASS_DROPBACK": 32, "PLAY_ACTION": 10, "SCREEN": 12, "SHOT": 10, "SACK": 1, "PENALTY": 1},
    (2, "X_LONG"):  {"RUN": 8,  "PASS_QUICK": 12, "PASS_DROPBACK": 36, "PLAY_ACTION": 8,  "SCREEN": 14, "SHOT": 12, "SACK": 1, "PENALTY": 1},

    (3, "SHORT"):   {"RUN": 24, "PASS_QUICK": 20, "PASS_DROPBACK": 18, "PLAY_ACTION": 8,  "SCREEN": 6, "SHOT": 6, "SACK": 1, "PENALTY": 1},
    (3, "MEDIUM"):  {"RUN": 10, "PASS_QUICK": 16, "PASS_DROPBACK": 38, "PLAY_ACTION": 8,  "SCREEN": 12, "SHOT": 12, "SACK": 2, "PENALTY": 2},
    (3, "LONG"):    {"RUN": 6,  "PASS_QUICK": 12, "PASS_DROPBACK": 44, "PLAY_ACTION": 6,  "SCREEN": 14, "SHOT": 14, "SACK": 3, "PENALTY": 2},
    (3, "X_LONG"):  {"RUN": 4,  "PASS_QUICK": 10, "PASS_DROPBACK": 46, "PLAY_ACTION": 4,  "SCREEN": 16, "SHOT": 16, "SACK": 3, "PENALTY": 1},

    (4, "SHORT"):   {"RUN": 18, "PASS_QUICK": 12, "PASS_DROPBACK": 16, "PLAY_ACTION": 4,  "SCREEN": 8, "SHOT": 6, "SACK": 2, "PENALTY": 2},
    (4, "MEDIUM"):  {"RUN": 8,  "PASS_QUICK": 12, "PASS_DROPBACK": 34, "PLAY_ACTION": 3,  "SCREEN": 16, "SHOT": 14, "SACK": 2, "PENALTY": 1},
    (4, "LONG"):    {"RUN": 5,  "PASS_QUICK": 10, "PASS_DROPBACK": 40, "PLAY_ACTION": 2,  "SCREEN": 18, "SHOT": 18, "SACK": 2, "PENALTY": 1},
    (4, "X_LONG"):  {"RUN": 4,  "PASS_QUICK": 8,  "PASS_DROPBACK": 44, "PLAY_ACTION": 2,  "SCREEN": 20, "SHOT": 18, "SACK": 2, "PENALTY": 0},
}

_PRIOR_NFL = {
    (1, "SHORT"):   {"RUN": 34, "PASS_QUICK": 20, "PASS_DROPBACK": 12, "PLAY_ACTION": 10, "SCREEN": 6, "SHOT": 6, "SACK": 1, "PENALTY": 1},
    (1, "MEDIUM"):  {"RUN": 22, "PASS_QUICK": 20, "PASS_DROPBACK": 22, "PLAY_ACTION": 12, "SCREEN": 8, "SHOT": 8, "SACK": 1, "PENALTY": 1},
    (1, "LONG"):    {"RUN": 10, "PASS_QUICK": 14, "PASS_DROPBACK": 36, "PLAY_ACTION": 8,  "SCREEN": 14, "SHOT": 14, "SACK": 2, "PENALTY": 2},
    (1, "X_LONG"):  {"RUN": 6,  "PASS_QUICK": 12, "PASS_DROPBACK": 40, "PLAY_ACTION": 6,  "SCREEN": 16, "SHOT": 16, "SACK": 2, "PENALTY": 2},

    (2, "SHORT"):   {"RUN": 28, "PASS_QUICK": 22, "PASS_DROPBACK": 14, "PLAY_ACTION": 10, "SCREEN": 6, "SHOT": 6, "SACK": 2, "PENALTY": 2},
    (2, "MEDIUM"):  {"RUN": 16, "PASS_QUICK": 18, "PASS_DROPBACK": 30, "PLAY_ACTION": 10, "SCREEN": 12, "SHOT": 10, "SACK": 2, "PENALTY": 2},
    (2, "LONG"):    {"RUN": 8,  "PASS_QUICK": 12, "PASS_DROPBACK": 44, "PLAY_ACTION": 6,  "SCREEN": 16, "SHOT": 12, "SACK": 2, "PENALTY": 2},
    (2, "X_LONG"):  {"RUN": 6,  "PASS_QUICK": 10, "PASS_DROPBACK": 46, "PLAY_ACTION": 5,  "SCREEN": 16, "SHOT": 13, "SACK": 2, "PENALTY": 2},

    (3, "SHORT"):   {"RUN": 14, "PASS_QUICK": 22, "PASS_DROPBACK": 30, "PLAY_ACTION": 6,  "SCREEN": 12, "SHOT": 10, "SACK": 3, "PENALTY": 3},
    (3, "MEDIUM"):  {"RUN": 6,  "PASS_QUICK": 16, "PASS_DROPBACK": 52, "PLAY_ACTION": 4,  "SCREEN": 12, "SHOT": 8,  "SACK": 2, "PENALTY": 2},
    (3, "LONG"):    {"RUN": 4,  "PASS_QUICK": 12, "PASS_DROPBACK": 56, "PLAY_ACTION": 3,  "SCREEN": 12, "SHOT": 8,  "SACK": 3, "PENALTY": 2},
    (3, "X_LONG"):  {"RUN": 3,  "PASS_QUICK": 10, "PASS_DROPBACK": 58, "PLAY_ACTION": 2,  "SCREEN": 13, "SHOT": 8,  "SACK": 4, "PENALTY": 2},

    (4, "SHORT"):   {"RUN": 14, "PASS_QUICK": 14, "PASS_DROPBACK": 40, "PLAY_ACTION": 2,  "SCREEN": 14, "SHOT": 10, "SACK": 3, "PENALTY": 3},
    (4, "MEDIUM"):  {"RUN": 5,  "PASS_QUICK": 12, "PASS_DROPBACK": 58, "PLAY_ACTION": 2,  "SCREEN": 14, "SHOT": 7,  "SACK": 1, "PENALTY": 1},
    (4, "LONG"):    {"RUN": 3,  "PASS_QUICK": 10, "PASS_DROPBACK": 60, "PLAY_ACTION": 2,  "SCREEN": 13, "SHOT": 8,  "SACK": 2, "PENALTY": 2},
    (4, "X_LONG"):  {"RUN": 2,  "PASS_QUICK": 8,  "PASS_DROPBACK": 62, "PLAY_ACTION": 2,  "SCREEN": 14, "SHOT": 8,  "SACK": 2, "PENALTY": 2},
}

# =====================================================
# Context multipliers
# =====================================================
ZONE_MULT = {
    "LOW_RED":   {"RUN": 1.25, "SHOT": 0.70, "SCREEN": 0.90},
    "HIGH_RED":  {"RUN": 1.10, "SHOT": 0.85},
    "MIDFIELD":  {},
    "OWN_SIDE":  {"SHOT": 0.95},
    "BACKED_UP": {"RUN": 0.85, "PASS_QUICK": 1.10, "SCREEN": 1.10},
    "UNK":       {},
}
CLOCK_MULT = {
    "SCRIPT_START": {"PLAY_ACTION": 1.10, "SHOT": 1.05},
    "15-10":        {"PLAY_ACTION": 1.05},
    "2-0":          {"PASS_QUICK": 1.10, "SHOT": 0.90, "RUN": 0.90},
    "OTHER":        {},
}
HURRY_MULT = {"PASS_QUICK": 1.15, "SCREEN": 1.05, "PLAY_ACTION": 0.85}
GOAL_TO_GO_MULT = {"RUN": 1.20, "SHOT": 0.80, "PLAY_ACTION": 1.05}
AFTER_FIRST_DOWN_MULT = {"RUN": 1.05, "PLAY_ACTION": 1.05}

# =====================================================
# Pressure priors (P(5+) vs P(4)), keyed by (down, dist_bucket)
# =====================================================
PRESSURE_PRIOR = {
    (1, "SHORT"):  {"4": 34, "5+": 6},
    (1, "MEDIUM"): {"4": 32, "5+": 8},
    (1, "LONG"):   {"4": 30, "5+": 10},
    (1, "X_LONG"): {"4": 28, "5+": 12},

    (2, "SHORT"):  {"4": 32, "5+": 8},
    (2, "MEDIUM"): {"4": 30, "5+": 10},
    (2, "LONG"):   {"4": 28, "5+": 12},
    (2, "X_LONG"): {"4": 26, "5+": 14},

    (3, "SHORT"):  {"4": 30, "5+": 10},
    (3, "MEDIUM"): {"4": 26, "5+": 14},
    (3, "LONG"):   {"4": 24, "5+": 16},
    (3, "X_LONG"): {"4": 22, "5+": 18},

    (4, "SHORT"):  {"4": 26, "5+": 14},
    (4, "MEDIUM"): {"4": 22, "5+": 18},
    (4, "LONG"):   {"4": 20, "5+": 20},
    (4, "X_LONG"): {"4": 18, "5+": 22},
}

# =====================================================
# Timeout usage priors: P(timeout_used=True)
# keyed by (quarter, clock_bucket, hurry_up)
# =====================================================
TIMEOUT_PRIOR = {
    (2, "3-2", False): {"NO": 34, "YES": 6},
    (2, "2-0", False): {"NO": 28, "YES": 12},
    (2, "3-2", True):  {"NO": 26, "YES": 14},
    (2, "2-0", True):  {"NO": 18, "YES": 22},

    (4, "3-2", False): {"NO": 32, "YES": 8},
    (4, "2-0", False): {"NO": 24, "YES": 16},
    (4, "3-2", True):  {"NO": 24, "YES": 16},
    (4, "2-0", True):  {"NO": 16, "YES": 24},
}

# =====================================================
# NEW: 4th-down decision priors (GO vs PUNT vs FIELD_GOAL)
# keyed by (dist_bucket, field_zone)
# These reflect typical CFB vs NFL tendencies in a bucketed way.
# =====================================================
FOURTH_TRI_CFB = {
    ("SHORT", "LOW_RED"):   {"GO": 26, "FIELD_GOAL": 10, "PUNT": 4},
    ("SHORT", "HIGH_RED"):  {"GO": 18, "FIELD_GOAL": 16, "PUNT": 6},
    ("SHORT", "MIDFIELD"):  {"GO": 10, "FIELD_GOAL": 2,  "PUNT": 28},
    ("SHORT", "OWN_SIDE"):  {"GO": 4,  "FIELD_GOAL": 0.5,"PUNT": 35},
    ("SHORT", "BACKED_UP"): {"GO": 2,  "FIELD_GOAL": 0.2,"PUNT": 38},

    ("MEDIUM", "LOW_RED"):   {"GO": 16, "FIELD_GOAL": 16, "PUNT": 8},
    ("MEDIUM", "HIGH_RED"):  {"GO": 10, "FIELD_GOAL": 22, "PUNT": 8},
    ("MEDIUM", "MIDFIELD"):  {"GO": 6,  "FIELD_GOAL": 1,  "PUNT": 33},
    ("MEDIUM", "OWN_SIDE"):  {"GO": 2,  "FIELD_GOAL": 0.2,"PUNT": 38},
    ("MEDIUM", "BACKED_UP"): {"GO": 1,  "FIELD_GOAL": 0.1,"PUNT": 39},

    ("LONG", "LOW_RED"):   {"GO": 8,  "FIELD_GOAL": 26, "PUNT": 6},
    ("LONG", "HIGH_RED"):  {"GO": 4,  "FIELD_GOAL": 30, "PUNT": 6},
    ("LONG", "MIDFIELD"):  {"GO": 3,  "FIELD_GOAL": 0.5,"PUNT": 36},
    ("LONG", "OWN_SIDE"):  {"GO": 1,  "FIELD_GOAL": 0.1,"PUNT": 39},
    ("LONG", "BACKED_UP"): {"GO": 0.5,"FIELD_GOAL": 0.1,"PUNT": 39.4},

    ("X_LONG", "LOW_RED"):   {"GO": 5,  "FIELD_GOAL": 30, "PUNT": 5},
    ("X_LONG", "HIGH_RED"):  {"GO": 3,  "FIELD_GOAL": 32, "PUNT": 5},
    ("X_LONG", "MIDFIELD"):  {"GO": 2,  "FIELD_GOAL": 0.2,"PUNT": 37.8},
    ("X_LONG", "OWN_SIDE"):  {"GO": 0.5,"FIELD_GOAL": 0.1,"PUNT": 39.4},
    ("X_LONG", "BACKED_UP"): {"GO": 0.2,"FIELD_GOAL": 0.1,"PUNT": 39.7},
}

FOURTH_TRI_NFL = {
    ("SHORT", "LOW_RED"):   {"GO": 22, "FIELD_GOAL": 14, "PUNT": 4},
    ("SHORT", "HIGH_RED"):  {"GO": 14, "FIELD_GOAL": 22, "PUNT": 4},
    ("SHORT", "MIDFIELD"):  {"GO": 8,  "FIELD_GOAL": 8,  "PUNT": 24},
    ("SHORT", "OWN_SIDE"):  {"GO": 3,  "FIELD_GOAL": 1,  "PUNT": 36},
    ("SHORT", "BACKED_UP"): {"GO": 1.5,"FIELD_GOAL": 0.2,"PUNT": 38.3},

    ("MEDIUM", "LOW_RED"):   {"GO": 14, "FIELD_GOAL": 20, "PUNT": 6},
    ("MEDIUM", "HIGH_RED"):  {"GO": 8,  "FIELD_GOAL": 26, "PUNT": 6},
    ("MEDIUM", "MIDFIELD"):  {"GO": 5,  "FIELD_GOAL": 10, "PUNT": 25},
    ("MEDIUM", "OWN_SIDE"):  {"GO": 2,  "FIELD_GOAL": 1,  "PUNT": 37},
    ("MEDIUM", "BACKED_UP"): {"GO": 1,  "FIELD_GOAL": 0.2,"PUNT": 38.8},

    ("LONG", "LOW_RED"):   {"GO": 8,  "FIELD_GOAL": 30, "PUNT": 2},
    ("LONG", "HIGH_RED"):  {"GO": 4,  "FIELD_GOAL": 34, "PUNT": 2},
    ("LONG", "MIDFIELD"):  {"GO": 3,  "FIELD_GOAL": 12, "PUNT": 25},
    ("LONG", "OWN_SIDE"):  {"GO": 1,  "FIELD_GOAL": 1,  "PUNT": 38},
    ("LONG", "BACKED_UP"): {"GO": 0.5,"FIELD_GOAL": 0.2,"PUNT": 39.3},

    ("X_LONG", "LOW_RED"):   {"GO": 5,  "FIELD_GOAL": 34, "PUNT": 1},
    ("X_LONG", "HIGH_RED"):  {"GO": 3,  "FIELD_GOAL": 36, "PUNT": 1},
    ("X_LONG", "MIDFIELD"):  {"GO": 2,  "FIELD_GOAL": 10, "PUNT": 28},
    ("X_LONG", "OWN_SIDE"):  {"GO": 0.5,"FIELD_GOAL": 0.5,"PUNT": 39},
    ("X_LONG", "BACKED_UP"): {"GO": 0.2,"FIELD_GOAL": 0.2,"PUNT": 39.6},
}

# Keep your older GO/NO_GO and TWO_PT priors (used elsewhere)
FOURTH_PRIOR = {}
TWO_PT_PRIOR = {"TWO": 4, "KICK": 36}

# =====================================================
# Helper for blending CFB/NFL priors
# =====================================================
def get_base_alpha(down: int, dist_bucket: str, league_mix_cfb: float) -> dict:
    key = (int(down), str(dist_bucket))
    cfb = _PRIOR_CFB.get(key)
    nfl = _PRIOR_NFL.get(key)
    if cfb is None and nfl is None:
        cfb = {"RUN": 20, "PASS_QUICK": 15, "PASS_DROPBACK": 25, "PLAY_ACTION": 10, "SCREEN": 10, "SHOT": 10, "SACK": 5, "PENALTY": 5}
        nfl = cfb

    offense_keys = ["RUN", "PASS_QUICK", "PASS_DROPBACK", "PLAY_ACTION", "SCREEN", "SHOT", "SACK", "PENALTY"]
    out = {}
    for k in offense_keys:
        out[k] = league_mix_cfb * float(cfb.get(k, 0)) + (1.0 - league_mix_cfb) * float(nfl.get(k, 0))
    return out
