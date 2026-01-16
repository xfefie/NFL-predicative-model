import pandas as pd

FEATURE_COLS = [
    "pv_possession",
    "quarter",
    "clock_bucket",
    "hurry_up",
    "down",
    "dist_bucket",
    "field_zone",
    "opp_personnel",
    "opp_formation",
    "def_shell",
    "pressure",
]

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cat_cols = ["pv_possession", "clock_bucket", "dist_bucket", "field_zone",
                "opp_personnel", "opp_formation", "def_shell", "pressure"]
    for c in cat_cols:
        if c not in out.columns:
            out[c] = "UNK"
        out[c] = out[c].fillna("UNK").astype(str)

    if "hurry_up" not in out.columns:
        out["hurry_up"] = False
    out["hurry_up"] = out["hurry_up"].fillna(False).astype(bool)

    for c in ["quarter", "down"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    return out[FEATURE_COLS]


