def mmss_to_seconds(mmss: str) -> int:
    try:
        m, s = mmss.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0

def clamp_int(x, lo, hi, default):
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default

def norm_choice(x: str, allowed: list, fallback="UNK") -> str:
    x = str(x).strip()
    return x if x in allowed else fallback
