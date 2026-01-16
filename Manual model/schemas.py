from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time

def now_ts() -> float:
    return time.time()

@dataclass
class TagEvent:
    ts: float
    session_id: str
    game_id: str
    play_no: int

    quarter: int
    clock_bucket: str
    hurry_up: bool

    down: int
    dist_bucket: str
    field_zone: str
    goal_to_go: bool

    pv_possession: str  # PV_OFF or PV_DEF

    opp_personnel: Optional[str] = None
    opp_formation: Optional[str] = None
    def_shell: Optional[str] = None
    pressure: Optional[str] = None

    call_type: Optional[str] = None

    # Results
    first_down: Optional[bool] = None
    td: Optional[bool] = None
    yards_bucket: Optional[str] = None
    pass_result: Optional[str] = None
    turnover: Optional[str] = None

    # Decisions
    fourth_decision: Optional[str] = None
    two_pt_decision: Optional[str] = None

    # NEW: did a timeout get used between this and next play?
    timeout_used: Optional[bool] = None

    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["meta"] is None:
            d["meta"] = {}
        return d
