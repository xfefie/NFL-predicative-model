# =====================================================
# PATH FIX
# =====================================================
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import uuid
import time

from config import (
    CLOCK_BUCKETS, DIST_BUCKETS, FIELD_ZONES,
    PERSONNEL, FORMATION, SHELL, PRESSURE,
    CALL_TYPES, PASS_RESULT, TURNOVER_RESULT, YARDS_BUCKETS,
    TWO_PT_CHOICE,
    FG_RANGE_ZONES_CFB, FG_RANGE_ZONES_NFL
)
from schemas import TagEvent, now_ts
from storage import upsert_event, upsert_many, load_events, list_session_game, get_play

from analytics.priors_model import (
    call_prior_alpha,
    posterior_mean,
    counts_from_live,
    derived_pass_conditionals,
    pressure_prior_alpha,
    timeout_prior_alpha,
    fourth_tri_prior,
)
from analytics.ep_model import ep_pre, epa_for_row, next_state_from_result

# =====================================================
# HELPERS
# =====================================================
def _pct(x):
    try:
        if x is None:
            return "NA"
        return f"{100*float(x):.0f}%"
    except Exception:
        return "NA"

def fg_in_range(field_zone: str, league_mix_cfb: float) -> bool:
    z = str(field_zone)
    if league_mix_cfb >= 0.6:
        return z in FG_RANGE_ZONES_CFB
    if league_mix_cfb <= 0.4:
        return z in FG_RANGE_ZONES_NFL
    return z in FG_RANGE_ZONES_CFB  # conservative in the middle

def make_export_df(df_sg: pd.DataFrame, session_id: str, game_id: str) -> pd.DataFrame:
    if df_sg is None or df_sg.empty:
        return pd.DataFrame()

    out = df_sg.copy()
    if "session_id" not in out.columns:
        out["session_id"] = session_id
    if "game_id" not in out.columns:
        out["game_id"] = game_id

    show_cols = [c for c in [
        "session_id","game_id",
        "play_no","quarter","clock_bucket","hurry_up","down","dist_bucket","field_zone","goal_to_go",
        "pv_possession","def_shell","pressure",
        "call_type","first_down","td","yards_bucket","pass_result","turnover","timeout_used",
        "fourth_decision","two_pt_decision"
    ] if c in out.columns]
    return out[show_cols]

def map_4th_tri_from_call_type(ct: str) -> str:
    ct = str(ct)
    if ct == "PUNT":
        return "PUNT"
    if ct == "FIELD_GOAL":
        return "FIELD_GOAL"
    return "GO"

def build_result_dict(row: dict) -> dict:
    return {
        "first_down": bool(row.get("first_down", False)),
        "td": bool(row.get("td", False)),
        "yards_bucket": str(row.get("yards_bucket", "NA")),
        "turnover": str(row.get("turnover", "NONE") if row.get("turnover") is not None else "NONE"),
    }

def build_state_pre_dict(row: dict) -> dict:
    return {
        "down": int(row.get("down", 1)),
        "dist_bucket": str(row.get("dist_bucket", "UNK")),
        "field_zone": str(row.get("field_zone", "UNK")),
        "clock_bucket": str(row.get("clock_bucket", "OTHER")),
        "goal_to_go": bool(row.get("goal_to_go", False)),
    }

# ============================
# COACH SUMMARY (4 sentences each side)
# ============================
def summarize_offense(df_labeled: pd.DataFrame) -> str:
    dfo = df_labeled[df_labeled.get("pv_possession") == "PV_OFF"].copy()
    if dfo.empty:
        return ("PV offense: no labeled offensive plays yet. Tag a few PV_OFF plays to unlock tendencies. "
                "Once we have them, we’ll show run/pass mix, top call families, and situational breakers. "
                "For now, the model relies on priors + early-game script assumptions.")

    call = dfo["call_type"].astype(str).value_counts(normalize=True)
    p_run = float(call.get("RUN", 0.0))
    p_passfam = float(call.get("PASS_QUICK", 0.0) + call.get("PASS_DROPBACK", 0.0) +
                      call.get("PLAY_ACTION", 0.0) + call.get("SCREEN", 0.0) + call.get("SHOT", 0.0))

    d1 = dfo[dfo.get("down") == 1]
    call1 = d1["call_type"].astype(str).value_counts(normalize=True) if not d1.empty else pd.Series(dtype=float)
    p1_run = float(call1.get("RUN", 0.0))
    p1_pa = float(call1.get("PLAY_ACTION", 0.0))
    p1_shot = float(call1.get("SHOT", 0.0))

    press = dfo[dfo.get("pressure").notna()]
    p_press5 = float((press["pressure"].astype(str) == "5+").mean()) if not press.empty else None

    breaker = []
    if p_run > 0.60:
        breaker.append("break with early-down play-action/shot looks")
    elif p_passfam > 0.65:
        breaker.append("break with run/screen to punish light boxes")
    if p_press5 is not None and p_press5 > 0.30:
        breaker.append("lean quick game/screens vs pressure")
    if not breaker:
        breaker.append("mix in constraint plays to stay unpredictable")

    s1 = f"PV offense is {_pct(p_run)} run / {_pct(p_passfam)} pass-family overall."
    s2 = f"On 1st down, run is {_pct(p1_run)} with PA {_pct(p1_pa)} and shots {_pct(p1_shot)}."
    s3 = f"Pressure faced (5+) is {_pct(p_press5)}." if p_press5 is not None else "Pressure faced isn’t stable yet (need more pressure tags)."
    s4 = f"Tendency-break idea: {', '.join(breaker)}."
    return " ".join([s1, s2, s3, s4])

def summarize_defense(df_labeled: pd.DataFrame) -> str:
    dfd = df_labeled[df_labeled.get("pv_possession") == "PV_DEF"].copy()
    if dfd.empty:
        return ("PV defense: no labeled defensive snaps yet. Tag PV_DEF plays to unlock opponent tendencies. "
                "Once we have them, we’ll show their run/pass/shot rates by down and field zone. "
                "For now, the model relies on priors + early-game scouting assumptions. "
                "As tags accumulate, we’ll identify the cleanest breaker windows.")

    call = dfd["call_type"].astype(str).value_counts(normalize=True)
    opp_run = float(call.get("RUN", 0.0))
    opp_passfam = float(call.get("PASS_QUICK", 0.0) + call.get("PASS_DROPBACK", 0.0) +
                        call.get("PLAY_ACTION", 0.0) + call.get("SCREEN", 0.0) + call.get("SHOT", 0.0))
    opp_shot = float(call.get("SHOT", 0.0))
    opp_screen = float(call.get("SCREEN", 0.0))

    shell = dfd[dfd.get("def_shell").notna()]
    p_two_high = float((shell["def_shell"].astype(str) == "2").mean()) if not shell.empty else None

    press = dfd[dfd.get("pressure").notna()]
    p_press5 = float((press["pressure"].astype(str) == "5+").mean()) if not press.empty else None

    d3 = dfd[dfd.get("down") == 3]
    call3 = d3["call_type"].astype(str).value_counts(normalize=True) if not d3.empty else pd.Series(dtype=float)
    opp3_drop = float(call3.get("PASS_DROPBACK", 0.0) + call3.get("SHOT", 0.0))

    breaker = []
    if opp_run > 0.60:
        breaker.append("load box / force long-yardage")
    if opp_shot > 0.12:
        breaker.append("rotate late / protect posts on likely shot downs")
    if opp_screen > 0.10 and (p_press5 is not None and p_press5 > 0.30):
        breaker.append("screen-alert when blitzing (peel/replace)")
    if not breaker:
        breaker.append("vary shell + simulated pressure to break their read")

    s1 = f"Opponent offense is {_pct(opp_run)} run / {_pct(opp_passfam)} pass-family with shots {_pct(opp_shot)} and screens {_pct(opp_screen)}."
    s2 = f"On 3rd down, dropback/shot tendency is {_pct(opp3_drop)}."
    s3 = f"Your tags show two-high {_pct(p_two_high)} and 5+ pressure {_pct(p_press5)}." if (p_two_high is not None and p_press5 is not None) else "Shell/pressure tendencies need more tags to stabilize."
    s4 = f"Tendency-break idea: {', '.join(breaker)}."
    return " ".join([s1, s2, s3, s4])

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="PV Tagger + Coaching Dashboard", layout="wide")
st.title("PV Tagger + Coaching Dashboard — Live Tagger + Coaching Probs + Coach Summary")

# =====================================================
# SESSION STATE
# =====================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "game_id" not in st.session_state:
    st.session_state.game_id = "practice_game"
if "play_no" not in st.session_state:
    st.session_state.play_no = 1
if "pv_possession" not in st.session_state:
    st.session_state.pv_possession = "PV_DEF"

# =====================================================
# TOP BAR
# =====================================================
top = st.columns([1.2, 1.6, 1, 1, 1.5])
with top[0]:
    st.text_input("Session", value=st.session_state.session_id, disabled=True)
with top[1]:
    st.session_state.game_id = st.text_input("Game ID", value=st.session_state.game_id)
with top[2]:
    st.session_state.play_no = st.number_input("Play #", min_value=1, value=int(st.session_state.play_no), step=1)
with top[3]:
    st.session_state.pv_possession = st.selectbox(
        "PV Possession",
        ["PV_OFF", "PV_DEF"],
        index=0 if st.session_state.pv_possession == "PV_OFF" else 1
    )
with top[4]:
    if st.button("➕ Next Play", use_container_width=True):
        st.session_state.play_no += 1
        st.rerun()

st.divider()
tab_tagger, tab_dash = st.tabs(["🏷️ Tagger + Import", "📊 Coaching Dashboard"])

# =====================================================
# TAGGER TAB
# =====================================================
with tab_tagger:
    st.subheader("Fast Tagger (Buckets Only) + CSV Importer")

    with st.expander("📥 Import CSV into database (upsert)", expanded=False):
        uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                df_imp = pd.read_csv(uploaded)

                if "session_id" not in df_imp.columns:
                    df_imp["session_id"] = st.session_state.session_id
                    st.warning("CSV missing session_id — auto-filled with current session.")
                if "game_id" not in df_imp.columns:
                    df_imp["game_id"] = st.session_state.game_id
                    st.warning("CSV missing game_id — auto-filled with current game_id.")

                req = ["session_id", "game_id", "play_no"]
                missing = [c for c in req if c not in df_imp.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    df_imp["play_no"] = pd.to_numeric(df_imp["play_no"], errors="coerce").fillna(0).astype(int)
                    if "ts" not in df_imp.columns:
                        df_imp["ts"] = time.time()

                    for bcol in ["hurry_up", "goal_to_go", "first_down", "td", "timeout_used"]:
                        if bcol in df_imp.columns:
                            df_imp[bcol] = df_imp[bcol].astype(str).str.lower().isin(["true", "1", "yes", "y"])

                    st.dataframe(df_imp.head(25), use_container_width=True, height=280)
                    if st.button("✅ Import / Upsert", use_container_width=True):
                        upsert_many(df_imp)
                        st.success(f"Imported {len(df_imp)} rows.")
                        st.rerun()
            except Exception as e:
                st.error("Import failed.")
                st.exception(e)

    st.divider()

    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown("### Situation")
        r1 = st.columns([1, 1.6, 1, 1, 1.1])
        with r1[0]:
            quarter = st.selectbox("Q", [1, 2, 3, 4], key="tag_q")
        with r1[1]:
            clock_bucket = st.selectbox("Clock Bucket", CLOCK_BUCKETS, index=0, key="tag_clock_bucket")
        with r1[2]:
            hurry_up = st.toggle("HURRY UP", value=False, key="tag_hurry")
        with r1[3]:
            down = st.selectbox("Down", [1, 2, 3, 4], key="tag_down")
        with r1[4]:
            dist_bucket = st.selectbox("Dist Bucket", DIST_BUCKETS, index=0, key="tag_dist_bucket")

        r2 = st.columns([1.2, 1])
        with r2[0]:
            field_zone = st.selectbox("Field Zone", FIELD_ZONES, index=FIELD_ZONES.index("MIDFIELD"), key="tag_field_zone")
        with r2[1]:
            goal_to_go = st.toggle("GOAL TO GO", value=False, key="tag_gtg")

        st.markdown("### Opponent / Defensive Look (optional)")
        p1, p2 = st.columns(2)
        with p1:
            opp_personnel = st.selectbox("Opp Personnel", PERSONNEL, index=PERSONNEL.index("11"), key="tag_pers")
            opp_formation = st.selectbox("Opp Formation", FORMATION, index=FORMATION.index("2x2"), key="tag_form")
        with p2:
            def_shell = st.selectbox("Def Shell (0/1/2-high)", SHELL, key="tag_shell")
            pressure = st.selectbox("Pressure (4 vs 5+)", PRESSURE, key="tag_press")

        st.caption("Tag fast. Label results after the play (or during stoppages).")

    with right:
        st.markdown("### Submit")
        if st.button("✅ SUBMIT TAG (FAST)", use_container_width=True, key="btn_submit"):
            ev = TagEvent(
                ts=now_ts(),
                session_id=st.session_state.session_id,
                game_id=st.session_state.game_id,
                play_no=int(st.session_state.play_no),
                quarter=int(quarter),
                clock_bucket=str(clock_bucket),
                hurry_up=bool(hurry_up),
                down=int(down),
                dist_bucket=str(dist_bucket),
                field_zone=str(field_zone),
                goal_to_go=bool(goal_to_go),
                pv_possession=st.session_state.pv_possession,
                opp_personnel=None if opp_personnel == "UNK" else opp_personnel,
                opp_formation=None if opp_formation == "UNK" else opp_formation,
                def_shell=None if def_shell == "UNK" else def_shell,
                pressure=None if pressure == "UNK" else pressure,
                call_type=None,
                first_down=None,
                td=None,
                yards_bucket=None,
                pass_result=None,
                turnover=None,
                fourth_decision=None,
                two_pt_decision=None,
                timeout_used=None,
                meta={"source": "manual_fast_priors_fullfile"},
            )
            upsert_event(ev.to_dict())
            st.session_state.play_no += 1
            st.success("Saved tag + advanced play #.")
            st.rerun()

        st.divider()
        st.markdown("### Label / Edit (apply to ANY play)")

        df_all = load_events()
        df_sg = list_session_game(df_all, st.session_state.session_id, st.session_state.game_id)
        play_options = df_sg["play_no"].astype(int).tolist() if not df_sg.empty else []
        selected_play = st.selectbox(
            "Select Play #",
            options=play_options if play_options else [max(1, int(st.session_state.play_no) - 1)],
            index=len(play_options) - 1 if play_options else 0,
            key="sel_play"
        )

        call_type = st.selectbox("Call Type", CALL_TYPES, index=0, key="lab_call")

        cA, cB = st.columns(2)
        with cA:
            first_down = st.toggle("First Down", value=False, key="lab_fd")
            td = st.toggle("TD", value=False, key="lab_td")
            yards_bucket = st.selectbox("Yards Bucket", YARDS_BUCKETS, index=0, key="lab_yards")
            timeout_used = st.toggle("Timeout used (between plays)", value=False, key="lab_to_used")
        with cB:
            pass_result = st.selectbox("Pass Result", PASS_RESULT, index=0, key="lab_pass_res")
            turnover = st.selectbox("Turnover", TURNOVER_RESULT, index=0, key="lab_to")

        row_for_play = get_play(df_all, st.session_state.session_id, st.session_state.game_id, int(selected_play))

        two_pt_decision = None
        if td:
            two_pt_decision = st.radio("After TD: 2pt or Kick?", TWO_PT_CHOICE, horizontal=True, key="lab_2pt")

        if st.button("💾 APPLY LABELS", use_container_width=True, key="btn_apply_labels"):
            if row_for_play.empty:
                st.error("Could not find that play.")
            else:
                d = row_for_play.iloc[-1].to_dict()
                d["call_type"] = str(call_type)
                d["first_down"] = bool(first_down)
                d["td"] = bool(td)
                d["yards_bucket"] = str(yards_bucket)
                d["pass_result"] = str(pass_result)
                d["turnover"] = str(turnover)
                d["timeout_used"] = bool(timeout_used)
                if td:
                    d["two_pt_decision"] = str(two_pt_decision)
                d["ts"] = now_ts()
                upsert_event(d)
                st.success(f"Saved labels for play #{selected_play}.")
                st.rerun()

    st.divider()
    st.markdown("### Latest Plays (this session/game)")
    df_all2 = load_events()
    df_sg2 = list_session_game(df_all2, st.session_state.session_id, st.session_state.game_id)
    if df_sg2.empty:
        st.info("No plays yet.")
    else:
        exp_df = make_export_df(df_sg2, st.session_state.session_id, st.session_state.game_id)
        st.dataframe(exp_df.tail(40), use_container_width=True, height=360)
        st.download_button(
            "⬇️ Download this game CSV (re-importable)",
            data=exp_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{st.session_state.game_id}_{st.session_state.session_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

# =====================================================
# DASHBOARD TAB
# =====================================================
with tab_dash:
    st.subheader("Coaching Dashboard (Full probs + 3rd→4th Preview + TD→2pt Preview + Coach Summary)")

    df_all = load_events()
    df_live = list_session_game(df_all, st.session_state.session_id, st.session_state.game_id)
    if df_live.empty:
        st.warning("No plays yet. Tag a few plays first.")
        st.stop()

    # normalize
    df_live["clock_bucket"] = df_live.get("clock_bucket", "OTHER").fillna("OTHER").astype(str)
    df_live["dist_bucket"] = df_live.get("dist_bucket", "UNK").fillna("UNK").astype(str)
    df_live["field_zone"] = df_live.get("field_zone", "UNK").fillna("UNK").astype(str)
    df_live["goal_to_go"] = df_live.get("goal_to_go", False).fillna(False).astype(bool)
    df_live["hurry_up"] = df_live.get("hurry_up", False).fillna(False).astype(bool)
    df_live["pv_possession"] = df_live.get("pv_possession", "PV_DEF").fillna("PV_DEF").astype(str)
    df_live["down"] = pd.to_numeric(df_live.get("down", 1), errors="coerce").fillna(1).astype(int)
    df_live["quarter"] = pd.to_numeric(df_live.get("quarter", 1), errors="coerce").fillna(1).astype(int)

    latest = df_live.tail(1).iloc[0].to_dict()
    df_labeled = df_live[df_live.get("call_type").notna()].copy() if "call_type" in df_live.columns else pd.DataFrame()
    latest_labeled = df_labeled.tail(1).iloc[0].to_dict() if not df_labeled.empty else None

    st.markdown("### Current Situation (latest tagged)")
    st.dataframe(pd.DataFrame([{
        "play_no": latest.get("play_no"),
        "Q": latest.get("quarter"),
        "clock_bucket": latest.get("clock_bucket"),
        "hurry_up": latest.get("hurry_up"),
        "down": latest.get("down"),
        "dist_bucket": latest.get("dist_bucket"),
        "field_zone": latest.get("field_zone"),
        "goal_to_go": latest.get("goal_to_go"),
        "shell": latest.get("def_shell"),
        "pressure_tag": latest.get("pressure"),
    }]), use_container_width=True)

    st.divider()
    st.markdown("### Prior Controls (CFB + NFL)")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.1])
    with c1:
        league_mix_cfb = st.slider("CFB weight (NFL=0, CFB=1)", 0.0, 1.0, 0.5, 0.05)
    with c2:
        prior_strength = st.slider("Prior strength (pseudo-plays)", 0.2, 4.0, 1.0, 0.1)
    with c3:
        st.caption("4th-down preview triggers after 3rd-down NO first down; 2pt preview after TD.")

    after_first_down = bool(latest_labeled.get("first_down", False)) if latest_labeled is not None else False

    cond = {
        "pv_possession": latest.get("pv_possession", "PV_DEF"),
        "quarter": int(latest.get("quarter", 1)),
        "down": int(latest.get("down", 1)),
        "dist_bucket": latest.get("dist_bucket", "UNK"),
        "field_zone": latest.get("field_zone", "UNK"),
        "clock_bucket": latest.get("clock_bucket", "OTHER"),
        "hurry_up": bool(latest.get("hurry_up", False)),
        "goal_to_go": bool(latest.get("goal_to_go", False)),
    }

    in_range_now = fg_in_range(cond["field_zone"], float(league_mix_cfb))

    # Call-type posterior
    live_counts_call = counts_from_live(df_labeled, cond, label_col="call_type") if not df_labeled.empty else {}
    prior_alpha = call_prior_alpha(
        down=cond["down"],
        dist_bucket=cond["dist_bucket"],
        field_zone=cond["field_zone"],
        clock_bucket=cond["clock_bucket"],
        hurry_up=cond["hurry_up"],
        league_mix_cfb=float(league_mix_cfb),
        prior_strength=float(prior_strength),
        goal_to_go=cond["goal_to_go"],
        after_first_down=after_first_down,
        fg_in_range=in_range_now,
    )
    post_call = posterior_mean(prior_alpha, live_counts_call)
    deriv = derived_pass_conditionals(post_call)

    # Pressure posterior
    df_press = df_live[df_live.get("pressure").notna()].copy() if "pressure" in df_live.columns else pd.DataFrame()
    live_counts_press = counts_from_live(df_press, cond, label_col="pressure") if not df_press.empty else {}
    prior_press = pressure_prior_alpha(cond["down"], cond["dist_bucket"], strength=float(prior_strength))
    post_press = posterior_mean(prior_press, live_counts_press)
    p_press_5p = post_press.get("5+", 0.0)

    # Timeout posterior
    df_to = df_live[df_live.get("timeout_used").notna()].copy() if "timeout_used" in df_live.columns else pd.DataFrame()
    if not df_to.empty:
        df_to = df_to.copy()
        df_to["timeout_used_label"] = df_to["timeout_used"].map(lambda x: "YES" if bool(x) else "NO")
        live_counts_to = counts_from_live(df_to, cond, label_col="timeout_used_label")
    else:
        live_counts_to = {}
    prior_to = timeout_prior_alpha(cond["quarter"], cond["clock_bucket"], cond["hurry_up"], strength=float(prior_strength))
    post_to = posterior_mean(prior_to, live_counts_to)
    p_timeout_yes = post_to.get("YES", 0.0)

    # EP + EPA
    ep_now = ep_pre(cond, league_mix_cfb=float(league_mix_cfb))
    epa_last = epa_for_row(latest_labeled, league_mix_cfb=float(league_mix_cfb)) if latest_labeled is not None else None

    st.divider()
    st.markdown("### Summary Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P(RUN)", f"{deriv['p_run']:.2%}")
    m2.metric("P(PASS)", f"{deriv['p_pass']:.2%}")
    m3.metric("P(Pressure 5+)", f"{p_press_5p:.2%}")
    m4.metric("P(Timeout used)", f"{p_timeout_yes:.2%}")
    m5.metric("EP (pre-snap)", f"{ep_now:+.2f}")

    # 3rd->4th preview
    st.divider()
    st.markdown("### 4th-Down Decision Preview (right after 3rd-down FAIL)")
    show_preview = False
    preview_state4 = None

    if latest_labeled is not None:
        res = build_result_dict(latest_labeled)
        st_pre = build_state_pre_dict(latest_labeled)

        if int(st_pre["down"]) == 3 and (not res["first_down"]) and (not res["td"]) and res["turnover"] == "NONE":
            show_preview = True
            try:
                preview_state4 = next_state_from_result(st_pre, res)
            except Exception:
                preview_state4 = {"down": 4, "dist_bucket": st_pre["dist_bucket"], "field_zone": st_pre["field_zone"], "clock_bucket": st_pre["clock_bucket"], "goal_to_go": st_pre["goal_to_go"]}

    if show_preview and preview_state4 is not None and int(preview_state4.get("down", 0)) == 4:
        cond4 = {
            "pv_possession": latest.get("pv_possession", "PV_DEF"),
            "quarter": int(latest.get("quarter", 1)),
            "down": 4,
            "dist_bucket": str(preview_state4.get("dist_bucket", "UNK")),
            "field_zone": str(preview_state4.get("field_zone", "UNK")),
            "clock_bucket": str(preview_state4.get("clock_bucket", "OTHER")),
            "hurry_up": bool(latest.get("hurry_up", False)),
            "goal_to_go": bool(preview_state4.get("goal_to_go", False)),
        }
        in_range4 = fg_in_range(cond4["field_zone"], float(league_mix_cfb))

        df_4 = df_labeled[df_labeled["down"] == 4].copy() if (not df_labeled.empty and "down" in df_labeled.columns) else pd.DataFrame()
        if not df_4.empty:
            df_4["fourth_tri"] = df_4["call_type"].astype(str).map(map_4th_tri_from_call_type)
            live_counts_4tri = counts_from_live(df_4, cond4, label_col="fourth_tri")
        else:
            live_counts_4tri = {}

        prior_4tri = fourth_tri_prior(
            dist_bucket=cond4["dist_bucket"],
            field_zone=cond4["field_zone"],
            league_mix_cfb=float(league_mix_cfb),
            strength=float(prior_strength),
            fg_in_range=in_range4
        )
        post_4tri = posterior_mean(prior_4tri, live_counts_4tri)

        st.dataframe(pd.DataFrame([{
            "4th_dist_bucket": cond4["dist_bucket"],
            "4th_field_zone": cond4["field_zone"],
            "fg_in_range": in_range4,
            "p_GO": post_4tri.get("GO", 0.0),
            "p_PUNT": post_4tri.get("PUNT", 0.0),
            "p_FIELD_GOAL": post_4tri.get("FIELD_GOAL", 0.0),
            "p_NO_GO (derived)": 1.0 - post_4tri.get("GO", 0.0),
        }]), use_container_width=True)
    else:
        st.caption("Preview appears after you label a 3rd-down with first_down = False (and no TD/turnover).")

    # TD -> 2pt preview (ONLY after TD)
    st.divider()
    st.markdown("### 2pt vs Kick Preview (ONLY after a TD is labeled)")
    if latest_labeled is not None and bool(latest_labeled.get("td", False)):
        vc = df_labeled[df_labeled.get("two_pt_decision").notna()]["two_pt_decision"].value_counts().to_dict() if ("two_pt_decision" in df_labeled.columns) else {}
        prior_2 = {"KICK": 36 * float(prior_strength), "TWO": 4 * float(prior_strength)}
        post_2 = posterior_mean(prior_2, vc)
        st.dataframe(pd.DataFrame([{"p_KICK": post_2.get("KICK", 0.0), "p_TWO": post_2.get("TWO", 0.0)}]), use_container_width=True)
    else:
        st.caption("This section only shows after the most recent labeled play is marked TD = True.")

    # Full call-type posterior table
    st.divider()
    st.markdown("### Full Call-Type Posterior (all probabilities)")
    post_tbl = pd.DataFrame([{"call_type": k, "prob": float(v)} for k, v in post_call.items()]).sort_values("prob", ascending=False)
    st.dataframe(post_tbl, use_container_width=True, height=320)

    st.markdown("### Pass conditional")
    st.dataframe(pd.DataFrame([{
        "P(shot | pass)": deriv["p_shot_given_pass"],
        "P(screen | pass)": deriv["p_screen_given_pass"],
        "P(PA | pass)": deriv["p_pa_given_pass"],
        "P(quick | pass)": deriv["p_quick_given_pass"],
        "P(dropback | pass)": deriv["p_dropback_given_pass"],
    }]), use_container_width=True)

    # EPA table
    st.divider()
    st.markdown("### EPA (bucket-based but consistent)")
    if epa_last is not None:
        st.write(f"EPA(last labeled play): **{epa_last:+.3f}**")
    else:
        st.caption("Label first_down/td/turnover/yards_bucket on a play to compute EPA.")

    df_ep = df_live.copy()
    df_ep["epa"] = df_ep.apply(lambda r: epa_for_row(r.to_dict(), league_mix_cfb=float(league_mix_cfb)), axis=1)
    show = df_ep[df_ep["epa"].notna()].copy()
    if show.empty:
        st.info("No plays with enough result labels for EPA yet.")
    else:
        cols = [c for c in ["play_no","down","dist_bucket","field_zone","clock_bucket","call_type","first_down","td","turnover","yards_bucket","epa"] if c in show.columns]
        st.dataframe(show[cols].sort_values("play_no"), use_container_width=True, height=320)

    # ============================
    # NEW: COACH SUMMARY (4 sentences each side)
    # ============================
    st.divider()
    st.markdown("## Snap Summary (Coach-ready)")

    st.markdown("### PV Offense (4 sentences)")
    st.write(summarize_offense(df_labeled))

    st.markdown("### PV Defense (4 sentences)")
    st.write(summarize_defense(df_labeled))
