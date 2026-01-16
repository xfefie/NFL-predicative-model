import pandas as pd
from config import DB_PATH

KEY_COLS = ["session_id", "game_id", "play_no"]

def load_events() -> pd.DataFrame:
    if DB_PATH.exists():
        return pd.read_parquet(DB_PATH)
    return pd.DataFrame()

def upsert_event(event_dict: dict) -> None:
    df = load_events()
    new = pd.DataFrame([event_dict])

    if df.empty:
        out = new
    else:
        out = pd.concat([df, new], ignore_index=True)
        out = out.drop_duplicates(subset=KEY_COLS, keep="last")

    out.to_parquet(DB_PATH, index=False)

def upsert_many(df_new: pd.DataFrame) -> None:
    if df_new is None or df_new.empty:
        return
    for c in KEY_COLS:
        if c not in df_new.columns:
            raise ValueError(f"Missing required column: {c}")

    df = load_events()
    if df.empty:
        out = df_new.copy()
    else:
        out = pd.concat([df, df_new], ignore_index=True)
        out = out.drop_duplicates(subset=KEY_COLS, keep="last")

    out.to_parquet(DB_PATH, index=False)

def list_session_game(df: pd.DataFrame, session_id: str, game_id: str) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[(df["session_id"] == session_id) & (df["game_id"] == game_id)].copy()
    if sub.empty:
        return sub
    return sub.sort_values(["play_no", "ts"]).drop_duplicates(subset=["play_no"], keep="last")

def get_play(df: pd.DataFrame, session_id: str, game_id: str, play_no: int) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[
        (df["session_id"] == session_id) &
        (df["game_id"] == game_id) &
        (df["play_no"] == play_no)
    ]
    return sub.tail(1)
