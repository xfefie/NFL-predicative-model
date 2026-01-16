import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from storage import upsert_many

REQUIRED = ["session_id", "game_id", "play_no"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/import_tags_csv.py path/to/file.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    # Basic type normalization
    df["play_no"] = pd.to_numeric(df["play_no"], errors="coerce").fillna(0).astype(int)
    if "down" in df.columns:
        df["down"] = pd.to_numeric(df["down"], errors="coerce").fillna(0).astype(int)

    # Ensure ts exists
    if "ts" not in df.columns:
        df["ts"] = pd.Timestamp.utcnow().timestamp()

    upsert_many(df)
    print(f"Imported + upserted {len(df)} rows from {csv_path}")

if __name__ == "__main__":
    main()
