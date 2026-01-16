import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import MODEL_PATH
from storage import load_events, load_historical
from model.features import featurize

def train_playtype_model(min_labeled: int = 200):
    """
    Trains on (historical + your tagged labeled plays), if historical exists.
    """
    df_live = load_events()
    df_hist = load_historical()

    frames = []
    if not df_hist.empty:
        frames.append(df_hist)
    if not df_live.empty:
        frames.append(df_live)

    if not frames:
        raise RuntimeError("No data found. Add historical_events.parquet or tag some plays.")

    df = __import__("pandas").concat(frames, ignore_index=True)
    df = df[df["outcome"].notna()].copy()

    if len(df) < min_labeled:
        raise RuntimeError(f"Need at least {min_labeled} labeled plays total. Currently: {len(df)}")

    X = featurize(df)
    y = df["outcome"].astype(str)

    cat = ["pv_possession", "clock_bucket", "dist_bucket", "field_zone",
           "opp_personnel", "opp_formation", "def_shell", "pressure"]
    bools = ["hurry_up"]
    num = ["quarter", "down"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("bool", "passthrough", bools),
            ("num", "passthrough", num),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=400)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)

    joblib.dump(pipe, MODEL_PATH)
    return acc

if __name__ == "__main__":
    acc = train_playtype_model()
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Holdout accuracy: {acc:.3f}")
