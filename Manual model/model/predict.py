import joblib
import pandas as pd
from config import MODEL_PATH
from model.features import featurize

_cached = None

def load_model():
    global _cached
    if _cached is None:
        _cached = joblib.load(MODEL_PATH)
    return _cached

def predict_proba(df_one_play: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    X = featurize(df_one_play)
    probs = model.predict_proba(X)
    classes = model.classes_
    return pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
