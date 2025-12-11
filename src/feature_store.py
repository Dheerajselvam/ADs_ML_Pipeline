import os, json
from typing import Dict

STORE_DIR = "data/feature_store"

def init_store():
    os.makedirs(STORE_DIR, exist_ok=True)

def path_for(user_id):
    return os.path.join(STORE_DIR, f"user_{user_id}.json")

def write_features(user_id: int, features: Dict):
    init_store()
    p = path_for(user_id)
    with open(p, "w") as f:
        json.dump(features, f)

def read_features(user_id: int):
    p = path_for(user_id)
    if not os.path.exists(p):
        return None
    return json.load(open(p))

def backfill_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        feats = {
            "age_bucket": r.get("age_bucket"),
            "geo": r.get("geo"),
            "interests": r.get("interests"),
            "last_seen": r.get("timestamp")
        }
        write_features(int(r.user_id), feats)

if __name__ == "__main__":
    print("Feature store module; call write_features/read_features/backfill_from_csv")
