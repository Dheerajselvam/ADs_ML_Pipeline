import hashlib
import json
import numpy as np

def hash_to_int(x, salt="ads_experiment"):
    h = hashlib.sha256(f"{x}-{salt}".encode("utf-8")).hexdigest()
    return int(h, 16)

def assign_user(user_id, pct_treatment=0.5, salt="ads_experiment"):
    """Deterministically assign user to treatment/control."""
    val = hash_to_int(user_id, salt) % 10000 / 10000.0  # 0..1
    return "treatment" if val < pct_treatment else "control"

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def to_dense(x):
    return x.toarray()

def prepare_dicts(df):
    features = []
    for _, r in df.iterrows():
        d = {
            "age=" + str(r.age_bucket): 1,
            "geo=" + str(r.geo): 1,
            "interest=" + str(r.interests): 1,
            "creative=" + str(r.creative_type): 1,
            "device=" + str(r.device): 1,
            "hour": float(r.hour_of_day),
            "bid": float(r.bid)
        }
        features.append(d)
    return features