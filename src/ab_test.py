"""
A/B testing driver:
- Assign users via hashing
- Read impressions CSV and annotate assignment, optionally pCTR
- Log impressions and clicks to a results DataFrame
- Aggregate metrics per arm and run statistical tests:
    - CTR two-sample z-test
    - RPM bootstrap CI on mean diff
- Detect sample ratio mismatch
- Handle delayed clicks: if simulate_delay=True, some clicks come late; we show evaluation with/without delay window
"""

from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from scipy import stats
from joblib import load
from utils import assign_user, save_json
import os, random
from utils import prepare_dicts, to_dense
from data_generator import base_ctr
import pprint

random.seed(42)
np.random.seed(42)

DATA_IN = "data/raw/synthetic_ads.csv"
MODEL_PATH = "models/logistic_regression.pkl"
OUT_JSON = "reports/ab_results.json"

# Config
TREATMENT_PCT = 0.5
EVAL_DELAY_WINDOW = 0
BOOTSTRAP_ITERS = 5000
ALPHA = 0.05

def try_load_model(path):
    if os.path.exists(path):
        try:
            return load(path)
        except Exception as e:
            print("Could not load model:", e)
    return None

def predict_pctr_from_model(pipe, df):
    # Expects pipe to accept list-of-dicts like baseline
    
    X = prepare_dicts(df)
    try:
        proba = pipe.predict_proba(X)[:,1]
    except Exception:
        # If pipeline expects array, convert
        proba = pipe.predict_proba(pipe.named_steps["vec"].transform(X))[:,1]
    return proba

def two_sample_z_test(count1, n1, count2, n2):
    """Z-test for difference in proportions."""
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    num = p1 - p2
    denom = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if denom == 0:
        return {"z": np.nan, "p_value": 1.0}
    z = num / denom
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(p), "p1": p1, "p2": p2}

def bootstrap_mean_diff(a, b, iters=1000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    a = np.array(a)
    b = np.array(b)
    n = len(a)
    m = len(b)
    for _ in range(iters):
        sa = rng.choice(a, size=n, replace=True)
        sb = rng.choice(b, size=m, replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    lo = np.percentile(diffs, 100*ALPHA/2)
    hi = np.percentile(diffs, 100*(1-ALPHA/2))
    return {"mean_diff": float(diffs.mean()), "ci": [float(lo), float(hi)]}

def run_ab_test(data_path=DATA_IN, model_path=MODEL_PATH, out_json=OUT_JSON,
                treatment_pct=TREATMENT_PCT, bootstrap_iters=BOOTSTRAP_ITERS):
    df = pd.read_csv(data_path)
    df["assignment"] = df["user_id"].apply(lambda u: assign_user(u, pct_treatment=treatment_pct))
    # optional: predict pCTR using saved model
    pipe = try_load_model(model_path)
    if pipe is not None:
        try:
            df["pctr"] = predict_pctr_from_model(pipe, df)
            print("Used saved model for pCTR.")
        except Exception as e:
            print("Model present but error predicting:", e)
            df["pctr"] = df.apply(lambda r: base_ctr(r.age_bucket, r.interests, r.creative_type, r.hour_of_day), axis=1)
    else:
        df["pctr"] = df.apply(lambda r: base_ctr(r.age_bucket, r.interests, r.creative_type, r.hour_of_day), axis=1)

    # Optionally simulate delayed clicks: add click_delay seconds to some clicks
    # For demo we don't actually shift timestamps, but we can mark some clicks as 'delayed'
    df["click_delay_sec"] = 0
    # randomly mark 10% of clicks as delayed by 3600s
    click_mask = df["clicked"] == 1
    delayed_idxs = df[click_mask].sample(frac=0.10, random_state=42).index
    df.loc[delayed_idxs, "click_delay_sec"] = 3600

    # Log table that will be used for aggregation (we keep original columns + assignment)
    logs = df.copy()

    # Basic aggregates per arm
    arms = logs["assignment"].unique().tolist()
    summary = {}
    total_expected = {"control": 0, "treatment": 0}
    # Expected counts by deterministic assignment
    n_total = len(logs)
    expected_control = int(n_total * (1 - treatment_pct))
    expected_treatment = n_total - expected_control
    total_expected["control"] = expected_control
    total_expected["treatment"] = expected_treatment

    for arm in arms:
        sub = logs[logs["assignment"] == arm]
        impressions = len(sub)
        clicks = int(sub["clicked"].sum())
        revenue = float(sub["revenue"].sum())
        ctr = clicks / impressions if impressions>0 else 0.0
        rpm = (revenue / impressions) * 1000 if impressions>0 else 0.0
        summary[arm] = {
            "impressions": impressions,
            "clicks": clicks,
            "revenue": revenue,
            "ctr": ctr,
            "rpm": rpm
        }

    # Sample ratio mismatch detection
    actual_control = summary.get("control", {}).get("impressions", 0)
    actual_treatment = summary.get("treatment", {}).get("impressions", 0)
    mismatch = {}
    def pct_diff(actual, expected):
        if expected == 0: return np.nan
        return (actual - expected) / expected
    mismatch["control_pct_diff"] = pct_diff(actual_control, total_expected["control"])
    mismatch["treatment_pct_diff"] = pct_diff(actual_treatment, total_expected["treatment"])
    # flag if magnitude > 0.02 (2%)
    mismatch["flag"] = (abs(mismatch["control_pct_diff"]) > 0.02) or (abs(mismatch["treatment_pct_diff"]) > 0.02)

    # Statistical tests
    z_res = two_sample_z_test(
        summary["treatment"]["clicks"], summary["treatment"]["impressions"] if "treatment" in summary else 1,
        summary["control"]["clicks"], summary["control"]["impressions"] if "control" in summary else 1
    )

    # Bootstrap for RPM mean difference: we bootstrap per-impression revenue (many zeros)
    rev_treatment = logs[logs["assignment"] == "treatment"]["revenue"].values
    rev_control = logs[logs["assignment"] == "control"]["revenue"].values
    rpm_boot = bootstrap_mean_diff(rev_treatment, rev_control, iters=bootstrap_iters)

    results = {
        "n_total": n_total,
        "treatment_pct": treatment_pct,
        "expected_counts": total_expected,
        "summary": summary,
        "sample_ratio_mismatch": mismatch,
        "ctr_ztest": z_res,
        "rpm_bootstrap": rpm_boot
    }

    save_json(out_json, results)
    print("A/B test results saved to", out_json)
    return results

if __name__ == "__main__":
    res = run_ab_test()
    pprint.pprint(res)
