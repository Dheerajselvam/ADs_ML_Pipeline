# src/monitoring.py
import json, os
import numpy as np
import pandas as pd
from scipy.stats import entropy

RAW = "data/raw/synthetic_ads.csv"
OUT = "reports/monitoring.json"
BASELINE_SAMPLE = "data/processed/train.csv"

def kl_divergence(p, q, eps=1e-9):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))

def distribution(series, bins=10):
    vals, edges = np.histogram(series, bins=bins)
    return vals.astype(float)

def run_monitoring():
    # simple checks: age_bucket distribution drift; pctr distribution if present; CTR change
    base = pd.read_csv(BASELINE_SAMPLE) if os.path.exists(BASELINE_SAMPLE) else None
    curr = pd.read_csv(RAW) if os.path.exists(RAW) else None

    report = {"alerts": [], "checks": {}}
    if base is not None and curr is not None:
        # age bucket drift
        base_dist = base['age_bucket'].value_counts(normalize=True).to_dict()
        curr_dist = curr['age_bucket'].value_counts(normalize=True).to_dict()
        # compute simple L1 diff
        age_keys = set(base_dist) | set(curr_dist)
        l1 = sum(abs(base_dist.get(k,0)-curr_dist.get(k,0)) for k in age_keys)
        report["checks"]["age_l1"] = l1
        if l1 > 0.2:
            report["alerts"].append(f"Age distribution L1 > 0.2 ({l1:.3f})")

        # CTR change
        base_ctr = base['clicked'].mean() if 'clicked' in base else None
        curr_ctr = curr['clicked'].mean() if 'clicked' in curr else None
        report["checks"]["base_ctr"] = base_ctr
        report["checks"]["curr_ctr"] = curr_ctr
        if base_ctr is not None and curr_ctr is not None:
            if curr_ctr < base_ctr * 0.7:
                report["alerts"].append("CTR dropped >30% vs baseline")

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print("Monitoring report written to", OUT)
    return report

if __name__ == "__main__":
    run_monitoring()
