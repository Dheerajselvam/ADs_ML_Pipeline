# src/rtb_simulator.py
"""
RTB Simulator:
- For each impression, compute bids submitted by advertisers.
- Simulate second-price auction.
- Use model pCTR (if available) or base_ctr.
- Budget management and simple pacing.
- Produce per-advertiser metrics and final CPT/RPM of served campaigns.
"""

import pandas as pd
import numpy as np
import os
from joblib import load
from utils import save_json
import random
random.seed(42)
np.random.seed(42)

DATA_IN = "data/raw/synthetic_ads.csv"
MODEL_PATH = "models/logistic_regression.pkl"
OUT_JSON = "reports/rtb_report.json"

# Simulation config
NUM_COMPETITORS = 3  # number of competing advertisers per impression
FLOOR = 0.01
AVG_REVENUE_PER_CLICK = 1.0  # to compute expected value (simple)
BUDGET_PER_ADVERTISER = 50.0  # starting budget per advertiser (small for demo)
PACING_HALF_LIFE = 0.5  # effect on bid as budget depletes (simple)

def try_load_model(path):
    if os.path.exists(path):
        try:
            return load(path)
        except Exception as e:
            print("Could not load model:", e)
    return None

def base_ctr(age, interest, creative, hour):
    ctr = 0.01
    if interest == "tech": ctr += 0.02
    if creative == "video": ctr += 0.015
    if age == "25-34": ctr += 0.01
    if 18 <= hour <= 22: ctr += 0.005
    return min(0.4, ctr)

def make_bid(advertiser_bid, pctr, budget_remaining, baseline_p=0.02):
    """
    Example pricing function: multiplier = pctr / baseline_p, clipped.
    Pacing: scale down bid when budget_remaining low.
    """
    if budget_remaining <= 0:
        return 0.0
    multiplier = min(3.0, pctr / (baseline_p + 1e-9))
    pacing = 1.0 - (1 - PACING_HALF_LIFE) * (1 - budget_remaining / BUDGET_PER_ADVERTISER)
    bid = advertiser_bid * multiplier * pacing
    return max(FLOOR, round(bid, 4))

def simulate_auction(req_row, model_pipe=None, advertisers_bids=None, advertiser_budgets=None):
    # get base pctr
    if model_pipe is not None:
        try:
            X = [{
                "age=" + str(req_row.age_bucket): 1,
                "geo=" + str(req_row.geo): 1,
                "interest=" + str(req_row.interests): 1,
                "creative=" + str(req_row.creative_type): 1,
                "device=" + str(req_row.device): 1,
                "hour": req_row.hour_of_day,
                "bid": float(req_row.bid)
            }]
            p = model_pipe.predict_proba(X)[:,1][0]
        except Exception:
            p = base_ctr(req_row.age_bucket, req_row.interests, req_row.creative_type, req_row.hour_of_day)
    else:
        p = base_ctr(req_row.age_bucket, req_row.interests, req_row.creative_type, req_row.hour_of_day)

    # Construct bids: include the ad's advertiser + NUM_COMPETITORS random advertisers
    # advertisers_bids is dict {adv_id: base_bid}, budgets dict adv->budget
    adv_pool = list(advertisers_bids.keys())
    # pick a set of bidders (ensure current advertiser participates)
    bidder_ids = random.sample(adv_pool, min(len(adv_pool), NUM_COMPETITORS))
    if req_row.advertiser_id not in bidder_ids:
        bidder_ids[0] = int(req_row.advertiser_id)  # force the original advertiser into the auction

    submission = []
    for adv in bidder_ids:
        base_bid = advertisers_bids.get(adv, 0.5)
        budget_remaining = advertiser_budgets.get(adv, 0.0)
        bid_amount = make_bid(base_bid, p, budget_remaining)
        submission.append((adv, bid_amount))
    # ensure at least two bids for second-price semantics (if not enough, pad with floor)
    if len(submission) < 2:
        submission.append((-1, FLOOR))
    # sort by bid descending
    submission_sorted = sorted(submission, key=lambda x: x[1], reverse=True)
    winner, winner_bid = submission_sorted[0]
    second_price = max(FLOOR, submission_sorted[1][1])
    price_paid = second_price
    # check budget
    if advertiser_budgets.get(winner, 0) < price_paid:
        # winner cannot pay -> treat as no win
        return {"won": False, "winner": None, "price": 0.0, "pctr": p}
    advertiser_budgets[winner] -= price_paid
    # determine click outcome
    click = 1 if random.random() < p else 0
    revenue = click * AVG_REVENUE_PER_CLICK
    return {"won": True, "winner": winner, "price": price_paid, "pctr": p, "click": click, "revenue": revenue}

def run_rtb_sim(data_path=DATA_IN, model_path=MODEL_PATH, out_json=OUT_JSON):
    df = pd.read_csv(data_path)
    model_pipe = try_load_model(model_path)
    # init advertiser bids and budgets
    advertisers = df["advertiser_id"].unique().tolist()
    advertisers_bids = {int(a): float(0.5 + (int(a) % 10) * 0.1) for a in advertisers}  # varied base bids
    advertiser_budgets = {int(a): float(BUDGET_PER_ADVERTISER) for a in advertisers}

    logs = []
    for _, row in df.iterrows():
        res = simulate_auction(row, model_pipe, advertisers_bids, advertiser_budgets)
        log = {
            "impression_id": row.impression_id,
            "ad_id": int(row.ad_id),
            "advertiser": int(row.advertiser_id),
            "pctr": res.get("pctr", None),
            "won": res.get("won", False),
            "price": res.get("price", 0.0),
            "click": res.get("click", 0),
            "revenue": res.get("revenue", 0.0)
        }
        logs.append(log)

    logs_df = pd.DataFrame(logs)
    # aggregate per advertiser
    agg = logs_df[logs_df["won"] == True].groupby("advertiser").agg(
        impressions=("won","count"),
        clicks=("click","sum"),
        revenue=("revenue","sum"),
        spend=("price","sum")
    ).reset_index()
    agg["ctr"] = agg["clicks"] / agg["impressions"]
    agg["rpm"] = (agg["revenue"] / agg["impressions"]) * 1000
    # global metrics
    total_impr = len(logs_df[logs_df["won"]==True])
    total_rev = logs_df["revenue"].sum()
    global_rpm = (total_rev / total_impr) * 1000 if total_impr>0 else 0.0
    global_ctr = logs_df["click"].sum() / total_impr if total_impr>0 else 0.0

    report = {
        "total_impressions_served": int(total_impr),
        "total_revenue": float(total_rev),
        "global_ctr": float(global_ctr),
        "global_rpm": float(global_rpm),
        "per_advertiser": agg.to_dict(orient="records")
    }
    save_json(out_json, report)
    print("RTB report saved to", out_json)
    return report

if __name__ == "__main__":
    r = run_rtb_sim()
    import pprint
    pprint.pprint(r)
