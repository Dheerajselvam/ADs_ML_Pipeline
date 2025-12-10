import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, precision_recall_fscore_support

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)  


def calibration_table(y_true, y_prob, bins=10):
    data = list(zip(y_true, y_prob))
    data.sort(key=lambda x: x[1])
    buckets = np.array_split(data, bins)


    rows = []
    for i, b in enumerate(buckets):
        ys, ps = zip(*b)
        rows.append({
            "bucket": i,
            "avg_pred": np.mean(ps),
            "avg_true": np.mean(ys),
            "count": len(b)
        })
    return rows


def offline_metrics(y_true, y_prob):
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob)
    }

def decile_lift(y_true, y_prob, decile=10):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p", ascending=False)
    n = len(df) // decile
    top = df.head(n)
    return {"top_decile_ctr": top.y.mean(), "overall_ctr": df.y.mean(), "n_top": len(top)}

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