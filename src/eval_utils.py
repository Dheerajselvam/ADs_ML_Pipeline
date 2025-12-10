import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

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