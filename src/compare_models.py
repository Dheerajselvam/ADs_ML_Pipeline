import joblib, json, pandas as pd, numpy as np
from eval_utils import offline_metrics, calibration_table, prepare_dicts
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

EVAL = "data/processed/eval.csv"
BASELINE_PIPE = "models/logistic_regression.pkl"
GBT_PIPE = "models/gbt_model.pkl"
DNN_PIPE = "models/dnn_model.pkl"
OUT_REPORT = "reports/compare_report.txt"
OUT_METRICS = "reports/all_offline_metrics.json"

eval_df = pd.read_csv(EVAL)
y_true = eval_df.clicked.values

models = {
    "logistic": BASELINE_PIPE,
    "gbt": GBT_PIPE,
    "dnn": DNN_PIPE
}

results = {}
for name, path in models.items():
    try:
        pipe = joblib.load(path)
        X_eval = prepare_dicts(eval_df)
        y_prob = pipe.predict_proba(X_eval)[:,1]
        metrics = offline_metrics(y_true, y_prob)
        results[name] = {
            "metrics": metrics,
            "y_prob": y_prob
        }
    except Exception as e:
        results[name] = {"error": str(e)}
        print("Error loading/predicting with", name, e)

lines = []
lines.append("MODEL COMPARISON REPORT\n")
for name, info in results.items():
    lines.append(f"--- {name} ---")
    if "error" in info:
        lines.append("ERROR: " + info["error"])
        continue
    m = info["metrics"]
    lines.append(json.dumps(m, indent=2))

    probs = info["y_prob"]
    df = pd.DataFrame({"prob": probs, "y": y_true})
    baseline_ctr = df.y.mean()
    top_decile = df.sort_values("prob", ascending=False).head(max(1,int(0.1*len(df))))
    lines.append(f"Baseline CTR: {baseline_ctr:.4f}, Top decile CTR: {top_decile.y.mean():.4f}")

    cal = calibration_table(y_true, probs, bins=10)
    lines.append("Calibration (first 3 buckets):")
    for b in cal[:3]:
        lines.append(str(b))

    preds = (probs >= 0.5).astype(int)
    lines.append("Confusion Matrix:\n" + str(confusion_matrix(y_true, preds)))

lines.append("\nERROR ANALYSIS: sample disagreement examples (logistic vs gbt)\n")
try:
    lprobs = results["logistic"]["y_prob"]
    gprobs = results["gbt"]["y_prob"]
    dprobs = results["dnn"]["y_prob"]
    idx = np.where((lprobs > 0.6) & (gprobs < 0.4))[0][:10]
    idx2 = np.where((gprobs > 0.4) & (dprobs < 0.4))[0][:10]
    lines.append(f"Indices where logistic very positive but gbt negative: {list(idx)}")
    lines.append(f"Indices where GBT very positive but DNN negative: {list(idx2)}")
except Exception:
    pass

Path(OUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_REPORT, "w") as f:
    f.write("\n".join(lines))

json.dump({k:v["metrics"] if "metrics" in v else v for k,v in results.items()},
          open(OUT_METRICS, "w"), indent=2)

print("✅ Comparison complete. Report at", OUT_REPORT)
