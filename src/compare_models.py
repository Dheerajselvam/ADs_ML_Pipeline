import joblib, json, pandas as pd, numpy as np
from eval_utils import offline_metrics, calibration_table
from sklearn.metrics import confusion_matrix
from utils import prepare_dicts
from pathlib import Path
import shutil

def compare_and_choose():
    EVAL = "data/processed/eval.csv"
    BASELINE_PIPE = "models/logistic.pkl"
    GBT_PIPE = "models/gbt.pkl"
    DNN_PIPE = "models/dnn.pkl"

    OUT_REPORT = "reports/compare_report.txt"
    OUT_METRICS = "reports/all_offline_metrics.json"
    OUT_BEST = "models/best_model.pkl"
    OUT_BEST_INFO = "reports/best_model_info.json"

    eval_df = pd.read_csv(EVAL)
    y_true = eval_df.clicked.values

    models = {
        "logistic": BASELINE_PIPE,
        "gbt": GBT_PIPE,
        "dnn": DNN_PIPE
    }

    results = {}

    # ---- Evaluate Each Model ----
    for name, path in models.items():
        try:
            pipe = joblib.load(path)
            X_eval = prepare_dicts(eval_df)
            y_prob = pipe.predict_proba(X_eval)[:, 1]

            metrics = offline_metrics(y_true, y_prob)

            results[name] = {
                "metrics": metrics,
                "y_prob": y_prob,
                "path": path
            }

        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"❌ Error loading {name}: {e}")

    # ---- Choose Best Model (Using AUROC) ----
    valid_models = {
        name: info
        for name, info in results.items()
        if "metrics" in info
    }

    if not valid_models:
        raise RuntimeError("No valid models available for comparison!")

    best_name = max(valid_models, key=lambda m: valid_models[m]["metrics"]["auc"])
    best_info = valid_models[best_name]

    # Copy best model to production path
    shutil.copy(best_info["path"], OUT_BEST)

    # ---- Write Human-Readable Report ----
    lines = []
    lines.append("MODEL COMPARISON REPORT\n")
    lines.append(f"BEST MODEL: {best_name.upper()} (AUC={best_info['metrics']['auc']:.4f})\n")

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
        top_decile = df.sort_values("prob", ascending=False).head(max(1, int(0.1 * len(df))))
        lines.append(f"Baseline CTR: {baseline_ctr:.4f}, "
                     f"Top-decile CTR: {top_decile.y.mean():.4f}")

        # Calibration
        cal = calibration_table(y_true, probs, bins=10)
        lines.append("Calibration (first 3 buckets):")
        for b in cal[:3]:
            lines.append(str(b))

        preds = (probs >= 0.5).astype(int)
        lines.append("Confusion Matrix:\n" + str(confusion_matrix(y_true, preds)))

    # ---- Error Analysis ----
    lines.append("\n\nERROR ANALYSIS\n")
    try:
        lprobs = results["logistic"]["y_prob"]
        gprobs = results["gbt"]["y_prob"]
        dprobs = results["dnn"]["y_prob"]

        idx1 = np.where((lprobs > 0.6) & (gprobs < 0.4))[0][:10]
        idx2 = np.where((gprobs > 0.4) & (dprobs < 0.4))[0][:10]

        lines.append(f"Disagreement logistic>gbt: {list(idx1)}")
        lines.append(f"Disagreement gbt>dnn: {list(idx2)}")
    except Exception:
        pass

    Path(OUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines))

    # ---- Save All Metrics + Best Model ----
    json.dump(
        {k: v["metrics"] if "metrics" in v else v for k, v in results.items()},
        open(OUT_METRICS, "w"), indent=2
    )

    json.dump(
        {
            "best_model": best_name.lower(),
            "path": OUT_BEST,
            "metrics": best_info["metrics"]
        },
        open(OUT_BEST_INFO, "w"), indent=2
    )

    print(f"✅ Comparison complete. Best model = {best_name.lower()}")
    print(f"📦 Saved to {OUT_BEST}")
    print("📄 Report at:", OUT_REPORT)

    return best_name.lower()

if __name__ == "__main__":
    compare_and_choose()
