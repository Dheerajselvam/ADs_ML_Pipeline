import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from eval_utils import offline_metrics, calibration_table
import json
from utils import prepare_dicts


def dnn_main():
    TRAIN = "data/processed/train.csv"
    EVAL = "data/processed/eval.csv"
    BASELINE_PIPE = "models/logistic.pkl"
    DNN_OUT = "models/dnn.pkl"
    METRICS_OUT = "reports/dnn_offline_metrics.json"
    CALIB_OUT = "reports/dnn_calibration_table.csv"

    train = pd.read_csv(TRAIN)
    eval_ = pd.read_csv(EVAL)

    X_train_dicts = prepare_dicts(train)
    y_train = train.clicked.values
    X_eval_dicts = prepare_dicts(eval_)
    y_eval = eval_.clicked.values

    vec = None
    try:
        baseline = joblib.load(BASELINE_PIPE)
        vec = baseline.named_steps["vec"]
        print("✅ Reused DictVectorizer from baseline.")
    except Exception as e:
        print("⚠️ Could not reuse baseline pipeline; creating new DictVectorizer.", e)
        vec = DictVectorizer(sparse=True)

    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)
    pipe = Pipeline([("vec", vec), ("mlp", mlp)])

    pipe.fit(X_train_dicts, y_train)

    y_prob = pipe.predict_proba(X_eval_dicts)[:, 1]
    metrics = offline_metrics(y_eval, y_prob)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)

    pd.DataFrame(calibration_table(y_eval, y_prob)).to_csv(CALIB_OUT, index=False)
    joblib.dump(pipe, DNN_OUT)

    print("✅ DNN (MLP) trained. Metrics:", metrics)

if __name__ == "__main__":
    dnn_main()