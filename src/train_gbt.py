import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from utils import prepare_dicts, to_dense
import json

TRAIN = "data/processed/train.csv"
EVAL = "data/processed/eval.csv"
BASELINE_PIPE = "models/logistic_regression.pkl"
GBT_OUT = "models/gbt_model.pkl"
METRICS_OUT = "reports/gbt_offline_metrics.json"
CALIB_OUT = "reports/gbt_calibration_table.csv"

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
    print("✅ Reused DictVectorizer from baseline pipeline.")
except Exception as e:
    print("⚠️ Could not reuse baseline pipeline, creating new DictVectorizer. Error:", e)
    vec = DictVectorizer(sparse=True)

gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1)
to_dense_transform = FunctionTransformer(to_dense, accept_sparse=True)
pipe = Pipeline([
            ("vec", vec),             
            ("to_dense", to_dense_transform),       
            ("gbt", gbt),
        ])

pipe.fit(X_train_dicts, y_train)

y_prob = pipe.predict_proba(X_eval_dicts)[:, 1]
metrics = offline_metrics(y_eval, y_prob)
json.dump(metrics, open(METRICS_OUT, "w"), indent=2)


pd.DataFrame(calibration_table(y_eval, y_prob)).to_csv(CALIB_OUT, index=False)
joblib.dump(pipe, GBT_OUT)

print("✅ GBT trained. Metrics:", metrics)
