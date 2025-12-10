import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from joblib import dump
import json


from eval_utils import offline_metrics, calibration_table
from utils import prepare_dicts


TRAIN = "data/processed/train.csv"
EVAL = "data/processed/eval.csv"
MODEL_OUT = "models/logistic_regression.pkl"
METRICS_OUT = "reports/offline_metrics.json"
CALIB_OUT = "reports/calibration_table.csv"


train = pd.read_csv(TRAIN)
eval_ = pd.read_csv(EVAL)


X_train = prepare_dicts(train)
y_train = train.clicked


X_eval = prepare_dicts(eval_)
y_eval = eval_.clicked


pipe = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("lr", LogisticRegression(max_iter=200))
        ])


pipe.fit(X_train, y_train)


y_prob = pipe.predict_proba(X_eval)[:, 1]
metrics = offline_metrics(y_eval, y_prob)
cal = calibration_table(y_eval, y_prob)


pd.DataFrame(cal).to_csv(CALIB_OUT, index=False)
json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
dump(pipe, MODEL_OUT)


print("✅ Baseline LR trained")
print(metrics)