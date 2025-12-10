import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = "../data/raw/synthetic_ads.csv"
TRAIN_OUT = "../data/processed/train.csv"
EVAL_OUT = "../data/processed/eval.csv"


CATEGORICAL = [
"age_bucket", "geo", "interests",
"creative_type", "device"
]
NUMERIC = ["hour_of_day", "bid"]




def featurize(df):
    df = df.copy()
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    return df




df = pd.read_csv(INPUT)
df = featurize(df)


train, eval_ = train_test_split(
df, test_size=0.2, random_state=42,
stratify=df["clicked"]
)


train.to_csv(TRAIN_OUT, index=False)
eval_.to_csv(EVAL_OUT, index=False)


print(f"✅ Train rows: {len(train)}, Eval rows: {len(eval_)}")