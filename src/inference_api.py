from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import uvicorn
import os

MODEL_PATH = "models/logistic_regression.pkl"  # choose model to serve

class RequestBody(BaseModel):
    user_features: dict
    ad_features: dict

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    else:
        print("Model not found; API will return base CTR.")

def prepare_dict_for_pipe(user_features, ad_features):
    d = {}
    # map keys expected by baseline pipeline naming convention; be conservative
    if "age_bucket" in user_features:
        d["age=" + str(user_features["age_bucket"])] = 1
    if "geo" in user_features:
        d["geo=" + str(user_features["geo"])] = 1
    if "interests" in user_features:
        d["interest=" + str(user_features["interests"])] = 1
    if "creative_type" in ad_features:
        d["creative=" + str(ad_features["creative_type"])] = 1
    if "device" in user_features:
        d["device=" + str(user_features["device"])] = 1
    d["hour"] = user_features.get("hour_of_day", 12)
    d["bid"] = float(ad_features.get("bid", 0.5))
    return [d]

@app.post("/predict_ctr")
def predict(req: RequestBody):
    if model is not None:
        X = prepare_dict_for_pipe(req.user_features, req.ad_features)
        try:
            p = model.predict_proba(X)[:,1][0]
        except Exception:
            # fallback if pipe expects vectorizer transform
            p = model.predict_proba(model.named_steps["vec"].transform(X))[:,1][0]
        return {"pctr": float(p)}
    else:
        # crude base ctr fallback
        bf = 0.02
        return {"pctr": bf}

if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=False)
