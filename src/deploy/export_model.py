import os, json, shutil
from datetime import datetime

def export_model(model_path, out_dir="deploy"):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(model_path, os.path.join(out_dir, os.path.basename(model_path)))
    meta = {
        "model": os.path.basename(model_path),
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Export from dev pipeline. Include vec + metadata in production."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Exported model to", out_dir)
    return out_dir

if __name__ == "__main__":
    export_model("models/logistic_regression.pkl")
