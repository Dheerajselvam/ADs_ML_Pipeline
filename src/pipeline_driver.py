import os, pprint
from data_generator import data_gen_main  # if you use a main() in generator; else call script directly
from featurize import featurize_main
from train_baseline_lr import lr_main
from train_gbt import gbt_main
from train_dnn import dnn_main
from compare_models import compare_and_choose
from ab_test import run_ab_test
from rtb_simulator import run_rtb_sim
from report_generator import generate_final_report
from monitoring import run_monitoring
from dashboard import dashboard_main
import subprocess

def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] Step {func.__name__} failed: {e}")
        return None

def main():
    print("=== PIPELINE DRIVER START ===")
    # # 1. Data generation (only if missing)
    # if not os.path.exists("data/raw/synthetic_ads.csv"):
    #     print("Generating synthetic data...")
    #     safe_run(data_gen_main)
    # else:
    #     print("Found existing synthetic data, skipping generation.")

    # # 2. Featurize + splits
    # print("Featurizing + train/eval split...")
    # safe_run(featurize_main)

    # # 3. Train models
    # print("Training baseline LR...")
    # safe_run(lr_main)
    # print("Training GBT...")
    # safe_run(gbt_main)
    # print("Training DNN...")
    # safe_run(dnn_main)

    # # 4. Compare models and pick best
    # print("Comparing models...")
    # best = safe_run(compare_and_choose)
    # print("Best model:", best)

    # # 5. Run A/B test (baseline vs best)
    # print("Running A/B test (baseline vs candidate)...")
    # ab_res = safe_run(run_ab_test, model_path=f"models/{best}" if best else None)

    # # 6. Run RTB simulation using chosen model
    # print("Running RTB simulation using best model...")
    # rtb_res = safe_run(run_rtb_sim, model_path=f"models/{best}" if best else None)

    # # 7. Run monitoring
    # print("Running monitoring checks...")
    # monitor_res = safe_run(run_monitoring)

    # # 8. Final report
    # print("Generating final report...")
    # final_rep = safe_run(generate_final_report)


    print("=== PIPELINE DRIVER DONE ===")

    subprocess.run(["streamlit", "run", "src\dashboard.py"])

if __name__ == "__main__":
    main()
