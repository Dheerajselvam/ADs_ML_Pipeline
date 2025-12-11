"""
A skeleton that shows what an Airflow DAG would execute.
(You don't need to run Airflow to use this; it's documentation + example)
"""
from datetime import timedelta

DAG_SPEC = {
    "dag_id": "ads_ml_daily",
    "schedule": "0 2 * * *",  # daily at 02:00
    "tasks": [
        {"id": "ingest", "cmd": "python src/data_generator.py"},
        {"id": "featurize", "cmd": "python src/featurize.py"},
        {"id": "train", "cmd": "python src/pipeline_driver.py --train-only"},
        {"id": "evaluate", "cmd": "python src/compare_models.py"},
        {"id": "monitor", "cmd": "python src/monitoring.py"},
        {"id": "report", "cmd": "python src/report_generator.py"}
    ],
    "retries": 1,
    "retry_delay_minutes": 10,
    "sla_minutes": 240
}

def print_dag():
    import json
    print(json.dumps(DAG_SPEC, indent=2))

if __name__ == "__main__":
    print_dag()
