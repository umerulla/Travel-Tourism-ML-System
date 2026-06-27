"""
Apache Airflow DAG: flight_price_training_pipeline
--------------------------------------------------
A REAL training pipeline (not print() stubs). Each task does actual work and
passes artifacts to the next task via XCom / the shared data directory.

Pipeline:
    load_data -> preprocess -> train -> evaluate -> register_with_mlflow

To deploy:
    1. Copy this file into  $AIRFLOW_HOME/dags/
    2. Place flights.csv in  $AIRFLOW_HOME/data/   (or set FLIGHTS_CSV env var)
    3. `airflow dags list`  ->  the DAG id should appear
    4. `airflow dags test flight_price_training_pipeline 2024-01-01`

This module is import-safe: heavy libraries are imported INSIDE the task
callables so the scheduler can parse the file quickly.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
DATA_DIR = os.environ.get("FLIGHT_DATA_DIR", os.path.join(AIRFLOW_HOME, "data"))
ARTIFACT_DIR = os.path.join(AIRFLOW_HOME, "flight_artifacts")
FLIGHTS_CSV = os.environ.get("FLIGHTS_CSV", os.path.join(DATA_DIR, "flights.csv"))

FEATURE_ORDER = ["from", "to", "flightType", "agency", "time", "distance"]
CATEGORICAL = ["from", "to", "flightType", "agency"]
TARGET = "price"

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def load_data(**context):
    import pandas as pd
    df = pd.read_csv(FLIGHTS_CSV)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    raw_path = os.path.join(ARTIFACT_DIR, "raw.parquet")
    df.to_parquet(raw_path)
    context["ti"].xcom_push(key="raw_path", value=raw_path)
    print(f"Loaded {len(df):,} rows -> {raw_path}")


def preprocess(**context):
    import pandas as pd, joblib
    from sklearn.preprocessing import LabelEncoder
    raw_path = context["ti"].xcom_pull(key="raw_path")
    df = pd.read_parquet(raw_path)[FEATURE_ORDER + [TARGET]].copy()
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(ARTIFACT_DIR, f"{col}_label_encoder.joblib"))
    proc_path = os.path.join(ARTIFACT_DIR, "processed.parquet")
    df.to_parquet(proc_path)
    context["ti"].xcom_push(key="proc_path", value=proc_path)
    print(f"Encoded {CATEGORICAL}; processed -> {proc_path}")


def train(**context):
    import pandas as pd, joblib
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    proc_path = context["ti"].xcom_pull(key="proc_path")
    df = pd.read_parquet(proc_path)
    X, y = df[FEATURE_ORDER], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_tr, y_tr)
    model_path = os.path.join(ARTIFACT_DIR, "xgb_flight_regressor.joblib")
    joblib.dump(model, model_path)
    pd.concat([X_te, y_te], axis=1).to_parquet(os.path.join(ARTIFACT_DIR, "test.parquet"))
    context["ti"].xcom_push(key="model_path", value=model_path)
    print(f"Trained model -> {model_path}")


def evaluate(**context):
    import pandas as pd, joblib, numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    model = joblib.load(context["ti"].xcom_pull(key="model_path"))
    test = pd.read_parquet(os.path.join(ARTIFACT_DIR, "test.parquet"))
    X_te, y_te = test[FEATURE_ORDER], test[TARGET]
    pred = model.predict(X_te)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
    }
    context["ti"].xcom_push(key="metrics", value=metrics)
    print(f"Evaluation metrics: {metrics}")


def register_with_mlflow(**context):
    import joblib
    try:
        import mlflow, mlflow.sklearn
    except ImportError:
        print("mlflow not installed in this environment; skipping registration.")
        return
    metrics = context["ti"].xcom_pull(key="metrics")
    model = joblib.load(context["ti"].xcom_pull(key="model_path"))
    mlflow.set_experiment("FlightFarePredictionExp")
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoostRegressor")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "xgb_flight_fare_model")
    print(f"Logged run to MLflow with metrics {metrics}")


with DAG(
    dag_id="flight_price_training_pipeline",
    default_args=default_args,
    description="End-to-end training pipeline for the flight price model",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["flight", "regression", "mlops"],
) as dag:

    t_load = PythonOperator(task_id="load_data", python_callable=load_data)
    t_prep = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_eval = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_reg = PythonOperator(task_id="register_with_mlflow", python_callable=register_with_mlflow)

    t_load >> t_prep >> t_train >> t_eval >> t_reg
