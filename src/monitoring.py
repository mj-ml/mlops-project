import mlflow
from sklearn.metrics import mean_absolute_percentage_error

from settings import MLFLOW_URL, RF_BEST_MODEL, THE_BEST_MODEL_VERSION, EXPERIMENT_NAME
from src.pipeline import training_pipeline


mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(
    log_models=True,
    log_datasets=False,
)
from src.model import (
    generate_features,
    split_train_test,
)
from src.read_data import (
    read_weather,
    extract_weather,
    read_load,
    extract_load,
    create_dataframe,
)
from model import target_variable, time_variable_to_drop


def fetch_model_monitoring(df_monitoring):
    model_name = RF_BEST_MODEL
    alias = THE_BEST_MODEL_VERSION
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")
    X_val = df_monitoring.drop(columns=[target_variable, time_variable_to_drop])
    y_val = df_monitoring[target_variable]

    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    return mape


def monitoring():
    df = read_weather()
    df_weather = extract_weather(df)
    df = read_load()
    df_load = extract_load(df)
    df_all = create_dataframe(df_weather, df_load)
    df_features = generate_features(df_all)
    _, df_test = split_train_test(df_features)
    mape = fetch_model_monitoring(df_test)
    if mape > 0.1:
        print("updating models required - rerun the pipeline")
        training_pipeline()
    else:
        print("no models need to be trained")

if __name__ == "__main__":
    monitoring()