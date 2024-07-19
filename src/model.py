from typing import Tuple

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

EXPERIMENT_NAME = "energy-prediction"

# mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(
    log_models=True,
    log_datasets=False,
)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df.dropna(inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["temp"] = df["temp"] - 273.15
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.dayofyear
    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param df:
    :return:
    """
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=False,
    )
    return train_df, test_df


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 10, 2)),
        "random_state": 42,
    }
    target_variable = "total load actual"
    time_variable_to_drop = "time"

    X_train = train_df.drop(columns=[target_variable, time_variable_to_drop])
    y_train = train_df[target_variable]

    X_val = test_df.drop(columns=[target_variable, time_variable_to_drop])
    y_val = test_df[target_variable]

    def objective(params):
        with mlflow.start_run():
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            mlflow.log_metric("mape", mape)

        return {"loss": rmse, "status": STATUS_OK}

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials(),
        rstate=rstate,
    )


def register_the_best_model(top_n=2):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.mape ASC"],
    )[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")