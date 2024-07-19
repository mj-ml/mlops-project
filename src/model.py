from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope


# mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("energy-prediction")
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
        "max_depth": scope.int(hp.quniform("max_depth", 1, 10, 1)),
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

        return {'loss': rmse, 'status': STATUS_OK}



    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
        rstate=rstate
    )

