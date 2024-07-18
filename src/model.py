from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["temp"] = df["temp"] - 273.15
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.dayofyear
    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=False,
    )
    return train_df, test_df
