from src.model import generate_features, split_train_test
from src.read_data import (
    read_load,
    read_weather,
    extract_load,
    extract_weather,
    create_dataframe,
)


def test_gen_features():
    df = read_weather()
    df_weather = extract_weather(df)
    df = read_load()
    df_load = extract_load(df)
    df_all = create_dataframe(df_weather, df_load)
    df_features = generate_features(df_all)

    assert df_features.empty is False

    df_train, df_test = split_train_test(df_features)

    assert df_train.empty is False
    assert df_test.empty is False
