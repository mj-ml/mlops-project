from src.model import generate_features, split_train_test, train_model, register_the_best_model
from src.read_data import (
    read_weather,
    extract_weather,
    read_load,
    extract_load,
    create_dataframe,
)


def training_pipeline():
    df = read_weather()
    df_weather = extract_weather(df)
    df = read_load()
    df_load = extract_load(df)
    df_all = create_dataframe(df_weather, df_load)
    df_features = generate_features(df_all)
    df_train, df_test = split_train_test(df_features)
    train_model(df_train, df_test)


if __name__ == "__main__":
    training_pipeline()
    register_the_best_model(top_n=2)