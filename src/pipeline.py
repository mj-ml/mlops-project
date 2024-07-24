from src.model import register_the_best_model, fetch_model_predict, monitoring


def preprocessing():
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
    df = read_weather()
    df_weather = extract_weather(df)
    df = read_load()
    df_load = extract_load(df)
    df_all = create_dataframe(df_weather, df_load)
    df_features = generate_features(df_all)
    df_train, df_test = split_train_test(df_features)
    return df_train, df_test


def training_pipeline():
    from src.model import (
        train_model,
    )
    df_train, df_test = preprocessing()
    train_model(df_train, df_test)


def monitoring_pipeline():
    _, df_test = preprocessing()
    mape = monitoring(df_test)
    if mape > 0.1:
        print("updating models required - rerun the pipeline")
        training_pipeline()
    else:
        print("no models need to be trained")


if __name__ == "__main__":
    training_pipeline()
    register_the_best_model(top_n=2)
    test_result = fetch_model_predict(
        20,
        10,
        10,
    )
    print(test_result)
