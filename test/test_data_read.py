from src.read_data import read_load, read_weather, extract_load, extract_weather, create_dataframe


def test_load_data_read():
    df = read_load()
    assert df.empty is False


def test_weather_data_read():
    df = read_weather()
    assert df.empty is False


def test_extract_load():
    df = read_load()
    df_load = extract_load(df)
    assert df_load.empty is False

def test_extract_weather():
    df = read_weather()
    df_weather = extract_weather(df)
    assert df_weather.empty is False


def test_create_dataframe():
    df = read_weather()
    df_weather = extract_weather(df)
    df = read_load()
    df_load = extract_load(df)
    df_all = create_dataframe(df_weather, df_load)
    assert df_all.empty is False
