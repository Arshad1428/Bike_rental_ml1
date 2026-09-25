import pandas as pd

from src.data_preprocessing import clean_data
from src.feature_engineering import add_derived_features
from src.train_test_split import get_train_test_split


def _sample_df():
    return pd.DataFrame(
        {
            "datetime": ["2011-01-01 00:00:00", "2011-01-01 01:00:00"],
            "season": [1, 1],
            "holiday": [0, 0],
            "workingday": [0, 0],
            "weather": [1, 1],
            "temp": [-5.0, 9.0],
            "atemp": [-2.0, 13.6],
            "humidity": [81, 80],
            "windspeed": [0, 0],
            "casual": [3, 8],
            "registered": [13, 32],
            "count": [16, 40],
        }
    )


def test_clean_data_extracts_temporal_features_and_clips_negatives():
    df = clean_data(_sample_df())
    assert {"hour", "month", "dayofweek", "year"}.issubset(df.columns)
    assert "datetime" not in df.columns
    assert (df["temp"] >= 0).all()


def test_add_derived_features_creates_flags():
    df = clean_data(_sample_df())
    df = add_derived_features(df)
    assert "is_weekend" in df.columns
    assert "is_rush_hour" in df.columns


def test_train_test_split_drops_leakage_columns():
    df = clean_data(_sample_df())
    df = add_derived_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.5, random_state=0)
    for col in ["casual", "registered", "count"]:
        assert col not in X_train.columns
        assert col not in X_test.columns
