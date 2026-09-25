import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CONTINUOUS_FEATURES = ["temp", "atemp", "humidity", "windspeed"]
NOMINAL_FEATURES = ["season", "weather"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "dayofweek" in df.columns:
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    if "hour" in df.columns:
        df["is_rush_hour"] = df["hour"].isin([7, 8, 17, 18, 19]).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    # season and weather are unordered categories, not numeric scales,
    # so they are one-hot encoded rather than passed through as raw integers.
    # Continuous weather readings are standardized; everything else
    # (temporal flags, engineered binary features) passes through untouched.
    return ColumnTransformer(
        [
            ("scaler", StandardScaler(), CONTINUOUS_FEATURES),
            ("encoder", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
        ],
        remainder="passthrough",
    )
