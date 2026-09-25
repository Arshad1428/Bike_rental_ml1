import os
import pandas as pd

RAW_DATA_PATH = "data/raw/bike_rental.csv"
PROCESSED_DATA_PATH = "data/processed/processed_bike_data.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Break the timestamp into explicit temporal signals
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["year"] = df["datetime"].dt.year
        df = df.drop(columns=["datetime"])

    # Guard against invalid negative sensor readings
    for col in ["temp", "atemp", "humidity", "windspeed"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def run_preprocessing(raw_path: str = RAW_DATA_PATH, processed_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = clean_data(df)
    save_processed(df, processed_path)
    return df


if __name__ == "__main__":
    df = run_preprocessing()
    print(f"Processed dataset shape: {df.shape}")
    print(f"Saved cleaned data to: {PROCESSED_DATA_PATH}")
