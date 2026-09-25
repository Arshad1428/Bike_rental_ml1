import pandas as pd

from src.feature_engineering import add_derived_features
from src.save_model import load_model, DEFAULT_MODEL_PATH


def build_input_row(
    dt: pd.Timestamp,
    season: int,
    holiday: int,
    workingday: int,
    weather: int,
    temp: float,
    atemp: float,
    humidity: float,
    windspeed: float,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "season": season,
                "holiday": holiday,
                "workingday": workingday,
                "weather": weather,
                "temp": max(temp, 0),
                "atemp": max(atemp, 0),
                "humidity": max(humidity, 0),
                "windspeed": max(windspeed, 0),
                "hour": dt.hour,
                "month": dt.month,
                "dayofweek": dt.dayofweek,
                "year": dt.year,
            }
        ]
    )
    return add_derived_features(row)


def predict_rentals(model, input_row: pd.DataFrame) -> float:
    prediction = model.predict(input_row)[0]
    return max(0.0, prediction)


if __name__ == "__main__":
    model = load_model(DEFAULT_MODEL_PATH)
    sample = build_input_row(
        dt=pd.Timestamp("2012-06-15 08:00:00"),
        season=2,
        holiday=0,
        workingday=1,
        weather=1,
        temp=22.0,
        atemp=25.0,
        humidity=55,
        windspeed=10,
    )
    print(f"Predicted rentals: {predict_rentals(model, sample):.0f}")
