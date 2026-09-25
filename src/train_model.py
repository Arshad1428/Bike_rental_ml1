import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import add_derived_features, build_preprocessor
from src.train_test_split import get_train_test_split


def evaluate_model(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.3f}\n")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def train_linear_model():
    df = run_preprocessing()
    df = add_derived_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model("Linear Regression Baseline", y_test, pipeline.predict(X_test))

    return pipeline, metrics


if __name__ == "__main__":
    train_linear_model()
