import os
import joblib

DEFAULT_MODEL_PATH = "models/random_forest_model.pkl"


def save_model(model, path: str = DEFAULT_MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str = DEFAULT_MODEL_PATH):
    return joblib.load(path)
