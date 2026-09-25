from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import add_derived_features, build_preprocessor
from src.train_test_split import get_train_test_split
from src.train_model import evaluate_model

PARAM_DIST = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [4, 8, 12, None],
}


def train_random_forest():
    df = run_preprocessing()
    df = add_derived_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )

    print("Running RandomizedSearchCV for Random Forest...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST,
        n_iter=5,
        cv=2,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    metrics = evaluate_model("Tuned Random Forest", y_test, best_model.predict(X_test))

    return best_model, metrics


if __name__ == "__main__":
    train_random_forest()
