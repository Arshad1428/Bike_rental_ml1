from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import add_derived_features, build_preprocessor
from src.train_test_split import get_train_test_split
from src.train_model import evaluate_model

PARAM_DIST = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.01, 0.05, 0.1],
}


def train_gradient_boosting():
    df = run_preprocessing()
    df = add_derived_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    print("Running RandomizedSearchCV for Gradient Boosting...")
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
    metrics = evaluate_model("Tuned Gradient Boosting", y_test, best_model.predict(X_test))

    return best_model, metrics


if __name__ == "__main__":
    train_gradient_boosting()
