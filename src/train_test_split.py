import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split

LEAKAGE_COLUMNS = ["casual", "registered"]
TARGET_COLUMN = "count"


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Drop the target and the sub-components that would leak it
    drop_cols = [TARGET_COLUMN] + [c for c in LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COLUMN]

    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)
