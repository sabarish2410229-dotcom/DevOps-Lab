"""
Stage 4: Model Building
--------------------------
Trains a RandomForestRegressor on the engineered training features
(the best-performing model from the original notebook's comparison
of Linear Regression / Decision Tree / Random Forest) and serializes
the fitted model.

Input:
    data/features/train.csv
Output:
    model.pkl
"""

import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(path: str = "data/features/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[model_building] Loaded training data (shape={df.shape})")
    return df


def train_model(df: pd.DataFrame, target_column: str, n_estimators: int, max_depth, random_state: int):
    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("[model_building] Model training complete")
    return model


def save_model(model, path: str = "model.pkl") -> None:
    joblib.dump(model, path)
    print(f"[model_building] Saved model -> {path}")


def main():
    params = load_params()
    mb_params = params["model_building"]
    target_column = params["feature_engineering"]["target_column"]

    df = load_train_data()
    model = train_model(
        df,
        target_column=target_column,
        n_estimators=mb_params["n_estimators"],
        max_depth=mb_params["max_depth"],
        random_state=mb_params["random_state"],
    )
    save_model(model)


if __name__ == "__main__":
    main()
