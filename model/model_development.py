import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "Neighborhood"
]
TARGET = "SalePrice"


def main():
    # 1) Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"train.csv not found at {DATA_PATH}. "
            "Download it from Kaggle and place it in model/train.csv"
        )
    df = pd.read_csv(DATA_PATH)

    # Keep only allowed columns (6 features + target)
    df = df[FEATURES + [TARGET]].copy()

    # 2) Preprocessing
    numeric_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt"]
    categorical_features = ["Neighborhood"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
        # No scaling needed for RandomForest
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 3) Algorithm: Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # Full pipeline (preprocess + model)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Split
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Train
    clf.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = clf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("MODEL EVALUATION")
    print(f"MAE : {mae:,.2f}")
    print(f"MSE : {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2 : {r2:.4f}")

    # 6) Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    # 7) Reload without retraining (test reload)
    loaded = joblib.load(MODEL_PATH)
    test_pred = loaded.predict(X_test.iloc[:3])
    print("\nReload test OK. Sample predictions:", test_pred)


if __name__ == "__main__":
    main()