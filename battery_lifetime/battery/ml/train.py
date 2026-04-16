from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


def train_from_dataframe(df: pd.DataFrame):
    df = df.copy()

    required_columns = [
        "battery_id",
        "cycle",
        "capacity",
        "temperature",
        "ica_peak1_v",
        "ica_peak1_val",
        "ica_peak2_v",
        "ica_peak2_val",
        "ica_area_abs",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for training: {missing}")

    df = df.dropna(subset=["capacity"]).reset_index(drop=True)

    candidate_feature_cols = [
        "cycle",
        "temperature",
        "r0_ohm",
        "ica_peak1_v",
        "ica_peak1_val",
        "ica_peak2_v",
        "ica_peak2_val",
        "ica_area_abs",
    ]

    feature_cols = [
        c for c in candidate_feature_cols
        if c in df.columns and df[c].notna().any()
    ]

    if not feature_cols:
        raise ValueError("No usable feature columns found for training.")

    X = df[feature_cols]
    y = df["capacity"]
    groups = df["battery_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_dir = Path(__file__).resolve().parent / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "battery_capacity_cycle_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
        },
        model_path,
    )

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "model_path": str(model_path),
        "feature_cols": feature_cols,
    }