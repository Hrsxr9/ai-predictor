"""Latih model forecasting harga, evaluasi pada periode terbaru, lalu simpan RF untuk prediksi lokal."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    NUMERIC_FEATURES,
    chronological_split,
    clean_data,
    dataset_fingerprint,
    make_features,
    read_csv,
)

MODEL_VERSION = 2
RF_NAME = "Random Forest"
LR_NAME = "Linear Regression"
BASELINE_NAME = "Harga sebelumnya (baseline)"


def runtime_versions() -> dict:
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def build_pipeline(model_name: str) -> Pipeline:
    if model_name not in (RF_NAME, LR_NAME):
        raise ValueError("Model tidak dikenal.")

    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if model_name == LR_NAME:
        numeric_steps.append(("scale", StandardScaler()))

    preprocess = ColumnTransformer([
        (
            "numeric",
            Pipeline(numeric_steps),
            NUMERIC_FEATURES,
        ),
        (
            "category",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="first" if model_name == LR_NAME else None,
            ),
            CATEGORICAL_FEATURES,
        ),
    ])

    estimator = (
        RandomForestRegressor(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        if model_name == RF_NAME
        else LinearRegression()
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def evaluate(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    mse = float(mean_squared_error(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted))
        if len(actual) >= 2 and np.ptp(actual) > 0
        else np.nan,
    }


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    values = pipeline.named_steps["model"].feature_importances_
    result = {}

    numeric_count = len(NUMERIC_FEATURES)
    for column, value in zip(NUMERIC_FEATURES, values[:numeric_count]):
        result[column] = float(value)

    encoder = pipeline.named_steps["preprocess"].named_transformers_["category"]
    offset = numeric_count
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        result[column] = float(
            values[offset : offset + len(categories)].sum()
        )
        offset += len(categories)

    return pd.DataFrame({
        "Fitur": list(result),
        "Importance": list(result.values()),
    }).sort_values("Importance", ascending=False, ignore_index=True)


def train_models(frame: pd.DataFrame, test_fraction: float = 0.2) -> dict:
    train, test = chronological_split(frame, test_fraction)
    x_train, x_test = make_features(train), make_features(test)

    evaluation = test[
        ["date", "product", "category", "price", "previous_price"]
    ].copy()

    rows, fitted = [], {}
    for name in [RF_NAME, LR_NAME]:
        pipeline = build_pipeline(name)
        pipeline.fit(x_train, train["price"])
        predicted = pipeline.predict(x_test)

        evaluation[name] = predicted
        rows.append({
            "Model": name,
            **evaluate(test["price"], predicted),
        })
        fitted[name] = pipeline

    previous = test["previous_price"].fillna(train["previous_price"].median())
    evaluation[BASELINE_NAME] = previous.to_numpy()
    rows.append({
        "Model": BASELINE_NAME,
        **evaluate(test["price"], previous),
    })

    metrics = pd.DataFrame(rows)

    per_product = []
    for product, group in evaluation.groupby("product", sort=True):
        for name in [RF_NAME, LR_NAME, BASELINE_NAME]:
            per_product.append({
                "Produk": product,
                "Model": name,
                "Baris uji": len(group),
                **evaluate(group["price"], group[name]),
            })

    # Model produksi dilatih ulang menggunakan seluruh histori yang tersedia.
    production_model = build_pipeline(RF_NAME)
    production_model.fit(make_features(frame), frame["price"])

    return {
        "model_version": MODEL_VERSION,
        "versions": runtime_versions(),
        "dataset_hash": dataset_fingerprint(frame),
        "pipeline": production_model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "per_product": pd.DataFrame(per_product),
        "evaluation": evaluation.reset_index(drop=True),
        "importance": feature_importance(fitted[RF_NAME]),
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": train["date"].min().strftime("%Y-%m-%d"),
            "train_end": train["date"].max().strftime("%Y-%m-%d"),
            "test_start": test["date"].min().strftime("%Y-%m-%d"),
            "test_end": test["date"].max().strftime("%Y-%m-%d"),
            "test_fraction": test_fraction,
        },
        "products": (
            frame[["product", "category"]]
            .drop_duplicates()
            .sort_values(["product", "category"])
            .to_dict("records")
        ),
        "input_ranges": {
            column: (
                float(frame[column].min()),
                float(frame[column].max()),
            )
            for column in ["previous_price", "previous_demand"]
            if frame[column].notna().any()
        },
        "history_end": frame["date"].max().strftime("%Y-%m-%d"),
        "demand_available": bool(frame["demand"].notna().any()),
    }


def save_model(bundle: dict, path: Path = DEFAULT_MODEL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_model(
    path: Path = DEFAULT_MODEL,
    expected_hash: str | None = None,
) -> dict:
    bundle = joblib.load(path)
    required = {
        "pipeline",
        "metrics",
        "evaluation",
        "importance",
        "split",
        "products",
        "per_product",
        "input_ranges",
        "history_end",
        "demand_available",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Isi model tidak lengkap. Latih model kembali.")
    if (
        bundle.get("model_version") != MODEL_VERSION
        or bundle.get("versions") != runtime_versions()
    ):
        raise ValueError(
            "Versi model/library berbeda. Latih model kembali di lingkungan ini."
        )
    if expected_hash is not None and bundle.get("dataset_hash") != expected_hash:
        raise ValueError(
            "Dataset telah berubah. Latih kembali model untuk dataset aktif."
        )
    return bundle


def main():
    parser = argparse.ArgumentParser(
        description="Training AI Price Predictor tanpa membuka web."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    try:
        frame, report = clean_data(read_csv(args.data))
        bundle = train_models(frame, args.test_fraction)
        save_model(bundle, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Training gagal: {exc}\n")

    print(
        f"Data valid: {report['rows_out']} | "
        f"Dihapus: {report['rows_removed']}"
    )
    print(
        bundle["metrics"].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    print(f"Model Random Forest tersimpan: {args.output}")


if __name__ == "__main__":
    main()
