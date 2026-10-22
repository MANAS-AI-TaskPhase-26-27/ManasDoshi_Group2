from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "close"
DATE_COLUMN = "date"
SYMBOL_COLUMN = "symbols"
BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train.csv"
TEST_FILE = BASE_DIR / "test.csv"

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def engineer_features(frame):
    data = frame.copy()
    data = data.drop(columns=["index"], errors="ignore")
    if DATE_COLUMN in data.columns:
        dates = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        month = dates.dt.month.astype(float)
        weekday = dates.dt.dayofweek.astype(float)
        day_of_year = dates.dt.dayofyear.astype(float)
        data["year"] = dates.dt.year.astype(float)
        data["date_missing"] = dates.isna().astype(float)
        data["month_sin"] = np.sin(2 * np.pi * month / 12)
        data["month_cos"] = np.cos(2 * np.pi * month / 12)
        data["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        data["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        data["year_day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        data["year_day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        data = data.drop(columns=[DATE_COLUMN])
    if SYMBOL_COLUMN in data.columns:
        data[SYMBOL_COLUMN] = (
            data[SYMBOL_COLUMN]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    for column in data.columns:
        if column != SYMBOL_COLUMN:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    return data

def fit_preprocessor(features):
    category_column = (
        SYMBOL_COLUMN if SYMBOL_COLUMN in features.columns else None
    )
    numeric_columns = [
        column for column in features.columns if column != category_column
    ]
    medians = {}
    means = {}
    scales = {}
    for column in numeric_columns:
        values = features[column].to_numpy(dtype=float)
        median = np.nanmedian(values)
        if not np.isfinite(median):
            median = 0.0
        filled = np.where(np.isfinite(values), values, median)
        mean = float(filled.mean())
        scale = float(filled.std())
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        medians[column] = float(median)
        means[column] = mean
        scales[column] = scale
    if category_column:
        categories = sorted(
            features[category_column].astype(str).unique().tolist()
        )
    else:
        categories = []
    return {
        "numeric_columns": numeric_columns,
        "category_column": category_column,
        "medians": medians,
        "means": means,
        "scales": scales,
        "categories": categories,
    }

def transform_features(features, prep):
    row_count = len(features)
    numeric_columns = prep["numeric_columns"]
    numeric_matrix = np.empty(
        (row_count, len(numeric_columns)), dtype=np.float64
    )
    for column_number, column in enumerate(numeric_columns):
        if column in features.columns:
            values = pd.to_numeric(
                features[column], errors="coerce"
            ).to_numpy(dtype=float)
        else:
            values = np.full(row_count, np.nan)
        values = np.where(
            np.isfinite(values), values, prep["medians"][column]
        )
        numeric_matrix[:, column_number] = (
            values - prep["means"][column]
        ) / prep["scales"][column]
    if prep["category_column"]:
        category_values = (
            features[prep["category_column"]]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .to_numpy()
        )
        categories = prep["categories"]
        category_lookup = {
            category: index for index, category in enumerate(categories)
        }
        category_matrix = np.zeros(
            (row_count, len(categories)), dtype=np.float64
        )
        for row_number, category in enumerate(category_values):
            column_number = category_lookup.get(category)
            if column_number is not None:
                category_matrix[row_number, column_number] = 1.0
        numeric_matrix = np.concatenate(
            [numeric_matrix, category_matrix], axis=1
        )
    intercept = np.ones((row_count, 1), dtype=np.float64)
    return np.concatenate([intercept, numeric_matrix], axis=1)

def fit_ridge(design_matrix, target, alpha):
    gram_matrix = design_matrix.T @ design_matrix
    right_side = design_matrix.T @ target
    penalty = np.eye(gram_matrix.shape[0], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(gram_matrix + penalty, right_side)
    return weights

def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) == 0:
        return {}
    errors = predicted - actual
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    total_variation = float(np.sum((actual - np.mean(actual)) ** 2))
    if total_variation > 0:
        r2 = float(1 - np.sum(errors ** 2) / total_variation)
    else:
        r2 = float("nan")
    return {
        "n": int(len(actual)),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

def grade_model(model_metrics, baseline_metrics):
    if not model_metrics or not baseline_metrics:
        return "Not enough labelled data to grade"
    if model_metrics["RMSE"] >= baseline_metrics["RMSE"]:
        return "F - did not beat the baseline"
    r2 = model_metrics["R2"]
    if r2 >= 0.80:
        return "A - strong fit on this evaluation set"
    if r2 >= 0.60:
        return "B - good fit on this evaluation set"
    if r2 >= 0.30:
        return "C - moderate fit on this evaluation set"
    if r2 >= 0.00:
        return "D - weak fit on this evaluation set"
    return "F - worse than a mean-based predictor"

def main():
    print("Loading CSV files...")
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    if TARGET not in train.columns:
        raise ValueError("train.csv must contain a 'close' column.")
    if TARGET in test.columns:
        actual_test_close = pd.to_numeric(
            test[TARGET], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    else:
        actual_test_close = np.full(len(test), np.nan)
    train = train.drop_duplicates().copy()
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train[TARGET] = train[TARGET].replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(train) < 10:
        raise ValueError("There are too few valid labelled training rows.")
    y_train = train[TARGET].to_numpy(dtype=float)
    X_raw = train.drop(columns=[TARGET])
    X = engineer_features(X_raw)
    X_test = engineer_features(
        test.drop(columns=[TARGET], errors="ignore")
    )
    X_test = X_test.reindex(columns=X.columns)
    missing_fraction = X.isna().mean()
    columns_to_drop = missing_fraction[
        missing_fraction > 0.50
    ].index.tolist()
    X = X.drop(columns=columns_to_drop)
    X_test = X_test.drop(
        columns=columns_to_drop, errors="ignore"
    ).reindex(columns=X.columns)
    train_dates = pd.to_datetime(train[DATE_COLUMN], errors="coerce")
    dated_rows = train_dates.notna()
    if dated_rows.sum() < 10:
        raise ValueError("Not enough valid dates for chronological validation.")
    cutoff = train_dates[dated_rows].quantile(0.80)
    fit_mask = ((train_dates <= cutoff) | (~dated_rows)).to_numpy()
    validation_mask = (train_dates > cutoff).to_numpy()
    fit_rows = np.flatnonzero(fit_mask)
    validation_rows = np.flatnonzero(validation_mask)
    X_fit = X.iloc[fit_rows]
    y_fit = y_train[fit_rows]
    X_validation = X.iloc[validation_rows]
    y_validation = y_train[validation_rows]
    validation_prep = fit_preprocessor(X_fit)
    fit_matrix = transform_features(X_fit, validation_prep)
    validation_matrix = transform_features(
        X_validation, validation_prep
    )
    validation_results = []
    for alpha in RIDGE_ALPHAS:
        weights = fit_ridge(fit_matrix, y_fit, alpha)
        validation_predictions = validation_matrix @ weights
        scores = regression_metrics(
            y_validation, validation_predictions
        )
        validation_results.append((alpha, scores))
    best_alpha, best_validation = min(
        validation_results, key=lambda item: item[1]["RMSE"]
    )
    validation_baseline = regression_metrics(
        y_validation,
        np.full(len(y_validation), np.median(y_fit))
    )
    validation_grade = grade_model(
        best_validation, validation_baseline
    )
    final_prep = fit_preprocessor(X)
    full_train_matrix = transform_features(X, final_prep)
    final_weights = fit_ridge(
        full_train_matrix, y_train, best_alpha
    )
    test_matrix = transform_features(X_test, final_prep)
    test_predictions = test_matrix @ final_weights
    test_metrics = regression_metrics(
        actual_test_close, test_predictions
    )
    test_baseline_predictions = np.full(
        len(test), np.median(y_train)
    )
    test_baseline_metrics = regression_metrics(
        actual_test_close, test_baseline_predictions
    )
    test_grade = grade_model(test_metrics, test_baseline_metrics)
    output = pd.DataFrame({
        "row_number": np.arange(len(test)),
        "date": test[DATE_COLUMN].to_numpy()
        if DATE_COLUMN in test.columns else "",
        "symbols": test[SYMBOL_COLUMN].to_numpy()
        if SYMBOL_COLUMN in test.columns else "",
        "actual_close_for_evaluation_only": actual_test_close,
        "predicted_close": test_predictions,
    })
    output["absolute_error"] = np.abs(
        output["actual_close_for_evaluation_only"]
        - output["predicted_close"]
    )
    output.to_csv("predictions.csv", index=False)
    print("\n--- TRAINING AND VALIDATION ---")
    print(f"Usable training rows: {len(train):,}")
    print(f"Validation cutoff date: {cutoff.date()}")
    print(f"Training rows in fit split: {len(fit_rows):,}")
    print(f"Validation rows: {len(validation_rows):,}")
    print(f"Selected Ridge alpha: {best_alpha}")
    print(f"Validation MAE:  {best_validation['MAE']:.4f}")
    print(f"Validation RMSE: {best_validation['RMSE']:.4f}")
    print(f"Validation R2:   {best_validation['R2']:.6f}")
    print(f"Validation grade: {validation_grade}")
    print("\n--- TEST RESULTS ---")
    print(f"Test rows predicted: {len(test):,}")
    if test_metrics:
        print(f"Rows with actual close for evaluation: {test_metrics['n']:,}")
        print(f"Test MAE:  {test_metrics['MAE']:.4f}")
        print(f"Test MSE:  {test_metrics['MSE']:.4f}")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Test R2:   {test_metrics['R2']:.6f}")
        print(f"Median baseline RMSE: {test_baseline_metrics['RMSE']:.4f}")
        print(f"Educational grade: {test_grade}")
    else:
        print("Test labels are absent; predictions were made but cannot be scored.")
    print("\nSaved all predictions to predictions.csv")

if __name__ == "__main__":
    main()