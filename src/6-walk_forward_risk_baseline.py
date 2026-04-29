"""
Auxiliary risk baseline.
This script predicts realized-volatility proxies, not return distributions.
Return distribution baselines are implemented in 7-walk_forward_distribution_baseline.py.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


PRICE_FEATURES = [
    "log_return_1d",
    "abs_log_return_1d",
    "squared_log_return_1d",
    "parkinson_vol_1d",
    "hist_rv_return_3d",
    "hist_abs_return_mean_3d",
    "hist_parkinson_vol_3d",
    "hist_rv_return_5d",
    "hist_abs_return_mean_5d",
    "hist_parkinson_vol_5d",
    "hist_rv_return_10d",
    "hist_abs_return_mean_10d",
    "hist_parkinson_vol_10d",
    "hist_rv_return_20d",
    "hist_abs_return_mean_20d",
    "hist_parkinson_vol_20d",
]


def parse_csv_arg(x):
    return [v.strip() for v in str(x).split(",") if v.strip()]


def parse_horizon_from_target(target_col):
    m = re.search(r"_(\d+)d", str(target_col))
    if not m:
        raise ValueError(f"Cannot parse horizon from target column: {target_col}")
    return int(m.group(1))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def directional_accuracy(y_true, y_pred, baseline_level):
    """
    Directional accuracy for whether volatility is above a baseline level.
    baseline_level is usually the train-set mean target.
    """
    true_up = y_true > baseline_level
    pred_up = y_pred > baseline_level
    return np.mean(true_up == pred_up)


def get_event_features(df):
    event_cols = [c for c in df.columns if c.startswith("event_")]

    # Drop rolling sums of mean-like variables because the current build script
    # sums them over windows. Counts and sums are fine, but means should not be summed.
    bad_roll_mean_cols = [
        c for c in event_cols
        if (
            ("tone_weighted_mean_roll" in c)
            or ("goldstein_weighted_mean_roll" in c)
        )
    ]

    event_cols = [c for c in event_cols if c not in bad_roll_mean_cols]
    return event_cols


def make_model(model_type):
    if model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

    raise ValueError(f"Unknown model_type: {model_type}")


def prepare_asset_data(df, asset, target_col, extra_cols):
    data = df[df["asset"] == asset].copy()
    data = data.sort_values("date").reset_index(drop=True)

    needed_cols = ["date", target_col] + list(extra_cols)
    missing = [c for c in needed_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for {asset} {target_col}: {missing}")

    data = data[needed_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 500:
        raise ValueError(f"Too few rows for {asset} {target_col}: {len(data)}")

    # Walk-forward by year.
    # Train on all data before test year, test on that year.
    data["year"] = pd.to_datetime(data["date"]).dt.year
    return data


def append_result_and_predictions(
    results,
    all_preds,
    asset,
    target_col,
    model_type,
    feature_set,
    test_year,
    train,
    test,
    y_test,
    pred,
):
    train_mean = float(np.mean(train[target_col].values))

    row = {
        "asset": asset,
        "target": target_col,
        "model_type": model_type,
        "feature_set": feature_set,
        "test_year": int(test_year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(rmse(y_test, pred)),
        "directional_acc": float(directional_accuracy(y_test, pred, train_mean)),
        "train_target_mean": train_mean,
    }
    results.append(row)

    pred_df = pd.DataFrame({
        "date": test["date"].values,
        "asset": asset,
        "target": target_col,
        "model_type": model_type,
        "feature_set": feature_set,
        "test_year": test_year,
        "y_true": y_test,
        "y_pred": pred,
    })
    all_preds.append(pred_df)


def evaluate_naive_setting(df, asset, target_col, first_test_year):
    data = prepare_asset_data(
        df=df,
        asset=asset,
        target_col=target_col,
        extra_cols=[],
    )

    results = []
    all_preds = []

    test_years = sorted(data["year"].unique())
    test_years = [y for y in test_years if y >= first_test_year]

    for test_year in test_years:
        train = data[data["year"] < test_year].copy()
        test = data[data["year"] == test_year].copy()

        if len(train) < 500 or len(test) < 30:
            continue

        y_test = test[target_col].values
        pred_value = float(np.mean(train[target_col].values))
        pred = np.full_like(y_test, fill_value=pred_value, dtype=float)

        append_result_and_predictions(
            results=results,
            all_preds=all_preds,
            asset=asset,
            target_col=target_col,
            model_type="naive",
            feature_set="price_history",
            test_year=test_year,
            train=train,
            test=test,
            y_test=y_test,
            pred=pred,
        )

    result_df = pd.DataFrame(results)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return result_df, pred_df


def evaluate_ewma_setting(df, asset, target_col, first_test_year, ewma_alpha):
    horizon = parse_horizon_from_target(target_col)

    if "rv_return" in target_col:
        base_col = "squared_log_return_1d"
        transform = "sqrt_ewm_mean"
    elif "rv_ohlc" in target_col:
        base_col = "parkinson_vol_1d"
        transform = "ewm_mean"
    else:
        raise ValueError(f"Unknown risk target type: {target_col}")

    data = prepare_asset_data(
        df=df,
        asset=asset,
        target_col=target_col,
        extra_cols=[base_col],
    )

    base = data[base_col].astype(float).clip(lower=0.0)
    if transform == "sqrt_ewm_mean":
        # EWMA return-volatility proxy from historical squared daily returns.
        data["ewma_risk_pred"] = np.sqrt(
            base.ewm(alpha=ewma_alpha, adjust=False).mean()
        ) * np.sqrt(horizon)
    else:
        # EWMA OHLC-volatility proxy from historical Parkinson volatility.
        data["ewma_risk_pred"] = (
            base.ewm(alpha=ewma_alpha, adjust=False).mean()
        ) * np.sqrt(horizon)

    results = []
    all_preds = []

    test_years = sorted(data["year"].unique())
    test_years = [y for y in test_years if y >= first_test_year]

    for test_year in test_years:
        train = data[data["year"] < test_year].copy()
        test = data[data["year"] == test_year].copy()

        if len(train) < 500 or len(test) < 30:
            continue

        y_test = test[target_col].values
        pred = test["ewma_risk_pred"].values.astype(float)

        fallback = float(np.mean(train[target_col].values))
        bad = ~np.isfinite(pred) | (pred < 0)
        pred[bad] = fallback
        pred = np.maximum(pred, 0.0)

        append_result_and_predictions(
            results=results,
            all_preds=all_preds,
            asset=asset,
            target_col=target_col,
            model_type="ewma",
            feature_set="price_history",
            test_year=test_year,
            train=train,
            test=test,
            y_test=y_test,
            pred=pred,
        )

    result_df = pd.DataFrame(results)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return result_df, pred_df


def evaluate_ridge_setting(df, asset, target_col, feature_cols, feature_set, first_test_year):
    data = prepare_asset_data(
        df=df,
        asset=asset,
        target_col=target_col,
        extra_cols=feature_cols,
    )

    results = []
    all_preds = []

    test_years = sorted(data["year"].unique())
    test_years = [y for y in test_years if y >= first_test_year]

    for test_year in test_years:
        train = data[data["year"] < test_year].copy()
        test = data[data["year"] == test_year].copy()

        if len(train) < 500 or len(test) < 30:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].values

        X_test = test[feature_cols]
        y_test = test[target_col].values

        model = make_model("ridge")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0.0)

        append_result_and_predictions(
            results=results,
            all_preds=all_preds,
            asset=asset,
            target_col=target_col,
            model_type="ridge",
            feature_set=feature_set,
            test_year=test_year,
            train=train,
            test=test,
            y_test=y_test,
            pred=pred,
        )

    result_df = pd.DataFrame(results)
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return result_df, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_table",
        default="/project/hrao/GDELT/modeling/model_table.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/risk_baseline_v1",
    )
    parser.add_argument(
        "--model_type",
        default="",
        help="Deprecated. Use --models instead. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list from: naive,ewma,ridge. If omitted, uses --model_type or ridge.",
    )
    parser.add_argument(
        "--first_test_year",
        type=int,
        default=2019,
    )
    parser.add_argument(
        "--ewma_alpha",
        type=float,
        default=0.06,
    )
    args = parser.parse_args()

    if args.models:
        requested_models = parse_csv_arg(args.models)
    elif args.model_type:
        requested_models = parse_csv_arg(args.model_type)
    else:
        requested_models = ["ridge"]

    allowed_models = {"naive", "ewma", "ridge"}
    bad = [m for m in requested_models if m not in allowed_models]
    if bad:
        raise ValueError(f"Unknown risk models: {bad}. Allowed: {sorted(allowed_models)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.model_table)
    df["date"] = pd.to_datetime(df["date"])

    event_features = get_event_features(df)
    if not event_features:
        raise ValueError("No event features found. Expected columns starting with event_.")

    # Log-transform nonnegative event magnitude features.
    # This reduces the effect of very large event counts.
    for col in event_features:
        if df[col].min(skipna=True) >= 0:
            df[col] = np.log1p(df[col])

    target_cols = [
        "target_rv_return_1d",
        "target_rv_return_3d",
        "target_rv_return_5d",
        "target_rv_ohlc_1d",
        "target_rv_ohlc_3d",
        "target_rv_ohlc_5d",
    ]

    assets = sorted(df["asset"].unique())

    experiment_configs = {
        "price_only": PRICE_FEATURES,
        "event_only": event_features,
        "price_plus_event": PRICE_FEATURES + event_features,
    }

    all_results = []
    all_preds = []

    for asset in assets:
        for target_col in target_cols:
            print(f"\n=== Asset={asset} Target={target_col} ===")

            if "naive" in requested_models:
                print("Running naive risk baseline")
                result_df, pred_df = evaluate_naive_setting(
                    df=df,
                    asset=asset,
                    target_col=target_col,
                    first_test_year=args.first_test_year,
                )
                all_results.append(result_df)
                all_preds.append(pred_df)

            if "ewma" in requested_models:
                print("Running EWMA risk baseline")
                result_df, pred_df = evaluate_ewma_setting(
                    df=df,
                    asset=asset,
                    target_col=target_col,
                    first_test_year=args.first_test_year,
                    ewma_alpha=args.ewma_alpha,
                )
                all_results.append(result_df)
                all_preds.append(pred_df)

            if "ridge" in requested_models:
                for feature_set_name, feature_cols in experiment_configs.items():
                    print(
                        f"Running ridge risk baseline | {feature_set_name} | "
                        f"n_features={len(feature_cols)}"
                    )

                    result_df, pred_df = evaluate_ridge_setting(
                        df=df,
                        asset=asset,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        feature_set=feature_set_name,
                        first_test_year=args.first_test_year,
                    )

                    all_results.append(result_df)
                    all_preds.append(pred_df)

    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    suffix = "all" if len(requested_models) > 1 else requested_models[0]
    results_file = out_dir / f"walk_forward_results_{suffix}.csv"
    preds_file = out_dir / f"walk_forward_predictions_{suffix}.csv"
    summary_file = out_dir / f"walk_forward_summary_{suffix}.csv"

    results.to_csv(results_file, index=False)
    preds.to_csv(preds_file, index=False)

    summary = (
        results
        .groupby(["feature_set", "asset", "target", "model_type"])
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            directional_acc_mean=("directional_acc", "mean"),
            n_test_total=("n_test", "sum"),
        )
        .reset_index()
    )

    summary.to_csv(summary_file, index=False)

    print("\nSaved:")
    print(results_file)
    print(preds_file)
    print(summary_file)

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
