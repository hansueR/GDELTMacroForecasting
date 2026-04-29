from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


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
    return [v.strip() for v in x.split(",") if v.strip()]


def get_event_feature_groups(df: pd.DataFrame):
    event_cols = [c for c in df.columns if c.startswith("event_")]

    bad_roll_mean_cols = [
        c for c in event_cols
        if (
            ("tone_weighted_mean_roll" in c)
            or ("goldstein_weighted_mean_roll" in c)
        )
    ]

    event_cols = [c for c in event_cols if c not in bad_roll_mean_cols]

    daily_event_cols = [c for c in event_cols if "_roll" not in c]
    rolling_event_cols = [c for c in event_cols if "_roll" in c]
    all_event_cols = daily_event_cols + rolling_event_cols

    if not daily_event_cols:
        raise ValueError("No daily event features found. Expected event_ columns without _roll.")
    if not rolling_event_cols:
        raise ValueError("No rolling event features found. Expected event_ columns containing _roll.")
    if not all_event_cols:
        raise ValueError("No event features found. Expected columns starting with event_.")

    print("n_daily_event_features:", len(daily_event_cols))
    print("n_rolling_event_features:", len(rolling_event_cols))
    print("n_all_event_features:", len(all_event_cols))

    return {
        "daily": daily_event_cols,
        "rolling": rolling_event_cols,
        "all": all_event_cols,
    }


def get_event_features(df: pd.DataFrame):
    return get_event_feature_groups(df)["all"]


def make_ridge_configs(experiment_preset, event_groups):
    if experiment_preset == "standard":
        return {
            "price_only": PRICE_FEATURES,
            "event_only": event_groups["all"],
            "price_plus_event": PRICE_FEATURES + event_groups["all"],
        }

    if experiment_preset == "event_window_ablation":
        return {
            "price_only": PRICE_FEATURES,
            "event_daily_only": event_groups["daily"],
            "event_rolling_only": event_groups["rolling"],
            "event_all": event_groups["all"],
            "price_plus_event_daily": PRICE_FEATURES + event_groups["daily"],
            "price_plus_event_rolling": PRICE_FEATURES + event_groups["rolling"],
            "price_plus_event_all": PRICE_FEATURES + event_groups["all"],
        }

    raise ValueError(f"Unknown experiment_preset: {experiment_preset}")


def make_ridge_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def gaussian_nll(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    var = sigma ** 2
    return 0.5 * (np.log(2.0 * np.pi * var) + ((y_true - mu) ** 2) / var)


def safe_std(y, sigma_floor):
    sigma = float(np.std(y, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = sigma_floor
    return max(sigma, sigma_floor)


def estimate_sigma_from_train_residuals(y_train, mu_train, sigma_floor):
    residuals = y_train - mu_train
    sigma = safe_std(residuals, sigma_floor)

    if not np.isfinite(sigma) or sigma <= sigma_floor:
        sigma = safe_std(y_train, sigma_floor)

    return max(sigma, sigma_floor)


def directional_accuracy_from_mu(y_true, mu):
    true_up = y_true > 0
    pred_up = mu > 0
    return float(np.mean(true_up == pred_up))


def auc_from_mu(y_true, mu):
    """
    AUC for predicting whether the future return is positive.
    The score is the predicted return mean mu.
    """
    y_bin = (y_true > 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return float(roc_auc_score(y_bin, mu))


def prepare_asset_data(df, asset, target_col, extra_cols):
    data = df[df["asset"] == asset].copy()
    data = data.sort_values("date").reset_index(drop=True)

    needed_cols = ["date", target_col] + extra_cols
    missing = [c for c in needed_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for {asset}: {missing}")

    data = data[needed_cols].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("date").reset_index(drop=True)
    data["year"] = pd.to_datetime(data["date"]).dt.year
    return data


def get_walk_forward_split(data, test_year, horizon):
    train_candidate = data[data["year"] < test_year].copy()
    test = data[data["year"] == test_year].copy()

    # A row dated just before the test year can have a future h-day target
    # that overlaps the test year. Drop the last h training rows.
    if len(train_candidate) > horizon:
        train = train_candidate.iloc[:-horizon].copy()
    else:
        train = train_candidate.iloc[0:0].copy()

    return train, test


def append_result_and_predictions(
    results,
    all_preds,
    asset,
    horizon,
    target_col,
    feature_set,
    model_type,
    test_year,
    train,
    test,
    y_test,
    mu_test,
    sigma_test,
):
    nll = gaussian_nll(y_test, mu_test, sigma_test)

    results.append({
        "asset": asset,
        "horizon": int(horizon),
        "target": target_col,
        "feature_set": feature_set,
        "model_type": model_type,
        "test_year": int(test_year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "nll_mean": float(np.mean(nll)),
        "nll_median": float(np.median(nll)),
        "directional_acc": directional_accuracy_from_mu(y_test, mu_test),
        "auc": auc_from_mu(y_test, mu_test),
        "mu_mae": float(np.mean(np.abs(y_test - mu_test))),
        "mu_rmse": float(np.sqrt(np.mean((y_test - mu_test) ** 2))),
        "sigma_mean": float(np.mean(sigma_test)),
    })

    all_preds.append(pd.DataFrame({
        "date": test["date"].values,
        "asset": asset,
        "horizon": int(horizon),
        "target": target_col,
        "feature_set": feature_set,
        "model_type": model_type,
        "test_year": int(test_year),
        "y_true_return": y_test,
        "y_true_up": (y_test > 0).astype(int),
        "mu_pred": mu_test,
        "sigma_pred": sigma_test,
        "nll": nll,
    }))


def evaluate_naive(
    df,
    asset,
    horizon,
    first_test_year,
    sigma_floor,
):
    target_col = f"target_return_{horizon}d"

    data = prepare_asset_data(
        df=df,
        asset=asset,
        target_col=target_col,
        extra_cols=[],
    )

    results = []
    all_preds = []

    test_years = sorted(y for y in data["year"].unique() if y >= first_test_year)

    for test_year in test_years:
        train, test = get_walk_forward_split(data, test_year, horizon)

        if len(train) < 500 or len(test) < 30:
            continue

        y_train = train[target_col].values
        y_test = test[target_col].values

        mu_value = float(np.mean(y_train))
        sigma_value = safe_std(y_train, sigma_floor)

        mu_test = np.full_like(y_test, fill_value=mu_value, dtype=float)
        sigma_test = np.full_like(y_test, fill_value=sigma_value, dtype=float)

        append_result_and_predictions(
            results=results,
            all_preds=all_preds,
            asset=asset,
            horizon=horizon,
            target_col=target_col,
            feature_set="price_history",
            model_type="naive",
            test_year=test_year,
            train=train,
            test=test,
            y_test=y_test,
            mu_test=mu_test,
            sigma_test=sigma_test,
        )

    return pd.DataFrame(results), pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def evaluate_ewma(
    df,
    asset,
    horizon,
    first_test_year,
    sigma_floor,
    ewma_alpha,
):
    target_col = f"target_return_{horizon}d"

    data = prepare_asset_data(
        df=df,
        asset=asset,
        target_col=target_col,
        extra_cols=["log_return_1d"],
    )

    r = data["log_return_1d"].astype(float)

    # At date t, log_return_1d is already known.
    # EWMA uses information up to date t to forecast future t+1 ... t+h.
    data["ewma_mean_1d"] = r.ewm(alpha=ewma_alpha, adjust=False).mean()
    data["ewma_std_1d"] = r.ewm(alpha=ewma_alpha, adjust=False).std(bias=False)

    results = []
    all_preds = []

    test_years = sorted(y for y in data["year"].unique() if y >= first_test_year)

    for test_year in test_years:
        train, test = get_walk_forward_split(data, test_year, horizon)

        if len(train) < 500 or len(test) < 30:
            continue

        y_train = train[target_col].values
        y_test = test[target_col].values

        fallback_sigma = safe_std(y_train, sigma_floor)

        mu_test = horizon * test["ewma_mean_1d"].values
        sigma_test = np.sqrt(horizon) * test["ewma_std_1d"].values

        bad_sigma = ~np.isfinite(sigma_test) | (sigma_test <= 0)
        sigma_test[bad_sigma] = fallback_sigma
        sigma_test = np.maximum(sigma_test, sigma_floor)

        append_result_and_predictions(
            results=results,
            all_preds=all_preds,
            asset=asset,
            horizon=horizon,
            target_col=target_col,
            feature_set="price_history",
            model_type="ewma",
            test_year=test_year,
            train=train,
            test=test,
            y_test=y_test,
            mu_test=mu_test,
            sigma_test=sigma_test,
        )

    return pd.DataFrame(results), pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def evaluate_ridge(
    df,
    asset,
    horizon,
    feature_cols,
    feature_set,
    first_test_year,
    sigma_floor,
):
    target_col = f"target_return_{horizon}d"

    data = prepare_asset_data(
        df=df,
        asset=asset,
        target_col=target_col,
        extra_cols=feature_cols,
    )

    results = []
    all_preds = []

    test_years = sorted(y for y in data["year"].unique() if y >= first_test_year)

    for test_year in test_years:
        train, test = get_walk_forward_split(data, test_year, horizon)

        if len(train) < 500 or len(test) < 30:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].values

        X_test = test[feature_cols]
        y_test = test[target_col].values

        model = make_ridge_model()
        model.fit(X_train, y_train)

        mu_train = model.predict(X_train)
        mu_test = model.predict(X_test)

        sigma_value = estimate_sigma_from_train_residuals(
            y_train=y_train,
            mu_train=mu_train,
            sigma_floor=sigma_floor,
        )
        sigma_test = np.full_like(mu_test, fill_value=sigma_value, dtype=float)

        append_result_and_predictions(
            results=results,
            all_preds=all_preds,
            asset=asset,
            horizon=horizon,
            target_col=target_col,
            feature_set=feature_set,
            model_type="ridge",
            test_year=test_year,
            train=train,
            test=test,
            y_test=y_test,
            mu_test=mu_test,
            sigma_test=sigma_test,
        )

    return pd.DataFrame(results), pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_table",
        default="/project/hrao/GDELT/modeling/model_table.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/distribution_baselines_v2",
    )
    parser.add_argument(
        "--models",
        default="naive,ewma,ridge",
        help="Comma-separated list from: naive,ewma,ridge",
    )
    parser.add_argument(
        "--first_test_year",
        type=int,
        default=2019,
    )
    parser.add_argument(
        "--sigma_floor",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--ewma_alpha",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--experiment_preset",
        default="standard",
        choices=["standard", "event_window_ablation"],
    )
    args = parser.parse_args()

    requested_models = parse_csv_arg(args.models)
    allowed = {"naive", "ewma", "ridge"}
    bad = [m for m in requested_models if m not in allowed]
    if bad:
        raise ValueError(f"Unknown models: {bad}. Allowed: {sorted(allowed)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.model_table)
    df["date"] = pd.to_datetime(df["date"])

    required_targets = [
        "target_return_1d",
        "target_return_3d",
        "target_return_5d",
    ]
    missing_targets = [c for c in required_targets if c not in df.columns]
    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}. "
            "Run module 1 first to create future return labels. "
            "Regenerate daily_asset_prices.parquet and model_table.parquet."
        )

    event_groups = get_event_feature_groups(df)

    for col in event_groups["all"]:
        if df[col].min(skipna=True) >= 0:
            df[col] = np.log1p(df[col])

    ridge_configs = make_ridge_configs(args.experiment_preset, event_groups)
    print("ridge_config_feature_counts:")
    for feature_set, feature_cols in ridge_configs.items():
        print(f"  {feature_set}: {len(feature_cols)}")
    if args.experiment_preset == "event_window_ablation":
        largest_n_features = max(len(cols) for cols in ridge_configs.values())
        if len(ridge_configs["price_plus_event_all"]) != largest_n_features:
            raise ValueError("Expected price_plus_event_all to have the largest feature count.")

    assets = sorted(df["asset"].unique())
    horizons = [1, 3, 5]

    all_results = []
    all_preds = []

    for asset in assets:
        for horizon in horizons:
            print(f"\n=== Asset={asset} Horizon={horizon} ===")

            if "naive" in requested_models:
                print("Running naive")
                result_df, pred_df = evaluate_naive(
                    df=df,
                    asset=asset,
                    horizon=horizon,
                    first_test_year=args.first_test_year,
                    sigma_floor=args.sigma_floor,
                )
                all_results.append(result_df)
                all_preds.append(pred_df)

            if "ewma" in requested_models:
                print("Running EWMA")
                result_df, pred_df = evaluate_ewma(
                    df=df,
                    asset=asset,
                    horizon=horizon,
                    first_test_year=args.first_test_year,
                    sigma_floor=args.sigma_floor,
                    ewma_alpha=args.ewma_alpha,
                )
                all_results.append(result_df)
                all_preds.append(pred_df)

            if "ridge" in requested_models:
                for feature_set, feature_cols in ridge_configs.items():
                    print(f"Running ridge | {feature_set} | n_features={len(feature_cols)}")
                    result_df, pred_df = evaluate_ridge(
                        df=df,
                        asset=asset,
                        horizon=horizon,
                        feature_cols=feature_cols,
                        feature_set=feature_set,
                        first_test_year=args.first_test_year,
                        sigma_floor=args.sigma_floor,
                    )
                    all_results.append(result_df)
                    all_preds.append(pred_df)

    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    results_file = out_dir / "distribution_baselines_by_year.csv"
    preds_file = out_dir / "distribution_baselines_predictions.csv"
    summary_file = out_dir / "distribution_baselines_summary.csv"

    results.to_csv(results_file, index=False)
    preds.to_csv(preds_file, index=False)

    summary = (
        results
        .groupby(["model_type", "feature_set", "asset", "horizon", "target"])
        .agg(
            nll_mean=("nll_mean", "mean"),
            nll_median=("nll_median", "mean"),
            directional_acc_mean=("directional_acc", "mean"),
            auc_mean=("auc", "mean"),
            mu_mae_mean=("mu_mae", "mean"),
            mu_rmse_mean=("mu_rmse", "mean"),
            sigma_mean=("sigma_mean", "mean"),
            n_test_total=("n_test", "sum"),
        )
        .reset_index()
        .sort_values(["asset", "horizon", "model_type", "feature_set"])
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
