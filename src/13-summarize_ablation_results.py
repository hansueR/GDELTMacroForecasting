from pathlib import Path
import argparse

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "feature_set",
    "asset",
    "horizon",
    "target",
    "n_test_total",
    "nll_mean",
    "directional_acc_mean",
    "auc_mean",
    "mu_mae_mean",
    "mu_rmse_mean",
    "sigma_mean",
]


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def summarize_by_year(df):
    required = [
        "model_type",
        "feature_set",
        "asset",
        "horizon",
        "target",
        "n_test",
        "nll_mean",
        "directional_acc",
        "auc",
        "mu_mae",
        "mu_rmse",
        "sigma_mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"By-year file is missing columns: {missing}")

    rows = []
    group_cols = ["model_type", "feature_set", "asset", "horizon", "target"]
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_test_total"] = int(g["n_test"].sum())
        row["nll_mean"] = weighted_mean(g["nll_mean"], g["n_test"])
        row["directional_acc_mean"] = weighted_mean(g["directional_acc"], g["n_test"])
        row["auc_mean"] = weighted_mean(g["auc"], g["n_test"])
        row["mu_mae_mean"] = weighted_mean(g["mu_mae"], g["n_test"])
        row["mu_rmse_mean"] = weighted_mean(g["mu_rmse"], g["n_test"])
        row["sigma_mean"] = weighted_mean(g["sigma_mean"], g["n_test"])
        rows.append(row)

    return pd.DataFrame(rows)


def pick_metric_column(df, preferred, weighted):
    if preferred in df.columns:
        return preferred
    if weighted in df.columns:
        return weighted
    raise ValueError(f"Missing metric column. Expected {preferred} or {weighted}.")


def normalize_summary(df):
    out = df.copy()

    rename_map = {
        pick_metric_column(out, "nll_mean", "nll_mean_weighted"): "nll_mean",
        pick_metric_column(out, "directional_acc_mean", "directional_acc_weighted"): "directional_acc_mean",
        pick_metric_column(out, "auc_mean", "auc_weighted"): "auc_mean",
        pick_metric_column(out, "mu_mae_mean", "mu_mae_weighted"): "mu_mae_mean",
        pick_metric_column(out, "mu_rmse_mean", "mu_rmse_weighted"): "mu_rmse_mean",
        pick_metric_column(out, "sigma_mean", "sigma_mean_weighted"): "sigma_mean",
    }
    out = out.rename(columns=rename_map)

    required = [
        "model_type",
        "feature_set",
        "asset",
        "horizon",
        "target",
        "n_test_total",
        "nll_mean",
        "directional_acc_mean",
        "auc_mean",
        "mu_mae_mean",
        "mu_rmse_mean",
        "sigma_mean",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Summary file is missing columns: {missing}")

    return out[required].copy()


def read_distribution_summary(result_dir):
    result_dir = Path(result_dir)
    summary_path = result_dir / "distribution_baselines_summary.csv"
    by_year_path = result_dir / "distribution_baselines_by_year.csv"

    if summary_path.exists():
        return normalize_summary(pd.read_csv(summary_path))
    if by_year_path.exists():
        return normalize_summary(summarize_by_year(pd.read_csv(by_year_path)))

    raise FileNotFoundError(
        f"Missing distribution summary/by-year files in {result_dir}"
    )


def make_event_window_ablation(event_window_dir):
    summary = read_distribution_summary(event_window_dir)
    table = summary[summary["model_type"] == "ridge"].copy()
    table = table[SUMMARY_COLUMNS].copy()
    table = table.sort_values(["asset", "horizon", "target", "feature_set"]).reset_index(drop=True)
    return table


def make_event_window_gain(event_table):
    key_cols = ["asset", "horizon", "target"]

    price = event_table[event_table["feature_set"] == "price_only"]
    daily = event_table[event_table["feature_set"] == "price_plus_event_daily"]
    all_events = event_table[event_table["feature_set"] == "price_plus_event_all"]

    merged = (
        price[key_cols + ["nll_mean", "auc_mean"]]
        .merge(
            daily[key_cols + ["nll_mean", "auc_mean"]],
            on=key_cols,
            suffixes=("_price_only", "_price_plus_event_daily"),
        )
        .merge(
            all_events[key_cols + ["nll_mean", "auc_mean"]],
            on=key_cols,
        )
        .rename(columns={
            "nll_mean": "nll_price_plus_event_all",
            "auc_mean": "auc_price_plus_event_all",
            "nll_mean_price_only": "nll_price_only",
            "auc_mean_price_only": "auc_price_only",
            "nll_mean_price_plus_event_daily": "nll_price_plus_event_daily",
            "auc_mean_price_plus_event_daily": "auc_price_plus_event_daily",
        })
    )

    out = merged[key_cols].copy()
    out["nll_price_only"] = merged["nll_price_only"]
    out["nll_price_plus_event_daily"] = merged["nll_price_plus_event_daily"]
    out["nll_price_plus_event_all"] = merged["nll_price_plus_event_all"]
    out["gain_daily_vs_price"] = out["nll_price_only"] - out["nll_price_plus_event_daily"]
    out["gain_all_vs_price"] = out["nll_price_only"] - out["nll_price_plus_event_all"]
    out["gain_rolling_extra_vs_daily"] = (
        out["nll_price_plus_event_daily"] - out["nll_price_plus_event_all"]
    )
    out["auc_price_only"] = merged["auc_price_only"]
    out["auc_price_plus_event_daily"] = merged["auc_price_plus_event_daily"]
    out["auc_price_plus_event_all"] = merged["auc_price_plus_event_all"]
    out["auc_gain_daily_vs_price"] = (
        out["auc_price_plus_event_daily"] - out["auc_price_only"]
    )
    out["auc_gain_all_vs_price"] = (
        out["auc_price_plus_event_all"] - out["auc_price_only"]
    )
    out["auc_gain_rolling_extra_vs_daily"] = (
        out["auc_price_plus_event_all"] - out["auc_price_plus_event_daily"]
    )

    return out.sort_values(key_cols).reset_index(drop=True)


def make_topk_ablation(topk_root_dir):
    rows = []
    topk_root_dir = Path(topk_root_dir)

    for k in [10, 20, 50]:
        summary = read_distribution_summary(topk_root_dir / f"topk_{k}")
        table = summary[
            (summary["model_type"] == "ridge")
            & (summary["feature_set"] == "price_plus_event")
        ].copy()
        table["top_k"] = k
        rows.append(table)

    out = pd.concat(rows, ignore_index=True)
    out = out[
        [
            "top_k",
            "asset",
            "horizon",
            "target",
            "n_test_total",
            "nll_mean",
            "directional_acc_mean",
            "auc_mean",
            "mu_mae_mean",
            "mu_rmse_mean",
            "sigma_mean",
        ]
    ].copy()
    return out.sort_values(["asset", "horizon", "target", "top_k"]).reset_index(drop=True)


def make_topk_best(topk_table):
    key_cols = ["asset", "horizon", "target"]
    rows = []

    for keys, g in topk_table.groupby(key_cols, dropna=False):
        best = g.loc[g["nll_mean"].idxmin()]
        row = dict(zip(key_cols, keys))
        row["best_top_k"] = int(best["top_k"])
        row["best_nll_mean"] = float(best["nll_mean"])

        for k in [10, 20, 50]:
            one = g[g["top_k"] == k]
            row[f"top{k}_nll"] = float(one["nll_mean"].iloc[0]) if not one.empty else np.nan
            row[f"top{k}_auc"] = float(one["auc_mean"].iloc[0]) if not one.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(key_cols).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_window_dir",
        default="/project/hrao/GDELT/results/event_window_ablation",
    )
    parser.add_argument(
        "--topk_root_dir",
        default="/project/hrao/GDELT/results/topk_ablation",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/ablation_tables",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_table = make_event_window_ablation(args.event_window_dir)
    event_gain = make_event_window_gain(event_table)
    topk_table = make_topk_ablation(args.topk_root_dir)
    topk_best = make_topk_best(topk_table)

    outputs = {
        "table_event_window_ablation.csv": event_table,
        "table_event_window_gain.csv": event_gain,
        "table_topk_ablation.csv": topk_table,
        "table_topk_best.csv": topk_best,
    }

    print("Saved:")
    for name, df in outputs.items():
        path = out_dir / name
        df.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
