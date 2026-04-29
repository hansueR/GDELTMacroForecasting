from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def parse_horizon_from_target(target):
    m = re.search(r"_(\d+)d$", str(target))
    if m is None:
        return np.nan
    return int(m.group(1))


def find_existing_path(user_path, candidates):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"File not found: {p}")

    for c in candidates:
        p = Path(c)
        if p.exists():
            return p

    return None


def summarize_distribution_by_year(df):
    required = [
        "model_type",
        "feature_set",
        "asset",
        "horizon",
        "target",
        "n_test",
        "nll_mean",
        "nll_median",
        "directional_acc",
        "auc",
        "mu_mae",
        "mu_rmse",
        "sigma_mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Distribution result file is missing columns: {missing}")

    group_cols = ["model_type", "feature_set", "asset", "horizon", "target"]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_test_total"] = int(g["n_test"].sum())
        row["n_splits"] = int(g["test_year"].nunique()) if "test_year" in g.columns else int(len(g))

        for metric in [
            "nll_mean",
            "nll_median",
            "directional_acc",
            "auc",
            "mu_mae",
            "mu_rmse",
            "sigma_mean",
        ]:
            row[f"{metric}_weighted"] = weighted_mean(g[metric], g["n_test"])

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["asset", "horizon", "model_type", "feature_set"]).reset_index(drop=True)
    return out


def make_distribution_event_gain(dist_summary):
    """
    Compare ridge price_plus_event against ridge price_only.
    For NLL/MAE/RMSE, positive improvement means price_plus_event is better.
    For accuracy/AUC, positive gain means price_plus_event is better.
    """
    df = dist_summary.copy()
    df = df[df["model_type"] == "ridge"].copy()

    key_cols = ["asset", "horizon", "target"]

    price = df[df["feature_set"] == "price_only"].copy()
    event = df[df["feature_set"] == "price_plus_event"].copy()

    if price.empty or event.empty:
        return pd.DataFrame()

    merged = event.merge(
        price,
        on=key_cols,
        suffixes=("_price_plus_event", "_price_only"),
    )

    out = merged[key_cols].copy()
    out["nll_improvement"] = (
        merged["nll_mean_weighted_price_only"]
        - merged["nll_mean_weighted_price_plus_event"]
    )
    out["directional_acc_gain"] = (
        merged["directional_acc_weighted_price_plus_event"]
        - merged["directional_acc_weighted_price_only"]
    )
    out["auc_gain"] = (
        merged["auc_weighted_price_plus_event"]
        - merged["auc_weighted_price_only"]
    )
    out["mu_mae_improvement"] = (
        merged["mu_mae_weighted_price_only"]
        - merged["mu_mae_weighted_price_plus_event"]
    )
    out["mu_rmse_improvement"] = (
        merged["mu_rmse_weighted_price_only"]
        - merged["mu_rmse_weighted_price_plus_event"]
    )

    out["nll_price_only"] = merged["nll_mean_weighted_price_only"]
    out["nll_price_plus_event"] = merged["nll_mean_weighted_price_plus_event"]
    out["auc_price_only"] = merged["auc_weighted_price_only"]
    out["auc_price_plus_event"] = merged["auc_weighted_price_plus_event"]

    out = out.sort_values(["asset", "horizon"]).reset_index(drop=True)
    return out


def summarize_risk_by_year(df):
    required = [
        "feature_set",
        "asset",
        "target",
        "model_type",
        "n_test",
        "mae",
        "rmse",
        "directional_acc",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Risk result file is missing columns: {missing}")

    df = df.copy()
    if "horizon" not in df.columns:
        df["horizon"] = df["target"].apply(parse_horizon_from_target)

    group_cols = ["model_type", "feature_set", "asset", "horizon", "target"]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_test_total"] = int(g["n_test"].sum())
        row["n_splits"] = int(g["test_year"].nunique()) if "test_year" in g.columns else int(len(g))
        row["mae_weighted"] = weighted_mean(g["mae"], g["n_test"])
        row["rmse_weighted"] = weighted_mean(g["rmse"], g["n_test"])
        row["directional_acc_weighted"] = weighted_mean(g["directional_acc"], g["n_test"])
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["asset", "horizon", "model_type", "feature_set"]).reset_index(drop=True)
    return out


def make_risk_event_gain(risk_summary):
    """
    Compare ridge price_plus_event against ridge price_only for auxiliary risk proxies.
    Positive improvement means price_plus_event is better.
    """
    df = risk_summary.copy()
    df = df[df["model_type"] == "ridge"].copy()

    key_cols = ["asset", "horizon", "target"]

    price = df[df["feature_set"] == "price_only"].copy()
    event = df[df["feature_set"] == "price_plus_event"].copy()

    if price.empty or event.empty:
        return pd.DataFrame()

    merged = event.merge(
        price,
        on=key_cols,
        suffixes=("_price_plus_event", "_price_only"),
    )

    out = merged[key_cols].copy()
    out["mae_improvement"] = (
        merged["mae_weighted_price_only"]
        - merged["mae_weighted_price_plus_event"]
    )
    out["rmse_improvement"] = (
        merged["rmse_weighted_price_only"]
        - merged["rmse_weighted_price_plus_event"]
    )
    out["directional_acc_gain"] = (
        merged["directional_acc_weighted_price_plus_event"]
        - merged["directional_acc_weighted_price_only"]
    )

    out["mae_price_only"] = merged["mae_weighted_price_only"]
    out["mae_price_plus_event"] = merged["mae_weighted_price_plus_event"]
    out["rmse_price_only"] = merged["rmse_weighted_price_only"]
    out["rmse_price_plus_event"] = merged["rmse_weighted_price_plus_event"]

    out = out.sort_values(["asset", "horizon", "target"]).reset_index(drop=True)
    return out


def write_readme(out_dir, dist_path, risk_path):
    text = f"""# Module 4 evaluation outputs

This module does not train new models.

Inputs:
- Distribution baseline results: {dist_path}
- Risk auxiliary baseline results: {risk_path if risk_path is not None else "not found / skipped"}

Outputs:
- table_distribution_baselines.csv
- table_distribution_event_gain.csv
- table_risk_auxiliary.csv
- table_risk_event_gain.csv

Metric interpretation:
- Lower NLL is better.
- Higher directional accuracy is better.
- Higher AUC is better.
- Lower MAE/RMSE is better.
- In event-gain tables, positive improvement/gain means price_plus_event is better than price_only.
"""
    (out_dir / "README_module4.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution_by_year",
        default="/project/hrao/GDELT/results/distribution_baselines_v2/distribution_baselines_by_year.csv",
    )
    parser.add_argument(
        "--risk_by_year",
        default="",
        help=(
            "Optional path to risk baseline by-year CSV. "
            "If empty, the script tries common default locations."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/module4_evaluation_tables",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_path = Path(args.distribution_by_year)
    if not dist_path.exists():
        raise FileNotFoundError(
            f"Distribution result file not found: {dist_path}. "
            "Run scripts_AI/7-walk_forward_distribution_baseline.sh first."
        )

    risk_path = find_existing_path(
        args.risk_by_year,
        candidates=[
            "/project/hrao/GDELT/results/risk_baseline_v1/walk_forward_results_ridge.csv",
            "/project/hrao/GDELT/results/risk_baselines_v1/walk_forward_results_ridge.csv",
            "/project/hrao/GDELT/results/baseline_v1/walk_forward_results_ridge.csv",
        ],
    )

    dist_by_year = pd.read_csv(dist_path)
    dist_summary = summarize_distribution_by_year(dist_by_year)
    dist_gain = make_distribution_event_gain(dist_summary)

    dist_summary.to_csv(out_dir / "table_distribution_baselines.csv", index=False)
    dist_gain.to_csv(out_dir / "table_distribution_event_gain.csv", index=False)

    if risk_path is not None:
        risk_by_year = pd.read_csv(risk_path)
        risk_summary = summarize_risk_by_year(risk_by_year)
        risk_gain = make_risk_event_gain(risk_summary)

        risk_summary.to_csv(out_dir / "table_risk_auxiliary.csv", index=False)
        risk_gain.to_csv(out_dir / "table_risk_event_gain.csv", index=False)
    else:
        print("WARNING: No risk baseline file found. Risk auxiliary tables are skipped.")

    write_readme(out_dir, dist_path, risk_path)

    print("\nSaved module 4 evaluation tables to:")
    print(out_dir)

    print("\nFiles:")
    for p in sorted(out_dir.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
