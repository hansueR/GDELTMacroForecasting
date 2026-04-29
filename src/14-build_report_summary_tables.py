from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


def parse_horizon_from_target(x):
    m = re.search(r"_(\d+)d", str(x))
    if m:
        return int(m.group(1))
    return np.nan


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def pick_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def get_metric(df, names):
    col = pick_col(df, names)
    if col is None:
        return np.nan
    return df[col]


def summarize_return_by_year(df, has_feature_set=True):
    group_cols = ["model_type", "asset", "horizon", "target"]
    if has_feature_set and "feature_set" in df.columns:
        group_cols.insert(1, "feature_set")
    if "risk_target" in df.columns:
        group_cols.append("risk_target")

    metric_cols = [
        "nll_mean",
        "nll_median",
        "directional_acc",
        "auc",
        "mu_mae",
        "mu_rmse",
        "risk_mae",
        "risk_rmse",
        "sigma_mean",
    ]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_test_total"] = int(g["n_test"].sum())
        row["n_splits"] = int(g["test_year"].nunique()) if "test_year" in g.columns else int(len(g))

        for m in metric_cols:
            if m in g.columns:
                row[m + "_weighted"] = weighted_mean(g[m], g["n_test"])

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_return_table(df, model_group, default_feature_set=None):
    df = df.copy()

    if "horizon" not in df.columns and "target" in df.columns:
        df["horizon"] = df["target"].apply(parse_horizon_from_target)

    if "test_year" in df.columns and "n_test" in df.columns and "n_test_total" not in df.columns:
        df = summarize_return_by_year(df, has_feature_set=("feature_set" in df.columns))

    out = pd.DataFrame(index=df.index)
    out["model_group"] = model_group
    out["model_type"] = df["model_type"] if "model_type" in df.columns else model_group
    out["feature_set"] = df["feature_set"] if "feature_set" in df.columns else default_feature_set
    out["asset"] = df["asset"]
    out["horizon"] = df["horizon"].astype(int)
    out["target"] = df["target"]
    out["risk_target"] = df["risk_target"] if "risk_target" in df.columns else ""

    out["n_test_total"] = get_metric(df, ["n_test_total"])
    out["n_splits"] = get_metric(df, ["n_splits"])

    out["nll_mean"] = get_metric(df, ["nll_mean", "nll_mean_weighted"])
    out["nll_median"] = get_metric(df, ["nll_median", "nll_median_weighted"])
    out["directional_acc"] = get_metric(df, ["directional_acc_mean", "directional_acc_weighted", "directional_acc"])
    out["auc"] = get_metric(df, ["auc_mean", "auc_weighted", "auc"])
    out["mu_mae"] = get_metric(df, ["mu_mae_mean", "mu_mae_weighted", "mu_mae"])
    out["mu_rmse"] = get_metric(df, ["mu_rmse_mean", "mu_rmse_weighted", "mu_rmse"])
    out["risk_mae"] = get_metric(df, ["risk_mae_mean", "risk_mae_weighted", "risk_mae"])
    out["risk_rmse"] = get_metric(df, ["risk_rmse_mean", "risk_rmse_weighted", "risk_rmse"])
    out["sigma_mean"] = get_metric(df, ["sigma_mean", "sigma_mean_weighted"])

    return out.sort_values(["asset", "horizon", "model_group", "model_type", "feature_set"]).reset_index(drop=True)


def summarize_risk_by_year(df):
    df = df.copy()
    if "horizon" not in df.columns:
        df["horizon"] = df["target"].apply(parse_horizon_from_target)

    group_cols = ["model_type", "feature_set", "asset", "horizon", "target"]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_test_total"] = int(g["n_test"].sum())
        row["n_splits"] = int(g["test_year"].nunique()) if "test_year" in g.columns else int(len(g))
        row["mae"] = weighted_mean(g["mae"], g["n_test"])
        row["rmse"] = weighted_mean(g["rmse"], g["n_test"])
        row["directional_acc"] = weighted_mean(g["directional_acc"], g["n_test"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["asset", "horizon", "model_type", "feature_set"]).reset_index(drop=True)


def normalize_risk_table(path):
    df = pd.read_csv(path)
    if "horizon" not in df.columns:
        df["horizon"] = df["target"].apply(parse_horizon_from_target)

    if "test_year" in df.columns and "n_test" in df.columns:
        return summarize_risk_by_year(df)

    out = df.copy()
    rename = {
        "mae_mean": "mae",
        "rmse_mean": "rmse",
        "directional_acc_mean": "directional_acc",
    }
    out = out.rename(columns=rename)
    keep = [
        "model_type",
        "feature_set",
        "asset",
        "horizon",
        "target",
        "n_test_total",
        "mae",
        "rmse",
        "directional_acc",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values(["asset", "horizon", "model_type", "feature_set"]).reset_index(drop=True)


def make_model_key(model_type, feature_set, model_group=None):
    model_type = str(model_type)
    feature_set = str(feature_set)

    if model_group == "main_model":
        return "TB"

    if model_type == "naive":
        return "NV"
    if model_type == "ewma":
        return "EW"
    if model_type == "ridge" and feature_set == "price_only":
        return "RP"
    if model_type == "ridge" and feature_set == "event_only":
        return "RE"
    if model_type == "ridge" and feature_set == "price_plus_event":
        return "RPE"

    return ""


def get_risk_source(target):
    target = str(target)
    if "rv_return" in target:
        return "rv_return"
    if "rv_ohlc" in target:
        return "rv_ohlc"
    return ""


def merge_direct_risk_metrics(all_model_results, risk_table):
    """
    Merge direct realized-volatility forecasting results into the main report table.

    This adds direct risk metrics for:
      NV  = naive risk baseline
      EW  = EWMA risk baseline
      RP  = ridge price-only
      RE  = ridge event-only
      RPE = ridge price+event

    For TB, direct_rv_return_mae/rmse are copied from the main model's auxiliary
    risk_mae/risk_rmse when the risk target is target_rv_return_*.
    """
    out = all_model_results.copy()

    direct_cols = [
        "direct_rv_return_mae",
        "direct_rv_return_rmse",
        "direct_rv_return_directional_acc",
        "direct_rv_ohlc_mae",
        "direct_rv_ohlc_rmse",
        "direct_rv_ohlc_directional_acc",
    ]
    for col in direct_cols:
        if col not in out.columns:
            out[col] = np.nan

    out["model_key"] = out.apply(
        lambda r: make_model_key(
            r.get("model_type", ""),
            r.get("feature_set", ""),
            r.get("model_group", ""),
        ),
        axis=1,
    )

    if risk_table is not None and not risk_table.empty:
        risk = risk_table.copy()
        if "horizon" not in risk.columns and "target" in risk.columns:
            risk["horizon"] = risk["target"].apply(parse_horizon_from_target)

        risk["model_key"] = risk.apply(
            lambda r: make_model_key(
                r.get("model_type", ""),
                r.get("feature_set", ""),
                "baseline",
            ),
            axis=1,
        )
        risk["risk_source"] = risk["target"].apply(get_risk_source)
        risk = risk[
            risk["model_key"].ne("")
            & risk["risk_source"].isin(["rv_return", "rv_ohlc"])
        ].copy()

        if not risk.empty:
            pivot = risk.pivot_table(
                index=["model_key", "asset", "horizon"],
                columns="risk_source",
                values=["mae", "rmse", "directional_acc"],
                aggfunc="first",
            )
            pivot.columns = [
                f"direct_{risk_source}_{metric}"
                for metric, risk_source in pivot.columns
            ]
            pivot = pivot.reset_index()

            out = out.merge(
                pivot,
                on=["model_key", "asset", "horizon"],
                how="left",
                suffixes=("", "_from_risk"),
            )

            for col in direct_cols:
                from_col = f"{col}_from_risk"
                if from_col in out.columns:
                    out[col] = out[col].combine_first(out[from_col])
                    out = out.drop(columns=[from_col])

    # Fill TB's direct RV-return risk metrics from the main model auxiliary risk output.
    if "risk_target" in out.columns:
        tb_mask = (
            out["model_group"].eq("main_model")
            & out["risk_target"].astype(str).str.contains("rv_return", na=False)
        )
        if "risk_mae" in out.columns:
            out.loc[tb_mask, "direct_rv_return_mae"] = out.loc[
                tb_mask, "direct_rv_return_mae"
            ].combine_first(out.loc[tb_mask, "risk_mae"])
        if "risk_rmse" in out.columns:
            out.loc[tb_mask, "direct_rv_return_rmse"] = out.loc[
                tb_mask, "direct_rv_return_rmse"
            ].combine_first(out.loc[tb_mask, "risk_rmse"])

    out = out.drop(columns=["model_key"])
    return out


def make_baseline_event_gain(all_model_results):
    df = all_model_results.copy()
    df = df[
        (df["model_group"] == "baseline")
        & (df["model_type"] == "ridge")
    ].copy()

    key_cols = ["asset", "horizon", "target"]

    price = df[df["feature_set"] == "price_only"].copy()
    event = df[df["feature_set"] == "price_plus_event"].copy()

    if price.empty or event.empty:
        return pd.DataFrame()

    merged = event.merge(price, on=key_cols, suffixes=("_price_plus_event", "_price_only"))

    out = merged[key_cols].copy()
    out["nll_price_only"] = merged["nll_mean_price_only"]
    out["nll_price_plus_event"] = merged["nll_mean_price_plus_event"]
    out["nll_improvement"] = out["nll_price_only"] - out["nll_price_plus_event"]

    out["auc_price_only"] = merged["auc_price_only"]
    out["auc_price_plus_event"] = merged["auc_price_plus_event"]
    out["auc_gain"] = out["auc_price_plus_event"] - out["auc_price_only"]

    out["directional_acc_price_only"] = merged["directional_acc_price_only"]
    out["directional_acc_price_plus_event"] = merged["directional_acc_price_plus_event"]
    out["directional_acc_gain"] = (
        out["directional_acc_price_plus_event"] - out["directional_acc_price_only"]
    )

    return out.sort_values(key_cols).reset_index(drop=True)


def make_main_vs_best_baseline(all_model_results):
    df = all_model_results.copy()
    key_cols = ["asset", "horizon", "target"]

    baseline = df[df["model_group"] == "baseline"].copy()
    main = df[df["model_group"] == "main_model"].copy()

    baseline = baseline.dropna(subset=["nll_mean"])
    main = main.dropna(subset=["nll_mean"])

    if baseline.empty or main.empty:
        return pd.DataFrame()

    best_baseline = (
        baseline
        .sort_values(key_cols + ["nll_mean"])
        .groupby(key_cols, as_index=False)
        .first()
    )

    merged = main.merge(best_baseline, on=key_cols, suffixes=("_main", "_best_baseline"))

    out = merged[key_cols].copy()
    out["main_model_type"] = merged["model_type_main"]
    out["best_baseline_model_type"] = merged["model_type_best_baseline"]
    out["best_baseline_feature_set"] = merged["feature_set_best_baseline"]

    out["nll_main"] = merged["nll_mean_main"]
    out["nll_best_baseline"] = merged["nll_mean_best_baseline"]
    out["nll_improvement_over_best_baseline"] = out["nll_best_baseline"] - out["nll_main"]

    out["auc_main"] = merged["auc_main"]
    out["auc_best_baseline"] = merged["auc_best_baseline"]
    out["auc_gain_over_best_baseline"] = out["auc_main"] - out["auc_best_baseline"]

    out["directional_acc_main"] = merged["directional_acc_main"]
    out["directional_acc_best_baseline"] = merged["directional_acc_best_baseline"]
    out["directional_acc_gain_over_best_baseline"] = (
        out["directional_acc_main"] - out["directional_acc_best_baseline"]
    )

    out["risk_mae_main"] = merged["risk_mae_main"] if "risk_mae_main" in merged.columns else np.nan
    out["risk_rmse_main"] = merged["risk_rmse_main"] if "risk_rmse_main" in merged.columns else np.nan

    return out.sort_values(key_cols).reset_index(drop=True)


def load_ablation_tables(ablation_dir):
    ablation_dir = Path(ablation_dir)
    files = {
        "ablation_event_window": ablation_dir / "table_event_window_ablation.csv",
        "ablation_event_window_gain": ablation_dir / "table_event_window_gain.csv",
        "ablation_topk": ablation_dir / "table_topk_ablation.csv",
        "ablation_topk_best": ablation_dir / "table_topk_best.csv",
    }

    tables = {}
    for name, path in files.items():
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            print(f"WARNING: missing ablation table: {path}")
    return tables


def write_readme(out_dir, paths):
    text = f"""# Report summary tables

This directory contains compact tables for the final report.

Main inputs:
- Baseline return distribution: {paths.get("baseline_distribution")}
- Main two-branch model: {paths.get("main_model")}
- Risk baseline: {paths.get("risk_baseline")}
- Ablation directory: {paths.get("ablation_dir")}

Main outputs:
- report_summary_tables.xlsx
- report_all_model_results.csv, including return-distribution metrics and merged direct risk metrics when available
- report_main_vs_best_baseline.csv
- report_baseline_event_gain.csv
- report_risk_baseline.csv
- copied ablation CSV tables if available

Metric interpretation:
- Lower NLL is better.
- Higher directional accuracy is better.
- Higher AUC is better.
- Lower MAE/RMSE is better.
- Positive improvement/gain columns mean the later model or feature set is better than the comparison baseline.
"""
    (out_dir / "README_report_summary_tables.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/project/hrao/GDELT")
    parser.add_argument("--output_dir", default="/project/hrao/GDELT/results/report_summary_tables")

    parser.add_argument("--baseline_distribution", default="")
    parser.add_argument("--main_model", default="")
    parser.add_argument("--risk_baseline", default="")
    parser.add_argument("--ablation_dir", default="")

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_distribution) if args.baseline_distribution else first_existing([
        root / "results/distribution_baselines_v2/distribution_baselines_summary.csv",
        root / "results/distribution_baselines_v2/distribution_baselines_by_year.csv",
    ])

    main_path = Path(args.main_model) if args.main_model else first_existing([
        root / "results/two_branch_distribution_v1/two_branch_summary.csv",
        root / "results/two_branch_distribution_v1/two_branch_by_year.csv",
    ])

    risk_path = Path(args.risk_baseline) if args.risk_baseline else first_existing([
        root / "results/risk_baseline_v1/walk_forward_results_all.csv",
        root / "results/risk_baseline_v1/walk_forward_summary_all.csv",
        root / "results/risk_baseline_v1/walk_forward_results_ridge.csv",
        root / "results/risk_baseline_v1/walk_forward_summary_ridge.csv",
    ])

    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else root / "results/ablation_tables"

    if baseline_path is None:
        raise FileNotFoundError("Missing baseline distribution results. Run scripts_AI/7-walk_forward_distribution_baseline.sh first.")

    if main_path is None:
        raise FileNotFoundError("Missing main model results. Run scripts_AI/10-train_two_branch_distribution_model.sh first.")

    baseline_df = normalize_return_table(
        pd.read_csv(baseline_path),
        model_group="baseline",
        default_feature_set="unknown",
    )

    main_df = normalize_return_table(
        pd.read_csv(main_path),
        model_group="main_model",
        default_feature_set="price_plus_event_sequence",
    )

    all_models = pd.concat([baseline_df, main_df], ignore_index=True)

    if risk_path is not None:
        risk_table = normalize_risk_table(risk_path)
    else:
        print("WARNING: missing risk baseline results; risk table is skipped.")
        risk_table = pd.DataFrame()

    all_models = merge_direct_risk_metrics(all_models, risk_table)
    main_vs_best = make_main_vs_best_baseline(all_models)
    baseline_event_gain = make_baseline_event_gain(all_models)

    ablation_tables = load_ablation_tables(ablation_dir)

    outputs = {
        "all_model_results": all_models,
        "main_vs_best_baseline": main_vs_best,
        "baseline_event_gain": baseline_event_gain,
        "risk_baseline": risk_table,
    }
    outputs.update(ablation_tables)

    for name, df in outputs.items():
        if df is not None and not df.empty:
            df.to_csv(out_dir / f"report_{name}.csv", index=False)

    xlsx_path = out_dir / "report_summary_tables.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            for name, df in outputs.items():
                if df is not None and not df.empty:
                    sheet = name[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Saved Excel workbook: {xlsx_path}")
    except Exception as e:
        print(f"WARNING: could not write Excel workbook: {e}")
        print("CSV files were still written.")

    write_readme(
        out_dir,
        {
            "baseline_distribution": baseline_path,
            "main_model": main_path,
            "risk_baseline": risk_path,
            "ablation_dir": ablation_dir,
        },
    )

    print("\nSaved report tables to:")
    print(out_dir)

    print("\nFiles:")
    for p in sorted(out_dir.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
