from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd


def safe_name(x):
    x = str(x)
    x = x.strip()
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = x.strip("_")
    if x == "":
        x = "UNK"
    return x


def map_to_next_trading_day(event_dates, trading_calendar):
    """
    Map each event market day to the next available trading day.
    This fixes weekend / holiday dates.
    """
    event_dates = pd.to_datetime(event_dates)
    calendar = pd.to_datetime(trading_calendar).sort_values().drop_duplicates()

    idx = np.searchsorted(calendar.values, event_dates.values, side="left")
    valid = idx < len(calendar)

    mapped = pd.Series(pd.NaT, index=event_dates.index, dtype="datetime64[ns]")
    mapped.loc[valid] = calendar.iloc[idx[valid]].values
    return mapped


def build_daily_event_features(event_long, trading_calendar, top_root_k=20, top_country_k=20):
    event_long = event_long.copy()

    event_long["market_day_ny"] = pd.to_datetime(event_long["market_day_ny"])
    event_long["date"] = map_to_next_trading_day(
        event_long["market_day_ny"],
        trading_calendar
    )

    event_long = event_long.dropna(subset=["date"]).copy()

    event_long["EventRootCode"] = event_long["EventRootCode"].fillna("UNK").astype(str)
    event_long["ActionGeo_CountryCode"] = (
        event_long["ActionGeo_CountryCode"]
        .fillna("UNK")
        .astype(str)
    )

    # Weighted quantities for mean-like fields
    event_long["tone_weighted_sum"] = event_long["tone_mean"].fillna(0) * event_long["n_events"]
    event_long["goldstein_mean_weighted_sum"] = (
        event_long["goldstein_mean"].fillna(0) * event_long["n_events"]
    )

    # Low-dimensional daily global features
    daily = (
        event_long
        .groupby("date")
        .agg(
            event_n_total=("n_events", "sum"),
            event_unique_urls_sum=("n_unique_urls", "sum"),
            event_unique_event_codes_sum=("n_unique_event_codes", "sum"),
            event_goldstein_sum=("goldstein_sum", "sum"),
            event_goldstein_pos_sum=("goldstein_pos_sum", "sum"),
            event_goldstein_neg_abs_sum=("goldstein_neg_abs_sum", "sum"),
            event_mentions_sum=("mentions_sum", "sum"),
            event_sources_sum=("sources_sum", "sum"),
            event_articles_sum=("articles_sum", "sum"),
            event_conflict_count=("conflict_events", "sum"),
            event_coop_count=("coop_events", "sum"),
            event_material_count=("material_events", "sum"),
            event_verbal_count=("verbal_events", "sum"),
            event_tone_weighted_sum=("tone_weighted_sum", "sum"),
            event_goldstein_mean_weighted_sum=("goldstein_mean_weighted_sum", "sum"),
        )
        .reset_index()
    )

    daily["event_tone_weighted_mean"] = (
        daily["event_tone_weighted_sum"] / daily["event_n_total"].replace(0, np.nan)
    )

    daily["event_goldstein_weighted_mean"] = (
        daily["event_goldstein_mean_weighted_sum"] / daily["event_n_total"].replace(0, np.nan)
    )

    daily = daily.drop(
        columns=["event_tone_weighted_sum", "event_goldstein_mean_weighted_sum"]
    )

    # EventRootCode x Country count features.
    # This replaces the older separate root-only and country-only features.
    # The goal is to let the model observe where each event type occurs.
    #
    # Example output columns:
    #   event_root_14_country_US_count
    #   event_root_14_country_CH_count
    #   event_root_19_country_IR_count
    top_roots = (
        event_long
        .groupby("EventRootCode")["n_events"]
        .sum()
        .sort_values(ascending=False)
        .head(top_root_k)
        .index
    )

    top_countries = (
        event_long
        .groupby("ActionGeo_CountryCode")["n_events"]
        .sum()
        .sort_values(ascending=False)
        .head(top_country_k)
        .index
    )

    pair_df = event_long[
        event_long["EventRootCode"].isin(top_roots)
        & event_long["ActionGeo_CountryCode"].isin(top_countries)
    ].copy()

    pair_df["root_country_key"] = (
        "root_"
        + pair_df["EventRootCode"].map(safe_name)
        + "_country_"
        + pair_df["ActionGeo_CountryCode"].map(safe_name)
    )

    pair_pivot = (
        pair_df
        .groupby(["date", "root_country_key"])["n_events"]
        .sum()
        .unstack(fill_value=0)
    )

    pair_pivot.columns = [
        f"event_{c}_count" for c in pair_pivot.columns
    ]

    pair_pivot = pair_pivot.reset_index()

    # Merge event features
    event_features = daily.merge(pair_pivot, on="date", how="outer")

    # Reindex to trading calendar
    calendar_df = pd.DataFrame({"date": pd.to_datetime(trading_calendar).sort_values().drop_duplicates()})
    event_features = calendar_df.merge(event_features, on="date", how="left")

    event_cols = [c for c in event_features.columns if c != "date"]

    # Count/sum features: missing means zero event count
    event_features[event_cols] = event_features[event_cols].fillna(0)

    # Add rolling event windows
    base_event_cols = event_cols.copy()
    for window in [3, 5]:
        for col in base_event_cols:
            event_features[f"{col}_roll{window}d"] = (
                event_features[col]
                .rolling(window=window, min_periods=1)
                .sum()
            )

    return event_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--price_file",
        default="/project/hrao/GDELT/asset_prices/daily_asset_prices.parquet"
    )
    parser.add_argument(
        "--event_long_dir",
        default="/project/hrao/GDELT/features/event_long"
    )
    parser.add_argument(
        "--output_file",
        default="/project/hrao/GDELT/modeling/model_table.parquet"
    )
    parser.add_argument("--top_root_k", type=int, default=20)
    parser.add_argument("--top_country_k", type=int, default=20)

    args = parser.parse_args()

    price_file = Path(args.price_file)
    event_long_dir = Path(args.event_long_dir)

    if not price_file.exists():
        raise FileNotFoundError(f"Missing price file: {price_file}")

    event_files = sorted(event_long_dir.glob("events_market_day_long_*.parquet"))
    if not event_files:
        raise FileNotFoundError(f"No event long parquet files found in {event_long_dir}")

    print("Reading price file:", price_file)
    prices = pd.read_parquet(price_file)
    prices["date"] = pd.to_datetime(prices["date"])

    trading_calendar = prices["date"].sort_values().drop_duplicates()

    print("Reading event files:", len(event_files))
    event_parts = []
    for f in event_files:
        event_parts.append(pd.read_parquet(f))

    event_long = pd.concat(event_parts, ignore_index=True)

    print("Event long shape:", event_long.shape)

    event_features = build_daily_event_features(
        event_long=event_long,
        trading_calendar=trading_calendar,
        top_root_k=args.top_root_k,
        top_country_k=args.top_country_k,
    )

    print("Event feature shape:", event_features.shape)

    model = prices.merge(event_features, on="date", how="left")

    event_cols = [c for c in model.columns if c.startswith("event_")]
    model[event_cols] = model[event_cols].fillna(0)

    # Drop rows without enough historical price features or future targets.
    # Keep all 1d/3d/5d targets in the table, but for the first baseline you can train per target.
    model = model.sort_values(["asset", "date"]).reset_index(drop=True)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Model table shape:", model.shape)

    print("\nAssets:")
    print(model.groupby("asset")["date"].agg(["min", "max", "count"]))

    print("\nTarget non-null counts:")
    target_cols = [
        c for c in model.columns
        if c.startswith("target_return") or c.startswith("target_rv")
    ]
    target_cols = sorted(target_cols)
    print(model[target_cols].notna().sum())

    print("\nNumber of event features:", len(event_cols))
    print("\nSample columns:")
    print(model.columns[:80].tolist())

    print("\nHead:")
    print(model.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
