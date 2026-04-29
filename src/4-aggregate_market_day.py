from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def assign_market_day_ny(ts_series: pd.Series, close_hour=16) -> pd.Series:
    """
    ts_series: timezone-aware America/New_York timestamps
    rule:
      - before 16:00 NY time -> same calendar day
      - at/after 16:00 NY time -> next calendar day
    """
    local_ts = ts_series
    day = local_ts.dt.floor("D")
    after_close = local_ts.dt.hour >= close_hour
    market_day = day.where(~after_close, day + pd.Timedelta(days=1))
    return market_day.dt.date.astype("string")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="directory of clean daily parquet files")
    parser.add_argument("--output_file", required=True, help="output parquet path")
    parser.add_argument("--close_hour", type=int, default=16, help="market close hour in NY time")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    parts = []
    for f in files:
        df = pd.read_parquet(f)

        # 保险起见，确保时间列存在
        if "date_added_ts_ny" not in df.columns:
            raise ValueError(f"{f} missing date_added_ts_ny")

        df["market_day_ny"] = assign_market_day_ny(df["date_added_ts_ny"], close_hour=args.close_hour)

        # 一些常用辅助特征
        df["goldstein_pos"] = df["GoldsteinScale"].clip(lower=0)
        df["goldstein_neg_abs"] = (-df["GoldsteinScale"].clip(upper=0))
        df["is_conflict"] = df["QuadClass"].isin(["3", "4"]).astype("int8")
        df["is_coop"] = df["QuadClass"].isin(["1", "2"]).astype("int8")
        df["is_material"] = df["QuadClass"].isin(["2", "4"]).astype("int8")
        df["is_verbal"] = df["QuadClass"].isin(["1", "3"]).astype("int8")

        parts.append(df)

    full = pd.concat(parts, ignore_index=True)

    # 最基础的一版：按 市场日 × 事件粗类 × 事件地理国家 聚合
    agg = (
        full.groupby(
            ["market_day_ny", "EventRootCode", "ActionGeo_CountryCode"],
            dropna=False
        )
        .agg(
            n_events=("GLOBALEVENTID", "count"),
            n_unique_urls=("SOURCEURL", "nunique"),
            n_unique_event_codes=("EventCode", "nunique"),

            goldstein_mean=("GoldsteinScale", "mean"),
            goldstein_sum=("GoldsteinScale", "sum"),
            goldstein_pos_sum=("goldstein_pos", "sum"),
            goldstein_neg_abs_sum=("goldstein_neg_abs", "sum"),

            mentions_sum=("NumMentions", "sum"),
            sources_sum=("NumSources", "sum"),
            articles_sum=("NumArticles", "sum"),

            tone_mean=("AvgTone", "mean"),
            tone_std=("AvgTone", "std"),

            conflict_events=("is_conflict", "sum"),
            coop_events=("is_coop", "sum"),
            material_events=("is_material", "sum"),
            verbal_events=("is_verbal", "sum"),
        )
        .reset_index()
        .sort_values(["market_day_ny", "EventRootCode", "ActionGeo_CountryCode"])
        .reset_index(drop=True)
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)

    print("input files:", len(files))
    print("raw rows:", len(full))
    print("aggregated shape:", agg.shape)
    print("output:", out_path)
    print("\nhead:")
    print(agg.head(10).to_string(index=False))

if __name__ == "__main__":
    main()