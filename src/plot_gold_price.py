import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--price_file",
        default="/project/hrao/GDELT/asset_prices/daily_asset_prices.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/report_figures",
    )
    parser.add_argument(
        "--asset",
        default="Gold",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of recent months to plot.",
    )
    args = parser.parse_args()

    price_file = Path(args.price_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not price_file.exists():
        raise FileNotFoundError(f"Missing price file: {price_file}")

    df = pd.read_parquet(price_file)
    df["date"] = pd.to_datetime(df["date"])

    required_cols = ["date", "asset", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    data = df[df["asset"] == args.asset].copy()
    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError(
            f"No rows found for asset={args.asset}. "
            f"Available assets: {sorted(df['asset'].unique())}"
        )

    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(months=args.months)

    recent = data[data["date"] >= start_date].copy()
    recent = recent.sort_values("date").reset_index(drop=True)

    if recent.empty:
        raise ValueError("No data found in the selected recent time window.")

    # 给 y 轴留一点上下边距，这样之后在 PPT 上做标注更舒服
    y_min = recent["close"].min()
    y_max = recent["close"].max()
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0

    fig, ax = plt.subplots(figsize=(11, 5.2))

    ax.plot(recent["date"], recent["close"], linewidth=1.8)

    ax.set_title(f"{args.asset} Daily Close Price (Last {args.months} Months)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price", fontsize=12)

    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # 只保留左边和下边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # 网格线只保留淡淡的横向，方便看价格位置，但不喧宾夺主
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    # x 轴按月显示，适合 6 个月窗口
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()

    out_png = output_dir / "gold_close_price_recent6m.png"
    out_pdf = output_dir / "gold_close_price_recent6m.pdf"

    fig.savefig(out_png, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:")
    print(out_png)
    print(out_pdf)

    print("\nDate range:")
    print("Start:", recent["date"].min())
    print("End:  ", recent["date"].max())
    print("Rows: ", len(recent))


if __name__ == "__main__":
    main()