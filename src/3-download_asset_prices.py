from pathlib import Path
import argparse
import time
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "Missing package: yfinance. Install it with:\n"
        "pip install yfinance"
    ) from e


DEFAULT_TICKERS = {
    "QQQ": "QQQ",
    "Gold": "GLD",
    "WTI_Oil": "USO",
}


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns.
    This function converts them to normal columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]
    return df


def download_one_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")

            df = flatten_yfinance_columns(df)
            df = df.reset_index()

            # Normalize column names
            rename_map = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            df = df.rename(columns=rename_map)

            required_cols = ["date", "open", "high", "low", "close", "volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{ticker} missing columns: {missing}")

            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]

            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype("string")

            numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "high", "low", "close", "adj_close"])
            df = df.sort_values("date").reset_index(drop=True)

            return df

        except Exception as e:
            last_error = e
            print(f"[WARNING] {ticker} download failed on attempt {attempt}: {e}")
            time.sleep(5)

    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts") from last_error


def add_price_features(df: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    """
    Add price-side features, future return targets, and future realized-volatility targets.

    Future h-day return target:
      target_return_{h}d = sum of next h daily log returns
                           = log(adj_close_{t+h} / adj_close_t)

    Realized volatility proxy:
      target_rv_{h}d = sqrt(mean of squared future log returns over next h trading days)

    For h = 1:
      target_rv_1d = abs(next-day log return)

    Also compute an OHLC-based daily volatility proxy using Parkinson volatility:
      sqrt( (log(high / low)^2) / (4 log 2) )
    """
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    out_parts = []

    for asset, g in df.groupby("asset", sort=False):
        g = g.sort_values("date").copy()

        g["log_close"] = np.log(g["adj_close"])
        g["log_return_1d"] = g["log_close"].diff()
        g["abs_log_return_1d"] = g["log_return_1d"].abs()
        g["squared_log_return_1d"] = g["log_return_1d"] ** 2

        # OHLC-based volatility proxy: Parkinson estimator
        high_low_log = np.log(g["high"] / g["low"])
        g["parkinson_vol_1d"] = np.sqrt((high_low_log ** 2) / (4 * np.log(2)))

        # Historical price-only features
        for window in [3, 5, 10, 20]:
            g[f"hist_rv_return_{window}d"] = (
                g["squared_log_return_1d"]
                .rolling(window=window, min_periods=max(2, window // 2))
                .mean()
                .pipe(np.sqrt)
            )

            g[f"hist_abs_return_mean_{window}d"] = (
                g["abs_log_return_1d"]
                .rolling(window=window, min_periods=max(2, window // 2))
                .mean()
            )

            g[f"hist_parkinson_vol_{window}d"] = (
                g["parkinson_vol_1d"]
                .rolling(window=window, min_periods=max(2, window // 2))
                .mean()
            )

        # Future realized-volatility targets
        # Future return target uses the same strict future window.
        # Require a full future window. For example, 5-day target must have all next 5 trading days.
        for h in horizons:
            future_returns = pd.concat(
                [g["log_return_1d"].shift(-i) for i in range(1, h + 1)],
                axis=1
            )

            future_sq = future_returns ** 2

            future_ohlc_var = pd.concat(
                [(g["parkinson_vol_1d"].shift(-i)) ** 2 for i in range(1, h + 1)],
                axis=1
            )

            full_return_window = future_returns.notna().sum(axis=1) == h
            full_ohlc_window = future_ohlc_var.notna().sum(axis=1) == h

            g[f"target_return_{h}d"] = np.where(
                full_return_window,
                future_returns.sum(axis=1),
                np.nan
            )

            g[f"target_rv_return_{h}d"] = np.where(
                full_return_window,
                np.sqrt(future_sq.mean(axis=1)),
                np.nan
            )

            g[f"target_rv_ohlc_{h}d"] = np.where(
                full_ohlc_window,
                np.sqrt(future_ohlc_var.mean(axis=1)),
                np.nan
            )

        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        default="2016-03-01",
        help="start date, format YYYY-MM-DD"
    )
    parser.add_argument(
        "--end",
        default="2026-03-31",
        help="end date, format YYYY-MM-DD. yfinance treats this as exclusive."
    )
    parser.add_argument(
        "--output_file",
        default="/project/hrao/GDELT/asset_prices/daily_asset_prices.parquet",
        help="output parquet file"
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help=(
            "Optional comma-separated asset=ticker pairs. "
            "Example: QQQ=QQQ,Gold=GLD,WTI_Oil=USO"
        )
    )
    args = parser.parse_args()

    if args.tickers is None:
        asset_to_ticker = DEFAULT_TICKERS
    else:
        asset_to_ticker = {}
        for item in args.tickers.split(","):
            asset, ticker = item.split("=")
            asset_to_ticker[asset.strip()] = ticker.strip()

    all_parts = []

    print("Downloading asset prices...")
    print("Assets:", asset_to_ticker)

    for asset, ticker in asset_to_ticker.items():
        print(f"\n[START] {asset}: {ticker}")
        df = download_one_ticker(ticker, args.start, args.end)
        df["asset"] = asset
        all_parts.append(df)
        print(f"[DONE] {asset}: {df.shape}")

    prices = pd.concat(all_parts, ignore_index=True)
    prices = prices.sort_values(["asset", "date"]).reset_index(drop=True)

    prices = add_price_features(prices, horizons=(1, 3, 5))

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path, index=False)

    print("\nSaved:", out_path)
    print("Shape:", prices.shape)
    print("\nColumns:")
    print(prices.columns.tolist())
    print("\nHead:")
    print(prices.head(10).to_string(index=False))

    print("\nNon-null target counts:")
    target_cols = [
        c for c in prices.columns
        if c.startswith("target_return") or c.startswith("target_rv")
    ]
    target_cols = sorted(target_cols)
    print(prices[target_cols].notna().sum())


if __name__ == "__main__":
    main()
