from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


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


def get_event_features(df):
    event_cols = [c for c in df.columns if c.startswith("event_")]

    bad_roll_mean_cols = [
        c for c in event_cols
        if (
            ("tone_weighted_mean_roll" in c)
            or ("goldstein_weighted_mean_roll" in c)
        )
    ]

    event_cols = [c for c in event_cols if c not in bad_roll_mean_cols]
    return event_cols


def log_transform_event_features(df, event_features):
    df = df.copy()
    for col in event_features:
        if df[col].min(skipna=True) >= 0:
            df[col] = np.log1p(df[col])
    return df


def build_sequences(df, lookback, price_features, event_features, return_targets, risk_targets):
    rows = []
    x_price_list = []
    x_event_list = []
    y_return_list = []
    y_risk_list = []

    assets = sorted(df["asset"].unique())
    asset_to_id = {a: i for i, a in enumerate(assets)}

    required = (
        ["date", "asset"]
        + price_features
        + event_features
        + return_targets
        + risk_targets
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in model_table: {missing}")

    df = df[required].replace([np.inf, -np.inf], np.nan).copy()
    df["date"] = pd.to_datetime(df["date"])

    for asset in assets:
        g = df[df["asset"] == asset].copy()
        g = g.sort_values("date").reset_index(drop=True)

        needed = price_features + event_features + return_targets + risk_targets
        g = g.dropna(subset=needed).reset_index(drop=True)

        if len(g) < lookback + 10:
            print(f"WARNING: skip {asset}; too few rows after dropna: {len(g)}")
            continue

        price_arr = g[price_features].to_numpy(dtype=np.float32)
        event_arr = g[event_features].to_numpy(dtype=np.float32)
        y_return_arr = g[return_targets].to_numpy(dtype=np.float32)
        y_risk_arr = g[risk_targets].to_numpy(dtype=np.float32)

        for end_idx in range(lookback - 1, len(g)):
            start_idx = end_idx - lookback + 1

            x_price_list.append(price_arr[start_idx:end_idx + 1])
            x_event_list.append(event_arr[start_idx:end_idx + 1])
            y_return_list.append(y_return_arr[end_idx])
            y_risk_list.append(y_risk_arr[end_idx])

            rows.append({
                "date": g.loc[end_idx, "date"].strftime("%Y-%m-%d"),
                "asset": asset,
                "asset_id": asset_to_id[asset],
            })

    if not rows:
        raise ValueError("No sequence samples were created.")

    x_price = np.stack(x_price_list).astype(np.float32)
    x_event = np.stack(x_event_list).astype(np.float32)
    y_return = np.stack(y_return_list).astype(np.float32)
    y_risk = np.stack(y_risk_list).astype(np.float32)

    meta = pd.DataFrame(rows)
    meta["year"] = pd.to_datetime(meta["date"]).dt.year

    return x_price, x_event, y_return, y_risk, meta, asset_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_table",
        default="/project/hrao/GDELT/modeling/model_table.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/modeling/sequence_dataset_l20",
    )
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.model_table)
    df["date"] = pd.to_datetime(df["date"])

    event_features = get_event_features(df)
    if not event_features:
        raise ValueError("No event features found. Expected columns starting with event_.")

    df = log_transform_event_features(df, event_features)

    return_targets = [
        "target_return_1d",
        "target_return_3d",
        "target_return_5d",
    ]

    risk_targets = [
        "target_rv_return_1d",
        "target_rv_return_3d",
        "target_rv_return_5d",
    ]

    x_price, x_event, y_return, y_risk, meta, asset_to_id = build_sequences(
        df=df,
        lookback=args.lookback,
        price_features=PRICE_FEATURES,
        event_features=event_features,
        return_targets=return_targets,
        risk_targets=risk_targets,
    )

    npz_path = out_dir / "sequence_data.npz"
    meta_path = out_dir / "metadata.csv"
    spec_path = out_dir / "feature_spec.json"

    np.savez_compressed(
        npz_path,
        x_price=x_price,
        x_event=x_event,
        y_return=y_return,
        y_risk=y_risk,
        asset_id=meta["asset_id"].to_numpy(dtype=np.int64),
        date=meta["date"].to_numpy(),
        asset=meta["asset"].to_numpy(),
        year=meta["year"].to_numpy(dtype=np.int64),
    )

    meta.to_csv(meta_path, index=False)

    spec = {
        "lookback": args.lookback,
        "price_features": PRICE_FEATURES,
        "event_features": event_features,
        "return_targets": return_targets,
        "risk_targets": risk_targets,
        "asset_to_id": asset_to_id,
    }

    spec_path.write_text(json.dumps(spec, indent=2))

    print("Saved:")
    print(npz_path)
    print(meta_path)
    print(spec_path)
    print("Shapes:")
    print("x_price:", x_price.shape)
    print("x_event:", x_event.shape)
    print("y_return:", y_return.shape)
    print("y_risk:", y_risk.shape)
    print("assets:", asset_to_id)


if __name__ == "__main__":
    main()
