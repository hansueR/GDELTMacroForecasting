from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.two_branch_distribution_model import (  # noqa: E402
    TwoBranchDistributionModel,
    gaussian_nll_loss,
)


HORIZONS = [1, 3, 5]


class SequenceDataset(Dataset):
    def __init__(self, x_price, x_event, y_return, y_risk, asset_id):
        self.x_price = torch.tensor(x_price, dtype=torch.float32)
        self.x_event = torch.tensor(x_event, dtype=torch.float32)
        self.y_return = torch.tensor(y_return, dtype=torch.float32)
        self.y_risk = torch.tensor(y_risk, dtype=torch.float32)
        self.asset_id = torch.tensor(asset_id, dtype=torch.long)

    def __len__(self):
        return self.x_price.shape[0]

    def __getitem__(self, idx):
        return {
            "x_price": self.x_price[idx],
            "x_event": self.x_event[idx],
            "y_return": self.y_return[idx],
            "y_risk": self.y_risk[idx],
            "asset_id": self.asset_id[idx],
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_standardizer(x_train):
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardizer(x, mean, std):
    return (x - mean) / std


def gaussian_nll_numpy(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    var = sigma ** 2
    return 0.5 * (np.log(2.0 * np.pi * var) + ((y_true - mu) ** 2) / var)


def directional_accuracy(y_true, mu):
    return float(np.mean((y_true > 0) == (mu > 0)))


def auc_from_mu(y_true, mu):
    y_bin = (y_true > 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return float(roc_auc_score(y_bin, mu))


def make_walk_forward_indices(meta, test_year, max_horizon):
    train_indices = []
    test_indices = []

    for _, g in meta.groupby("asset"):
        g = g.sort_values("date").copy()

        train_g = g[g["year"] < test_year]
        test_g = g[g["year"] == test_year]

        # Drop the last max_horizon training rows per asset so future labels do
        # not cross into the test year.
        if len(train_g) > max_horizon:
            train_g = train_g.iloc[:-max_horizon]
        else:
            train_g = train_g.iloc[0:0]

        train_indices.extend(train_g["idx"].tolist())
        test_indices.extend(test_g["idx"].tolist())

    return np.array(train_indices, dtype=np.int64), np.array(test_indices, dtype=np.int64)


def train_one_epoch(model, loader, optimizer, device, risk_loss_weight):
    model.train()
    losses = []

    for batch in loader:
        x_price = batch["x_price"].to(device)
        x_event = batch["x_event"].to(device)
        y_return = batch["y_return"].to(device)
        y_risk = batch["y_risk"].to(device)
        asset_id = batch["asset_id"].to(device)

        out = model(x_price=x_price, x_event=x_event, asset_id=asset_id)
        mu = out["mu"]
        sigma = out["sigma"]

        loss = gaussian_nll_loss(y_return, mu, sigma)

        if risk_loss_weight > 0:
            risk_loss = torch.mean(torch.abs(sigma - y_risk))
            loss = loss + risk_loss_weight * risk_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    mu_list = []
    sigma_list = []
    y_return_list = []
    y_risk_list = []
    asset_id_list = []

    for batch in loader:
        x_price = batch["x_price"].to(device)
        x_event = batch["x_event"].to(device)
        asset_id = batch["asset_id"].to(device)

        out = model(x_price=x_price, x_event=x_event, asset_id=asset_id)

        mu_list.append(out["mu"].detach().cpu().numpy())
        sigma_list.append(out["sigma"].detach().cpu().numpy())
        y_return_list.append(batch["y_return"].numpy())
        y_risk_list.append(batch["y_risk"].numpy())
        asset_id_list.append(batch["asset_id"].numpy())

    return {
        "mu": np.concatenate(mu_list, axis=0),
        "sigma": np.concatenate(sigma_list, axis=0),
        "y_return": np.concatenate(y_return_list, axis=0),
        "y_risk": np.concatenate(y_risk_list, axis=0),
        "asset_id": np.concatenate(asset_id_list, axis=0),
    }


def make_prediction_and_metric_frames(
    pred,
    meta_test,
    test_year,
    model_name,
):
    pred_rows = []
    metric_rows = []

    y_return = pred["y_return"]
    y_risk = pred["y_risk"]
    mu = pred["mu"]
    sigma = pred["sigma"]

    meta_test = meta_test.reset_index(drop=True)

    for h_idx, horizon in enumerate(HORIZONS):
        target_name = f"target_return_{horizon}d"
        risk_name = f"target_rv_return_{horizon}d"

        y_h = y_return[:, h_idx]
        mu_h = mu[:, h_idx]
        sigma_h = sigma[:, h_idx]
        risk_h = y_risk[:, h_idx]
        nll_h = gaussian_nll_numpy(y_h, mu_h, sigma_h)

        for i in range(len(meta_test)):
            pred_rows.append({
                "date": meta_test.loc[i, "date"],
                "asset": meta_test.loc[i, "asset"],
                "horizon": horizon,
                "target": target_name,
                "risk_target": risk_name,
                "model_type": model_name,
                "test_year": int(test_year),
                "y_true_return": float(y_h[i]),
                "mu_pred": float(mu_h[i]),
                "sigma_pred": float(sigma_h[i]),
                "y_true_risk": float(risk_h[i]),
                "nll": float(nll_h[i]),
            })

        tmp = pd.DataFrame({
            "asset": meta_test["asset"].values,
            "y_return": y_h,
            "mu": mu_h,
            "sigma": sigma_h,
            "risk": risk_h,
            "nll": nll_h,
        })

        for asset, g in tmp.groupby("asset"):
            metric_rows.append({
                "asset": asset,
                "horizon": horizon,
                "target": target_name,
                "risk_target": risk_name,
                "model_type": model_name,
                "test_year": int(test_year),
                "n_test": int(len(g)),
                "nll_mean": float(g["nll"].mean()),
                "nll_median": float(g["nll"].median()),
                "directional_acc": directional_accuracy(
                    g["y_return"].to_numpy(),
                    g["mu"].to_numpy(),
                ),
                "auc": auc_from_mu(
                    g["y_return"].to_numpy(),
                    g["mu"].to_numpy(),
                ),
                "mu_mae": float(np.mean(np.abs(g["y_return"] - g["mu"]))),
                "mu_rmse": float(np.sqrt(np.mean((g["y_return"] - g["mu"]) ** 2))),
                "risk_mae": float(np.mean(np.abs(g["risk"] - g["sigma"]))),
                "risk_rmse": float(np.sqrt(np.mean((g["risk"] - g["sigma"]) ** 2))),
                "sigma_mean": float(g["sigma"].mean()),
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def summarize_by_year(metrics):
    group_cols = ["model_type", "asset", "horizon", "target", "risk_target"]

    def wmean(g, col):
        values = pd.to_numeric(g[col], errors="coerce")
        weights = pd.to_numeric(g["n_test"], errors="coerce")
        mask = values.notna() & weights.notna() & (weights > 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(values[mask], weights=weights[mask]))

    rows = []
    for keys, g in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_test_total"] = int(g["n_test"].sum())
        row["n_splits"] = int(g["test_year"].nunique())

        for col in [
            "nll_mean",
            "nll_median",
            "directional_acc",
            "auc",
            "mu_mae",
            "mu_rmse",
            "risk_mae",
            "risk_rmse",
            "sigma_mean",
        ]:
            row[f"{col}_weighted"] = wmean(g, col)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["asset", "horizon"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_dir",
        default="/project/hrao/GDELT/modeling/sequence_dataset_l20",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/two_branch_distribution_v1",
    )
    parser.add_argument("--first_test_year", type=int, default=2019)
    parser.add_argument("--max_horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--asset_emb_dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--risk_loss_weight", type=float, default=0.0)
    parser.add_argument("--sigma_floor", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    sequence_dir = Path(args.sequence_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(sequence_dir / "sequence_data.npz", allow_pickle=True)
    meta = pd.read_csv(sequence_dir / "metadata.csv")
    meta["date"] = pd.to_datetime(meta["date"])
    meta["idx"] = np.arange(len(meta))

    spec = json.loads((sequence_dir / "feature_spec.json").read_text())

    x_price = data["x_price"].astype(np.float32)
    x_event = data["x_event"].astype(np.float32)
    y_return = data["y_return"].astype(np.float32)
    y_risk = data["y_risk"].astype(np.float32)
    asset_id = data["asset_id"].astype(np.int64)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("Device:", device)

    price_dim = x_price.shape[-1]
    event_dim = x_event.shape[-1]
    num_assets = len(spec["asset_to_id"])

    test_years = sorted(y for y in meta["year"].unique() if y >= args.first_test_year)

    all_predictions = []
    all_metrics = []

    model_name = "two_branch_lstm_event_attention_gating"

    for test_year in test_years:
        print(f"\n=== Test year {test_year} ===")

        train_idx, test_idx = make_walk_forward_indices(
            meta=meta,
            test_year=test_year,
            max_horizon=args.max_horizon,
        )

        if len(train_idx) < 500 or len(test_idx) < 30:
            print(f"Skip {test_year}: n_train={len(train_idx)}, n_test={len(test_idx)}")
            continue

        xp_train_raw = x_price[train_idx]
        xe_train_raw = x_event[train_idx]
        xp_test_raw = x_price[test_idx]
        xe_test_raw = x_event[test_idx]

        price_mean, price_std = fit_standardizer(xp_train_raw)
        event_mean, event_std = fit_standardizer(xe_train_raw)

        xp_train = apply_standardizer(xp_train_raw, price_mean, price_std)
        xe_train = apply_standardizer(xe_train_raw, event_mean, event_std)
        xp_test = apply_standardizer(xp_test_raw, price_mean, price_std)
        xe_test = apply_standardizer(xe_test_raw, event_mean, event_std)

        train_ds = SequenceDataset(
            x_price=xp_train,
            x_event=xe_train,
            y_return=y_return[train_idx],
            y_risk=y_risk[train_idx],
            asset_id=asset_id[train_idx],
        )

        test_ds = SequenceDataset(
            x_price=xp_test,
            x_event=xe_test,
            y_return=y_return[test_idx],
            y_risk=y_risk[test_idx],
            asset_id=asset_id[test_idx],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = TwoBranchDistributionModel(
            price_dim=price_dim,
            event_dim=event_dim,
            num_assets=num_assets,
            hidden_dim=args.hidden_dim,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            horizons=HORIZONS,
            asset_emb_dim=args.asset_emb_dim,
            sigma_floor=args.sigma_floor,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                risk_loss_weight=args.risk_loss_weight,
            )

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"year={test_year} epoch={epoch} train_loss={loss:.6f}")

        ckpt_path = out_dir / "checkpoints" / f"model_test_year_{test_year}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "feature_spec": spec,
                "price_mean": price_mean,
                "price_std": price_std,
                "event_mean": event_mean,
                "event_std": event_std,
            },
            ckpt_path,
        )

        pred = predict(model=model, loader=test_loader, device=device)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        pred_df, metric_df = make_prediction_and_metric_frames(
            pred=pred,
            meta_test=meta_test,
            test_year=test_year,
            model_name=model_name,
        )

        all_predictions.append(pred_df)
        all_metrics.append(metric_df)

    if not all_predictions:
        raise ValueError("No predictions were produced.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_by_year(metrics)

    predictions_file = out_dir / "two_branch_predictions.csv"
    by_year_file = out_dir / "two_branch_by_year.csv"
    summary_file = out_dir / "two_branch_summary.csv"

    predictions.to_csv(predictions_file, index=False)
    metrics.to_csv(by_year_file, index=False)
    summary.to_csv(summary_file, index=False)

    print("\nSaved:")
    print(predictions_file)
    print(by_year_file)
    print(summary_file)


if __name__ == "__main__":
    main()
