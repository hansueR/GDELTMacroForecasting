import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ASSET_ORDER = ["Gold", "QQQ", "WTI_Oil"]
ASSET_DISPLAY = {
    "Gold": "Gold",
    "QQQ": "QQQ",
    "WTI_Oil": "WTI Oil",
}

HORIZON_ORDER = [1, 3, 5]
HORIZON_DISPLAY = {
    1: "1d",
    3: "3d",
    5: "5d",
}

TOPK_ORDER = [10, 20, 50]
TOPK_X = np.arange(len(TOPK_ORDER))


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def save_figure(fig, out_base: Path):
    fig.savefig(out_base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    plt.close(fig)


def add_heatmap_text(ax, data_2d, vmax_abs):
    threshold = 0.55 * vmax_abs if vmax_abs > 0 else 0.0

    for i in range(data_2d.shape[0]):
        for j in range(data_2d.shape[1]):
            val = data_2d[i, j]

            if pd.isna(val):
                text = "NA"
                color = "black"
            else:
                text = f"{val:+.3f}"
                color = "white" if abs(val) >= threshold else "black"

            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)


def style_heatmap_axes(ax, show_y_labels=True):
    ax.set_xticks(range(len(HORIZON_ORDER)))
    ax.set_xticklabels([HORIZON_DISPLAY[h] for h in HORIZON_ORDER])

    ax.set_yticks(range(len(ASSET_ORDER)))
    if show_y_labels:
        ax.set_yticklabels([ASSET_DISPLAY[a] for a in ASSET_ORDER])
    else:
        ax.set_yticklabels([])

    # Cell boundaries
    ax.set_xticks(np.arange(-0.5, len(HORIZON_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ASSET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_event_window_ablation(event_window_csv: Path, output_dir: Path):
    df = pd.read_csv(event_window_csv)

    nll_cols = [
        ("gain_daily_vs_price", "Daily vs Price"),
        ("gain_all_vs_price", "Daily + Rolling vs Price"),
        ("gain_rolling_extra_vs_daily", "Rolling Extra vs Daily"),
    ]

    auc_cols = [
        ("auc_gain_daily_vs_price", "Daily vs Price"),
        ("auc_gain_all_vs_price", "Daily + Rolling vs Price"),
        ("auc_gain_rolling_extra_vs_daily", "Rolling Extra vs Daily"),
    ]

    nll_abs_max = max(float(df[col].abs().max()) for col, _ in nll_cols)
    auc_abs_max = max(float(df[col].abs().max()) for col, _ in auc_cols)

    nll_norm = TwoSlopeNorm(vmin=-nll_abs_max, vcenter=0.0, vmax=nll_abs_max)
    auc_norm = TwoSlopeNorm(vmin=-auc_abs_max, vcenter=0.0, vmax=auc_abs_max)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 6.0),
        constrained_layout=False,
    )

    nll_im = None
    auc_im = None

    for col_idx, (metric_col, panel_title) in enumerate(nll_cols):
        ax = axes[0, col_idx]

        pivot = (
            df.pivot(index="asset", columns="horizon", values=metric_col)
            .reindex(index=ASSET_ORDER, columns=HORIZON_ORDER)
        )

        data = pivot.to_numpy(dtype=float)
        nll_im = ax.imshow(data, aspect="auto", norm=nll_norm, cmap="RdBu_r")

        ax.set_title(panel_title, pad=7)
        style_heatmap_axes(ax, show_y_labels=(col_idx == 0))
        add_heatmap_text(ax, data, nll_abs_max)

    for col_idx, (metric_col, panel_title) in enumerate(auc_cols):
        ax = axes[1, col_idx]

        pivot = (
            df.pivot(index="asset", columns="horizon", values=metric_col)
            .reindex(index=ASSET_ORDER, columns=HORIZON_ORDER)
        )

        data = pivot.to_numpy(dtype=float)
        auc_im = ax.imshow(data, aspect="auto", norm=auc_norm, cmap="RdBu_r")

        ax.set_title(panel_title, pad=7)
        style_heatmap_axes(ax, show_y_labels=(col_idx == 0))
        ax.set_xlabel("Forecast horizon")
        add_heatmap_text(ax, data, auc_abs_max)

    # Row labels
    fig.text(
        0.015,
        0.70,
        "NLL gain\npositive is better",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.015,
        0.30,
        "AUC gain\npositive is better",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )

    # Leave explicit space on the right for colorbars.
    # Using dedicated colorbar axes avoids the colorbar being placed too close
    # to the third heatmap panel.
    fig.subplots_adjust(
        left=0.085,
        right=0.890,
        top=0.93,
        bottom=0.10,
        wspace=0.26,
        hspace=0.30,
    )

    # Dedicated colorbar axes: [left, bottom, width, height]
    cbar1_ax = fig.add_axes([0.915, 0.575, 0.012, 0.325])
    cbar2_ax = fig.add_axes([0.915, 0.145, 0.012, 0.325])

    cbar1 = fig.colorbar(
        nll_im,
        cax=cbar1_ax,
    )
    cbar1.set_label("NLL gain")
    cbar1.ax.tick_params(labelsize=9)

    cbar2 = fig.colorbar(
        auc_im,
        cax=cbar2_ax,
    )
    cbar2.set_label("AUC gain")
    cbar2.ax.tick_params(labelsize=9)

    save_figure(fig, output_dir / "figure1_event_window_ablation")


def plot_topk_sensitivity(topk_csv: Path, output_dir: Path):
    df = pd.read_csv(topk_csv)
    df = df.sort_values(["asset", "horizon", "top_k"]).copy()

    baseline = (
        df[df["top_k"] == 10][["asset", "horizon", "nll_mean"]]
        .rename(columns={"nll_mean": "nll_at_10"})
    )

    df = df.merge(baseline, on=["asset", "horizon"], how="left")
    df["delta_nll_vs_10"] = df["nll_mean"] - df["nll_at_10"]

    idx = df.groupby(["asset", "horizon"])["nll_mean"].idxmin()
    best_counts = df.loc[idx, "top_k"].value_counts().sort_index()

    print("Best NLL count across asset-horizon settings:")
    for k in TOPK_ORDER:
        print(f"  top-k={k}: {int(best_counts.get(k, 0))}/9")

    # Common y-limits for cleaner comparison.
    max_delta = float(df["delta_nll_vs_10"].max())
    delta_ylim = (-3.0, max_delta * 1.08)

    auc_min = float(df["auc_mean"].min())
    auc_max = float(df["auc_mean"].max())
    auc_margin = max(0.01, 0.08 * (auc_max - auc_min))
    auc_ylim = (auc_min - auc_margin, auc_max + auc_margin)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.2),
        sharex=True,
        constrained_layout=False,
    )

    for col_idx, asset in enumerate(ASSET_ORDER):
        asset_df = df[df["asset"] == asset].copy()

        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        for horizon in HORIZON_ORDER:
            sub = asset_df[asset_df["horizon"] == horizon].sort_values("top_k")

            x = [TOPK_ORDER.index(int(k)) for k in sub["top_k"].values]

            ax_top.plot(
                x,
                sub["delta_nll_vs_10"].values,
                marker="o",
                linewidth=1.7,
                label=HORIZON_DISPLAY[horizon],
            )

            ax_bottom.plot(
                x,
                sub["auc_mean"].values,
                marker="o",
                linewidth=1.7,
                label=HORIZON_DISPLAY[horizon],
            )

        ax_top.axhline(0.0, linewidth=1.0, linestyle="--")
        ax_top.set_title(ASSET_DISPLAY[asset], pad=7)
        ax_top.set_ylim(delta_ylim)

        ax_bottom.set_ylim(auc_ylim)
        ax_bottom.set_xlabel("Top-k event features")

        if col_idx == 0:
            ax_top.set_ylabel(r"$\Delta$NLL vs top-$k=10$")
            ax_bottom.set_ylabel("AUC")

        ax_top.set_xticks(TOPK_X)
        ax_bottom.set_xticks(TOPK_X)
        ax_bottom.set_xticklabels([str(k) for k in TOPK_ORDER])

        ax_top.grid(axis="y", linewidth=0.5, alpha=0.35)
        ax_bottom.grid(axis="y", linewidth=0.5, alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.875,
        bottom=0.105,
        wspace=0.16,
        hspace=0.23,
    )

    save_figure(fig, output_dir / "figure2_topk_sensitivity")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/project/hrao/GDELT",
        help="Project root directory.",
    )

    parser.add_argument(
        "--event_window_csv",
        default="",
        help="Path to event-window gain CSV. If empty, use default project paths.",
    )

    parser.add_argument(
        "--topk_csv",
        default="",
        help="Path to top-k ablation CSV. If empty, use default project paths.",
    )

    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory. If empty, use ROOT/results/report_figures.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    if args.event_window_csv:
        event_window_csv = Path(args.event_window_csv)
    else:
        event_window_csv = first_existing([
            root / "results/report_summary_tables/report_ablation_event_window_gain.csv",
            root / "results/ablation_tables/table_event_window_gain.csv",
        ])

    if args.topk_csv:
        topk_csv = Path(args.topk_csv)
    else:
        topk_csv = first_existing([
            root / "results/report_summary_tables/report_ablation_topk.csv",
            root / "results/ablation_tables/table_topk_ablation.csv",
        ])

    if event_window_csv is None:
        raise FileNotFoundError(
            "Cannot find event-window gain CSV. Expected one of:\n"
            f"  {root / 'results/report_summary_tables/report_ablation_event_window_gain.csv'}\n"
            f"  {root / 'results/ablation_tables/table_event_window_gain.csv'}"
        )

    if topk_csv is None:
        raise FileNotFoundError(
            "Cannot find top-k ablation CSV. Expected one of:\n"
            f"  {root / 'results/report_summary_tables/report_ablation_topk.csv'}\n"
            f"  {root / 'results/ablation_tables/table_topk_ablation.csv'}"
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = root / "results/report_figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using input files:")
    print(f"  event_window_csv = {event_window_csv}")
    print(f"  topk_csv         = {topk_csv}")
    print(f"  output_dir       = {output_dir}")

    plot_event_window_ablation(event_window_csv, output_dir)
    plot_topk_sensitivity(topk_csv, output_dir)

    print("\nSaved figures:")
    print(output_dir / "figure1_event_window_ablation.png")
    print(output_dir / "figure1_event_window_ablation.pdf")
    print(output_dir / "figure2_topk_sensitivity.png")
    print(output_dir / "figure2_topk_sensitivity.pdf")


if __name__ == "__main__":
    main()
