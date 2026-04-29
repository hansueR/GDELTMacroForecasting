import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


BASELINE_ORDER = ["NV", "EW", "RP", "RE", "RPE", "TB"]
BASELINE_ONLY_ORDER = ["NV", "EW", "RP", "RE", "RPE"]

BASELINE_NAME = {
    "NV": "NV",
    "EW": "EW",
    "RP": "RP",
    "RE": "RE",
    "RPE": "RPE",
    "TB": "TB",
}

ASSET_ORDER = ["Gold", "QQQ", "WTI Oil"]
HORIZON_ORDER = [1, 3, 5]

# Softer default palette.
BASELINE_COLORS = {
    "NV":  "#C7D1DD",
    "EW":  "#DDD0BC",
    "RP":  "#C8D9C8",
    "RE":  "#E3C6C6",
    "RPE": "#D4CCE4",
    "TB":  "#E18370",
}

# Slightly stronger shades used only for highlighting.
HIGHLIGHT_COLORS = {
    "NV":  "#94A9C5",
    "EW":  "#C8AE86",
    "RP":  "#97B997",
    "RE":  "#D89C9C",
    "RPE": "#AC98CC",
    "TB":  "#D97663",
}


def build_table_i_dataframe():
    """
    Hard-coded values from Experiment 1 / Table I.
    Each row is one asset-horizon-baseline result.
    """
    rows = []

    data = {
        ("Gold", 1): {
            "NV": (-2.707, 0.500),
            "EW": (-3.043, 0.488),
            "RP": (-2.717, 0.487),
            "RE": (-2.658, 0.500),
            "RPE": (-2.663, 0.494),
            "TB": (-3.150, 0.503),
        },
        ("Gold", 3): {
            "NV": (-2.167, 0.500),
            "EW": (-2.409, 0.459),
            "RP": (-2.148, 0.523),
            "RE": (-2.056, 0.520),
            "RPE": (-2.026, 0.544),
            "TB": (-2.534, 0.512),
        },
        ("Gold", 5): {
            "NV": (-1.930, 0.500),
            "EW": (-2.104, 0.440),
            "RP": (-1.894, 0.498),
            "RE": (-1.767, 0.526),
            "RPE": (-1.722, 0.529),
            "TB": (-2.316, 0.493),
        },
        ("QQQ", 1): {
            "NV": (-2.694, 0.500),
            "EW": (-2.907, 0.470),
            "RP": (-2.653, 0.483),
            "RE": (-2.659, 0.485),
            "RPE": (-2.604, 0.483),
            "TB": (-2.878, 0.527),
        },
        ("QQQ", 3): {
            "NV": (-2.237, 0.500),
            "EW": (-2.350, 0.466),
            "RP": (-2.113, 0.485),
            "RE": (-2.140, 0.467),
            "RPE": (-2.007, 0.459),
            "TB": (-2.326, 0.538),
        },
        ("QQQ", 5): {
            "NV": (-2.023, 0.500),
            "EW": (-2.097, 0.454),
            "RP": (-1.910, 0.480),
            "RE": (-1.858, 0.467),
            "RPE": (-1.716, 0.449),
            "TB": (-2.109, 0.526),
        },
        ("WTI Oil", 1): {
            "NV": (-2.054, 0.500),
            "EW": (-2.316, 0.476),
            "RP": (-1.994, 0.489),
            "RE": (-2.011, 0.508),
            "RPE": (-1.928, 0.491),
            "TB": (-2.311, 0.500),
        },
        ("WTI Oil", 3): {
            "NV": (-1.397, 0.500),
            "EW": (-1.686, 0.450),
            "RP": (-1.277, 0.515),
            "RE": (-1.277, 0.509),
            "RPE": (-1.114, 0.502),
            "TB": (-1.686, 0.514),
        },
        ("WTI Oil", 5): {
            "NV": (-1.065, 0.500),
            "EW": (-1.384, 0.427),
            "RP": (-0.986, 0.520),
            "RE": (-0.873, 0.551),
            "RPE": (-0.752, 0.535),
            "TB": (-1.383, 0.489),
        },
    }

    for asset in ASSET_ORDER:
        for horizon in HORIZON_ORDER:
            for baseline in BASELINE_ORDER:
                nll, auc = data[(asset, horizon)][baseline]
                rows.append(
                    {
                        "asset": asset,
                        "horizon": horizon,
                        "baseline": baseline,
                        "nll": nll,
                        "auc": auc,
                        "group": f"{asset}\n{horizon}d",
                    }
                )

    return pd.DataFrame(rows)


def get_best_baseline_by_group(df, metric):
    """
    For each asset-horizon group, find the best-performing baseline
    among NV / EW / RP / RE / RPE.
    For NLL: lower is better.
    For AUC: higher is better.
    """
    best_map = {}

    for asset in ASSET_ORDER:
        for horizon in HORIZON_ORDER:
            group = f"{asset}\n{horizon}d"
            sub = df[
                (df["asset"] == asset) &
                (df["horizon"] == horizon) &
                (df["baseline"].isin(BASELINE_ONLY_ORDER))
            ].copy()

            if metric == "nll":
                best_idx = sub[metric].idxmin()
            else:
                best_idx = sub[metric].idxmax()

            best_map[group] = sub.loc[best_idx, "baseline"]

    return best_map


def plot_grouped_bars(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for asset in ASSET_ORDER:
        for horizon in HORIZON_ORDER:
            groups.append(f"{asset}\n{horizon}d")

    x = np.arange(len(groups))
    n_models = len(BASELINE_ORDER)
    bar_width = 0.105

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), sharex=True)

    best_baseline_map = {
        "nll": get_best_baseline_by_group(df, "nll"),
        "auc": get_best_baseline_by_group(df, "auc"),
    }

    metrics = [
        ("nll", "NLL (lower is better)"),
        ("auc", "AUC (higher is better)"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for i, baseline in enumerate(BASELINE_ORDER):
            values = []
            bar_colors = []
            for group in groups:
                value = df[
                    (df["group"] == group) &
                    (df["baseline"] == baseline)
                ][metric].iloc[0]
                values.append(value)

                if baseline == "TB":
                    # Keep TB visible, but softer than before.
                    bar_colors.append(to_rgba(HIGHLIGHT_COLORS["TB"], alpha=0.82))
                elif best_baseline_map[metric][group] == baseline:
                    # Highlight the best baseline in this setting.
                    bar_colors.append(to_rgba(HIGHLIGHT_COLORS[baseline], alpha=0.88))
                else:
                    # All other baselines stay muted.
                    bar_colors.append(to_rgba(BASELINE_COLORS[baseline], alpha=0.60))

            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=BASELINE_NAME[baseline],
                color=bar_colors,
                edgecolor="none",
                linewidth=0.0,
                zorder=3 if baseline == "TB" else 2,
            )

        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.20, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric == "nll":
            ax.set_ylabel("NLL")
            ax.axhline(0.0, linewidth=0.8, color="#666666")
            ax.set_ylim(-3.3, 0.05)
        else:
            ax.set_ylabel("AUC")
            ax.axhline(0.5, linewidth=0.9, linestyle="--", color="#4C84C2")
            ax.set_ylim(0.42, 0.56)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(BASELINE_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(
        "Experiment 1: Overall Model Comparison",
        fontsize=15,
        y=1.08,
    )

    fig.text(
        0.5,
        0.01,
        "Proposed model and the best baseline in each setting are shown with stronger colors.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    png_path = output_dir / "experiment1_overall_bars.png"
    pdf_path = output_dir / "experiment1_overall_bars.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/report_figures",
        help="Directory for saving the output figure.",
    )
    args = parser.parse_args()

    df = build_table_i_dataframe()
    plot_grouped_bars(df, args.output_dir)


if __name__ == "__main__":
    main()
