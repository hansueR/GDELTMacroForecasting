from pathlib import Path
import argparse
import re
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go


LOOKUP_URL = "https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt"

BASE_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]

USE_COLS = [
    "SQLDATE",
    "ActionGeo_CountryCode",
    "ActionGeo_FullName",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "NumMentions",
    "GoldsteinScale",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_root",
        default="/project/hrao/GDELT/clean",
    )
    parser.add_argument(
        "--lookup_file",
        default="/project/hrao/GDELT/lookups/CAMEO.eventcodes.txt",
    )
    parser.add_argument(
        "--date",
        default="2026-02-28",
        help="Target date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/report_figures",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of countries to show after aggregation. Use 0 to keep all.",
    )
    parser.add_argument(
        "--country_mode",
        default="dominant",
        choices=["dominant", "all_pairs"],
        help=(
            "dominant: one point per country using its strongest EventRootCode. "
            "all_pairs: one point per country-root pair."
        ),
    )
    parser.add_argument(
        "--intensity_mode",
        default="mentions",
        choices=["mentions", "mentions_x_abs_goldstein", "n_events"],
    )
    parser.add_argument(
        "--auto_download_lookup",
        action="store_true",
        help="Download official GDELT CAMEO lookup file if lookup_file is missing.",
    )
    parser.add_argument(
        "--text_mode",
        default="country",
        choices=["country", "country_plus_root", "root_only"],
        help=(
            "country: show only country code on the map; "
            "country_plus_root: show country code + root code; "
            "root_only: show only EventRootCode."
        ),
    )
    parser.add_argument(
        "--label_font_size",
        type=int,
        default=18,
        help="Font size of the country/event text labels on the map.",
    )
    return parser.parse_args()


def normalize_code(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def ensure_lookup_file(path: Path, auto_download: bool):
    if path.exists():
        return

    if not auto_download:
        raise FileNotFoundError(
            f"Missing lookup file: {path}\n"
            f"Download it with:\n"
            f"mkdir -p {path.parent}\n"
            f"wget -O {path} {LOOKUP_URL}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official CAMEO lookup file from {LOOKUP_URL}")
    urllib.request.urlretrieve(LOOKUP_URL, path)


def load_cameo_event_lookup(path: Path) -> dict:
    text = path.read_text(encoding="latin-1", errors="ignore")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    mapping = {}

    if len(lines) > 1:
        for ln in lines:
            if ln.upper().startswith("CAMEOEVENTCODE"):
                continue

            parts = re.split(r"\s+", ln, maxsplit=1)
            if len(parts) == 2 and re.fullmatch(r"\d{2,4}", parts[0]):
                mapping[parts[0]] = parts[1].strip()

        if mapping:
            return mapping

    text = re.sub(r"^\s*CAMEOEVENTCODE\s+EVENTDESCRIPTION\s+", "", text.strip())
    text = re.sub(r"([A-Za-z\)])(\d{2,4}):\[-?\d+(?:\.\d+)?\]", r"\1 \2", text)

    pattern = re.compile(r"(?<![A-Za-z0-9])(\d{2,4})(?:\:\[-?\d+(?:\.\d+)?\])?\s+")
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        code = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        desc = text[start:end].strip()
        if desc:
            mapping[code] = desc

    if not mapping:
        raise ValueError(f"Failed to parse lookup file: {path}")

    return mapping


def get_month_dir(clean_root: Path, date_str: str) -> Path:
    month = pd.Timestamp(date_str).strftime("%Y%m")
    return clean_root / f"events_{month}"


def get_sql_date(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%Y%m%d")


def safe_mode(series: pd.Series):
    s = series.dropna().astype(str)
    s = s[s.str.len() > 0]
    if s.empty:
        return ""
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else s.iloc[0]


def load_one_day(clean_root: Path, date_str: str) -> pd.DataFrame:
    month_dir = get_month_dir(clean_root, date_str)

    if not month_dir.exists():
        raise FileNotFoundError(f"Month directory not found: {month_dir}")

    files = sorted(month_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {month_dir}")

    target_sql = get_sql_date(date_str)
    parts = []

    for fp in files:
        df = pd.read_parquet(fp, columns=USE_COLS)
        df["SQLDATE"] = df["SQLDATE"].astype(str)
        day_df = df[df["SQLDATE"] == target_sql].copy()
        if not day_df.empty:
            parts.append(day_df)

    if not parts:
        raise ValueError(f"No events found for date={date_str} under {month_dir}")

    return pd.concat(parts, ignore_index=True)


def compute_intensity(df: pd.DataFrame, mode: str) -> pd.Series:
    n_events = pd.Series(np.ones(len(df)), index=df.index, dtype=float)
    mentions = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0.0).clip(lower=0.0)
    goldstein = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0.0)

    if mode == "n_events":
        return n_events
    if mode == "mentions":
        return mentions
    if mode == "mentions_x_abs_goldstein":
        return mentions * (1.0 + goldstein.abs())

    raise ValueError(f"Unknown intensity mode: {mode}")


def code_label(code: str, lookup: dict) -> str:
    code = normalize_code(code)
    desc = lookup.get(code, "")
    if desc:
        return f"{code}: {desc}"
    return f"{code}: [missing in lookup]"


def build_country_event_table(
    df: pd.DataFrame,
    lookup: dict,
    intensity_mode: str,
    country_mode: str,
    top_n: int,
    text_mode: str,
):
    df = df.copy()

    for col in [
        "ActionGeo_CountryCode",
        "EventRootCode",
        "EventBaseCode",
        "EventCode",
        "QuadClass",
    ]:
        df[col] = df[col].apply(normalize_code)

    df["ActionGeo_Lat"] = pd.to_numeric(df["ActionGeo_Lat"], errors="coerce")
    df["ActionGeo_Long"] = pd.to_numeric(df["ActionGeo_Long"], errors="coerce")
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce")
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")

    df = df.dropna(
        subset=[
            "ActionGeo_CountryCode",
            "EventRootCode",
            "ActionGeo_Lat",
            "ActionGeo_Long",
        ]
    ).copy()

    df = df[
        (df["ActionGeo_CountryCode"].astype(str).str.len() > 0)
        & (df["EventRootCode"].astype(str).str.len() > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid rows after filtering missing country/event/geo fields.")

    df["row_intensity"] = compute_intensity(df, intensity_mode)

    country_root = (
        df.groupby(["ActionGeo_CountryCode", "EventRootCode"], dropna=False)
        .agg(
            intensity=("row_intensity", "sum"),
            n_events=("EventRootCode", "size"),
            mentions_sum=("NumMentions", "sum"),
            goldstein_mean=("GoldsteinScale", "mean"),
            lat=("ActionGeo_Lat", "mean"),
            lon=("ActionGeo_Long", "mean"),
            dominant_base_code=("EventBaseCode", safe_mode),
            dominant_event_code=("EventCode", safe_mode),
            quad_class=("QuadClass", safe_mode),
            example_location=("ActionGeo_FullName", safe_mode),
        )
        .reset_index()
    )

    country_root = country_root.sort_values(
        ["ActionGeo_CountryCode", "intensity", "n_events"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    if country_mode == "dominant":
        plot_df = country_root.groupby("ActionGeo_CountryCode", as_index=False).first()
        plot_df = plot_df.sort_values("intensity", ascending=False).reset_index(drop=True)
    else:
        plot_df = country_root.sort_values("intensity", ascending=False).reset_index(drop=True)

    if top_n > 0:
        plot_df = plot_df.head(top_n).copy()

    plot_df["root_label"] = plot_df["EventRootCode"].apply(lambda x: code_label(x, lookup))
    plot_df["base_label"] = plot_df["dominant_base_code"].apply(lambda x: code_label(x, lookup))
    plot_df["event_label"] = plot_df["dominant_event_code"].apply(lambda x: code_label(x, lookup))
    plot_df["country_label"] = plot_df["ActionGeo_CountryCode"].astype(str)

    if text_mode == "country":
        plot_df["map_text"] = plot_df["country_label"]
    elif text_mode == "country_plus_root":
        plot_df["map_text"] = (
            plot_df["country_label"] + "<br>" + plot_df["EventRootCode"].astype(str)
        )
    elif text_mode == "root_only":
        plot_df["map_text"] = plot_df["EventRootCode"].astype(str)
    else:
        plot_df["map_text"] = plot_df["country_label"]

    return plot_df


def sort_codes(codes):
    def key(x):
        s = str(x)
        try:
            return int(s)
        except Exception:
            return 9999
    return sorted([str(c) for c in codes], key=key)


def rgba(rgb, alpha):
    r, g, b = rgb
    return f"rgba({r},{g},{b},{alpha:.3f})"


def add_visual_fields(df: pd.DataFrame):
    df = df.copy()

    root_codes = sort_codes(df["EventRootCode"].unique())
    color_map = {
        code: BASE_COLORS[i % len(BASE_COLORS)]
        for i, code in enumerate(root_codes)
    }

    min_val = float(df["intensity"].min())
    max_val = float(df["intensity"].max())

    if max_val > min_val:
        norm = (df["intensity"] - min_val) / (max_val - min_val)
    else:
        norm = pd.Series(np.ones(len(df)), index=df.index)

    df["alpha"] = 0.30 + 0.70 * norm
    df["marker_size"] = 10 + 30 * norm
    df["marker_color"] = [
        rgba(color_map[str(root)], alpha)
        for root, alpha in zip(df["EventRootCode"], df["alpha"])
    ]

    return df


def make_map(
    plot_df: pd.DataFrame,
    date_str: str,
    output_dir: Path,
    country_mode: str,
    label_font_size: int,
):
    fig = go.Figure()

    root_codes = sort_codes(plot_df["EventRootCode"].unique())

    for root in root_codes:
        sub = plot_df[plot_df["EventRootCode"].astype(str) == root].copy()
        root_label = sub["root_label"].iloc[0]

        customdata = np.stack(
            [
                sub["ActionGeo_CountryCode"].astype(str),
                sub["root_label"].astype(str),
                sub["base_label"].astype(str),
                sub["event_label"].astype(str),
                sub["quad_class"].astype(str),
                sub["intensity"].astype(float),
                sub["n_events"].astype(int),
                sub["mentions_sum"].fillna(0.0).astype(float),
                sub["goldstein_mean"].fillna(0.0).astype(float),
                sub["example_location"].astype(str),
            ],
            axis=-1,
        )

        fig.add_trace(
            go.Scattergeo(
                lon=sub["lon"],
                lat=sub["lat"],
                mode="markers+text",
                text=sub["map_text"],
                textposition="top center",
                textfont=dict(size=label_font_size),
                marker=dict(
                    size=sub["marker_size"],
                    color=sub["marker_color"],
                    line=dict(width=0.9, color="rgba(50,50,50,0.9)"),
                ),
                name=root_label,
                showlegend=True,
                customdata=customdata,
                hovertemplate=(
                    "Country: %{customdata[0]}<br>"
                    "Root: %{customdata[1]}<br>"
                    "Base: %{customdata[2]}<br>"
                    "Event: %{customdata[3]}<br>"
                    "QuadClass: %{customdata[4]}<br>"
                    "Intensity: %{customdata[5]:.2f}<br>"
                    "Event rows: %{customdata[6]}<br>"
                    "Mentions sum: %{customdata[7]:.2f}<br>"
                    "Goldstein mean: %{customdata[8]:.3f}<br>"
                    "Example location: %{customdata[9]}<extra></extra>"
                ),
            )
        )

    # More zoomed-in world view: remove the full oval frame and crop empty ocean space.
    fig.update_geos(
        projection_type="equirectangular",
        showframe=False,
        showcountries=True,
        countrycolor="rgb(180,180,180)",
        countrywidth=0.8,
        showcoastlines=False,
        showland=True,
        landcolor="rgb(244,244,244)",
        showocean=True,
        oceancolor="rgb(252,252,252)",
        showlakes=False,
        showrivers=False,
        bgcolor="white",
        lonaxis_range=[-145, 165],
        lataxis_range=[-42, 78],
    )

    fig.update_layout(
        title=dict(
            text=f"GDELT Event Map on {date_str}",
            x=0.05,
            xanchor="left",
        ),
        width=1500,
        height=900,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            title="EventRootCode label",
            x=1.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,200,200,0.8)",
            borderwidth=1,
            font=dict(size=14),
            title_font=dict(size=16),
        ),
        font=dict(size=16),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    html_path = output_dir / f"event_map_{date_str}_with_lookup_zoomed.html"
    png_path = output_dir / f"event_map_{date_str}_with_lookup_zoomed.png"

    fig.write_html(html_path)

    try:
        fig.write_image(png_path, scale=2)
        print(png_path)
    except Exception as e:
        print("PNG export failed. HTML was still saved.")
        print("Reason:", repr(e))

    print(html_path)


def main():
    args = parse_args()

    lookup_path = Path(args.lookup_file)
    ensure_lookup_file(lookup_path, args.auto_download_lookup)
    lookup = load_cameo_event_lookup(lookup_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    day_df = load_one_day(Path(args.clean_root), args.date)

    plot_df = build_country_event_table(
        df=day_df,
        lookup=lookup,
        intensity_mode=args.intensity_mode,
        country_mode=args.country_mode,
        top_n=args.top_n,
        text_mode=args.text_mode,
    )

    plot_df = add_visual_fields(plot_df)

    summary_path = output_dir / f"event_map_{args.date}_with_lookup_zoomed_summary.csv"
    plot_df.to_csv(summary_path, index=False)

    make_map(
        plot_df=plot_df,
        date_str=args.date,
        output_dir=output_dir,
        country_mode=args.country_mode,
        label_font_size=args.label_font_size,
    )

    print("Summary CSV:")
    print(summary_path)

    print("\nTop rows:")
    print(
        plot_df[
            [
                "ActionGeo_CountryCode",
                "EventRootCode",
                "root_label",
                "dominant_base_code",
                "base_label",
                "dominant_event_code",
                "event_label",
                "quad_class",
                "intensity",
                "n_events",
                "mentions_sum",
                "goldstein_mean",
                "lat",
                "lon",
            ]
        ].head(30).to_string(index=False)
    )


if __name__ == "__main__":
    main()