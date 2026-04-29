from pathlib import Path
import argparse
import pandas as pd

ALL_COLS = [
    "GLOBALEVENTID","SQLDATE","MonthYear","Year","FractionDate",
    "Actor1Code","Actor1Name","Actor1CountryCode","Actor1KnownGroupCode","Actor1EthnicCode",
    "Actor1Religion1Code","Actor1Religion2Code","Actor1Type1Code","Actor1Type2Code","Actor1Type3Code",
    "Actor2Code","Actor2Name","Actor2CountryCode","Actor2KnownGroupCode","Actor2EthnicCode",
    "Actor2Religion1Code","Actor2Religion2Code","Actor2Type1Code","Actor2Type2Code","Actor2Type3Code",
    "IsRootEvent","EventCode","EventBaseCode","EventRootCode","QuadClass",
    "GoldsteinScale","NumMentions","NumSources","NumArticles","AvgTone",
    "Actor1Geo_Type","Actor1Geo_FullName","Actor1Geo_CountryCode","Actor1Geo_ADM1Code","Actor1Geo_ADM2Code",
    "Actor1Geo_Lat","Actor1Geo_Long","Actor1Geo_FeatureID",
    "Actor2Geo_Type","Actor2Geo_FullName","Actor2Geo_CountryCode","Actor2Geo_ADM1Code","Actor2Geo_ADM2Code",
    "Actor2Geo_Lat","Actor2Geo_Long","Actor2Geo_FeatureID",
    "ActionGeo_Type","ActionGeo_FullName","ActionGeo_CountryCode","ActionGeo_ADM1Code","ActionGeo_ADM2Code",
    "ActionGeo_Lat","ActionGeo_Long","ActionGeo_FeatureID",
    "DATEADDED","SOURCEURL"
]

USE_COLS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "DATEADDED",
    "SOURCEURL",
]

DTYPE_MAP = {
    "GLOBALEVENTID": "string",
    "SQLDATE": "string",
    "Actor1CountryCode": "string",
    "Actor2CountryCode": "string",
    "EventCode": "string",
    "EventBaseCode": "string",
    "EventRootCode": "string",
    "QuadClass": "string",
    "ActionGeo_Type": "string",
    "ActionGeo_FullName": "string",
    "ActionGeo_CountryCode": "string",
    "ActionGeo_ADM1Code": "string",
    "DATEADDED": "string",
    "SOURCEURL": "string",
    "GoldsteinScale": "float64",
    "NumMentions": "Int64",
    "NumSources": "Int64",
    "NumArticles": "Int64",
    "AvgTone": "float64",
    "ActionGeo_Lat": "float64",
    "ActionGeo_Long": "float64",
}

STRING_DTYPE_MAP = {col: "string" for col in USE_COLS}

FLOAT_COLS = [
    "GoldsteinScale",
    "AvgTone",
    "ActionGeo_Lat",
    "ActionGeo_Long",
]

INT_COLS = [
    "NumMentions",
    "NumSources",
    "NumArticles",
]

def read_one_zip(path: Path) -> pd.DataFrame:
    # Read all selected columns as strings first.
    # Some GDELT files contain malformed numeric values, e.g., "42#.5".
    # If we force float dtype during read_csv, the whole month can fail.
    df = pd.read_csv(
        path,
        sep="\t",
        names=ALL_COLS,
        header=None,
        usecols=USE_COLS,
        dtype=STRING_DTYPE_MAP,
        compression="zip",
        keep_default_na=True,
        na_values=[""],
        low_memory=False,
        on_bad_lines="warn",
    )

    # Convert numeric columns safely.
    # Malformed numeric values will become NaN instead of crashing the script.
    for col in FLOAT_COLS:
        if col in df.columns:
            before_bad = df[col].notna() & pd.to_numeric(df[col], errors="coerce").isna()
            n_bad = int(before_bad.sum())
            if n_bad > 0:
                print(f"[WARNING] {path.name}: {col} has {n_bad} malformed numeric values")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in INT_COLS:
        if col in df.columns:
            before_bad = df[col].notna() & pd.to_numeric(df[col], errors="coerce").isna()
            n_bad = int(before_bad.sum())
            if n_bad > 0:
                print(f"[WARNING] {path.name}: {col} has {n_bad} malformed integer values")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["sql_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")

    df["date_added_ts_utc"] = pd.to_datetime(
        df["DATEADDED"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
        utc=True,
    )

    df["date_added_ts_ny"] = df["date_added_ts_utc"].dt.tz_convert("America/New_York")
    df["event_day_ny"] = df["date_added_ts_ny"].dt.strftime("%Y-%m-%d")
    df["source_file"] = path.name

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="raw zip directory for one month")
    parser.add_argument("--output_dir", required=True, help="where daily parquet files will be written")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.export.CSV.zip"))
    if not files:
        raise FileNotFoundError(f"No zip files found in {input_dir}")

    day_to_files = {}
    for f in files:
        day = f.name[:8]
        day_to_files.setdefault(day, []).append(f)

    print(f"Found {len(files)} zip files across {len(day_to_files)} days.")

    for day, day_files in sorted(day_to_files.items()):
        out_path = output_dir / f"{day}.parquet"
        if out_path.exists():
            print(f"Skip existing {out_path}")
            continue

        parts = []
        for f in day_files:
            parts.append(read_one_zip(f))

        day_df = pd.concat(parts, ignore_index=True)
        day_df = day_df.drop_duplicates(subset=["GLOBALEVENTID"]).reset_index(drop=True)

        day_df.to_parquet(out_path, index=False)
        print(f"{day}: {len(day_files)} zips -> {day_df.shape} -> {out_path}")

if __name__ == "__main__":
    main()
