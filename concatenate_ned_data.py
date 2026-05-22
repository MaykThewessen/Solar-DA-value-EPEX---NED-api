#!/usr/bin/env python3
"""
Concatenate monthly NED data files by year for a given source.

Reads all data_export_NED_{SOURCE}_YYYYMM.csv files in the data/ directory
and writes yearly concatenated files to data/yearly/{source_lower}_generation_YYYY.csv.

Usage:
    python concatenate_ned_data.py --source PV
    python concatenate_ned_data.py --source Wind
    python concatenate_ned_data.py --source Wind_Onshore
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Per-source schema: which production column to keep and what to rename it to.
SOURCE_SCHEMA: dict[str, tuple[str, str]] = {
    "PV": ("Solar_production_MW", "solar_generation"),
    "Wind": ("Wind_production_MW", "wind_generation"),
    "Wind_Onshore": ("Wind_production_MW", "wind_onshore_generation"),
    "Wind_Offshore": ("Wind_production_MW", "wind_offshore_generation"),
}


def _output_stem(source: str) -> str:
    """Map a NED source name to its output filename stem."""
    if source == "PV":
        return "solar"
    return source.lower()


def concatenate_ned_data_by_year(
    source: str,
    data_dir: Path = Path("data"),
    output_dir: Path | None = None,
) -> None:
    """Concatenate monthly NED data files for ``source`` into yearly CSVs.

    Files are matched by the pattern
    ``data_export_NED_{source}_YYYYMM.csv``. Output goes to
    ``{data_dir}/yearly/{stem}_generation_{YYYY}.csv``.
    """
    if source not in SOURCE_SCHEMA:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {sorted(SOURCE_SCHEMA)}"
        )

    src_col, out_col = SOURCE_SCHEMA[source]
    output_dir = output_dir or (data_dir / "yearly")
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"data_export_NED_{source}_*.csv"
    files = sorted(data_dir.glob(pattern))

    if not files:
        print(f"No {source} data files found matching {data_dir}/{pattern}.")
        return

    # Group files by year via regex on stem.
    year_re = re.compile(rf"data_export_NED_{re.escape(source)}_(\d{{4}})\d{{2}}\.csv$")
    yearly_data: dict[str, list[Path]] = {}
    for path in files:
        match = year_re.search(path.name)
        if match:
            yearly_data.setdefault(match.group(1), []).append(path)

    stem = _output_stem(source)

    for year, file_list in sorted(yearly_data.items()):
        print(f"Processing year {year} with {len(file_list)} files...")
        file_list.sort()

        frames: list[pd.DataFrame] = []
        for path in file_list:
            try:
                df = pd.read_csv(path)
                df = df[["time", src_col]].copy()
                df = df.rename(columns={src_col: out_col})
                frames.append(df)
                print(f"  - Loaded {path.name}: {len(df)} rows")
            except Exception as exc:  # noqa: BLE001 - mirror original behaviour
                print(f"  - Error loading {path}: {exc}")
                continue

        if not frames:
            print(f"  - No valid data found for year {year}")
            print()
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"])
        combined = combined.sort_values("time").reset_index(drop=True)

        out_path = output_dir / f"{stem}_generation_{year}.csv"
        combined.to_csv(out_path, index=False)

        print(f"  - Saved {out_path} with {len(combined)} total rows")
        print(f"  - Date range: {combined['time'].min()} to {combined['time'].max()}")
        print()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(SOURCE_SCHEMA),
        help="NED source to concatenate (PV, Wind, Wind_Onshore, Wind_Offshore).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing data_export_NED_{source}_YYYYMM.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: {data-dir}/yearly).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"Starting NED {args.source} data concatenation by year...")
    print("=" * 50)
    concatenate_ned_data_by_year(
        source=args.source,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print("=" * 50)
    print("Concatenation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
