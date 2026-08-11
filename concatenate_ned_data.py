#!/usr/bin/env python3
"""
Concatenate monthly NED data files by year for a given source.

Reads every data_NED_{SOURCE}_YYYYMM.csv found anywhere under data/ (they live
in per-source subdirectories, e.g. data/NED_PV/) and writes yearly concatenated
files to data/yearly/{stem}_generation_YYYY.csv.

Matching is recursive and fails loudly: a source that matches no files raises
rather than printing a note and exiting 0. The previous non-recursive pattern
looked in data/ itself and silently matched nothing once the files moved into
subdirectories, so every run reported success while producing no output.

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

    Files are matched recursively by ``data_NED_{source}_YYYYMM.csv`` anywhere
    under ``data_dir``. Output goes to
    ``{data_dir}/yearly/{stem}_generation_{YYYY}.csv``.

    Raises AssertionError if the pattern matches nothing, or if matched files
    carry no parseable YYYYMM stamp — a missing source must surface, not pass.
    """
    if source not in SOURCE_SCHEMA:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {sorted(SOURCE_SCHEMA)}"
        )

    src_col, out_col = SOURCE_SCHEMA[source]
    output_dir = output_dir or (data_dir / "yearly")

    pattern = f"data_NED_{source}_*.csv"
    files = sorted(p for p in data_dir.rglob(pattern) if output_dir not in p.parents)
    assert files, (
        f"No {source} data files matched {data_dir}/**/{pattern}. "
        f"Expected monthly exports under {data_dir}/NED_{source}/."
    )

    # Group files by year via regex on stem.
    year_re = re.compile(rf"data_NED_{re.escape(source)}_(\d{{4}})\d{{2}}\.csv$")
    yearly_data: dict[str, list[Path]] = {}
    for path in files:
        match = year_re.search(path.name)
        if match:
            yearly_data.setdefault(match.group(1), []).append(path)
    assert yearly_data, (
        f"Matched {len(files)} {source} file(s) but none carried a YYYYMM stamp; "
        f"first was {files[0].name}."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(source)

    for year, file_list in sorted(yearly_data.items()):
        print(f"Processing year {year} with {len(file_list)} files...")
        file_list.sort()

        # No try/except: an unreadable file or a renamed production column is a
        # real ingestion fault. Swallowing it silently dropped whole months from
        # the yearly output while the run still reported success.
        frames: list[pd.DataFrame] = []
        for path in file_list:
            df = pd.read_csv(path)[["time", src_col]].rename(columns={src_col: out_col})
            frames.append(df)
            print(f"  - Loaded {path.name}: {len(df)} rows")

        combined = pd.concat(frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"], utc=True)
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
