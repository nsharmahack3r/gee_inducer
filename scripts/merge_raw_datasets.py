"""Merge every raw fix CSV in ``RAW_DATASETS_DIR`` into a single table.

Reads all ``*.csv`` files in ``RAW_DATASETS_DIR`` (from ``.env``) and keeps
only the columns needed to place and identify a fix, plus the study it came
from so datasets stay separable:

* ``timestamp``
* ``lon``
* ``lat``
* ``individual_id``
* ``study_id``

Extra columns in the source CSVs (``species``, ...) are dropped. The merged
table is written to ``GEE_INDUCER_DIR`` by default.

Usage
-----
    uv run python scripts/merge_raw_datasets.py
    uv run python scripts/merge_raw_datasets.py --out data/merged_raw.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger("merge_raw_datasets")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEEP_COLS = ["timestamp", "lon", "lat", "individual_id", "study_id"]


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (expected in {PROJECT_ROOT / '.env'})")
    return Path(value)


def read_fixes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    return df[KEEP_COLS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path,
                        help="output CSV (default: GEE_INDUCER_DIR/merged_raw.csv)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    load_dotenv(PROJECT_ROOT / ".env")
    raw_dir = env_path("RAW_DATASETS_DIR")
    out_path = args.out or env_path("GEE_INDUCER_DIR") / "merged_raw.csv"

    files = sorted(p for p in raw_dir.glob("*.csv") if p.is_file())
    if not files:
        raise SystemExit(f"no CSV files in {raw_dir}")

    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            df = read_fixes(path)
        except ValueError as exc:
            log.error("%s; skipping", exc)
            continue
        log.info("%s: %d fixes", path.name, len(df))
        parts.append(df)

    if not parts:
        raise SystemExit(f"no usable fix CSVs in {raw_dir}")

    merged = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    log.info("wrote %s (%d fixes from %d files)", out_path, len(merged), len(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
