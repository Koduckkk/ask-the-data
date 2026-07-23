"""Generate the full messy corpus in one command.

This is the entry point the README quickstart calls. It builds the canonical
roster once, then emits every source file — the warehouse tables, the wide
vendor feed, and the writing files — and prints a report of what was written
and roughly how dirty it is.

    python src/generate_data.py                 # ~50k students, the default
    python src/generate_data.py --students 400000   # production scale
    python src/generate_data.py --seed 7        # a different reproducible batch

Everything is deterministic: the same arguments always produce byte-identical
files. Regenerating is the intended way to get the data — the CSVs are not
committed, the generator is.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import roster as roster_mod
from emit_vendor import emit_vendor, emit_writing
from emit_warehouse import RAW_DIR, write_warehouse
from roster import build_roster

# Per-source seeds, kept distinct so warehouse and vendor dirt do not correlate.
# Offset from the run seed so --seed reshuffles the whole corpus coherently.
WAREHOUSE_SEED_OFFSET = 11
VENDOR_SEED_OFFSET = 23


def generate(
    students_per_year_level: int | None = None,
    seed: int = roster_mod.SEED,
    out_dir: Path = RAW_DIR,
) -> dict[str, Path]:
    """Build the roster and emit every source file. Returns {name: path}."""
    # The corpus size is a module constant the emitters read via the roster, so
    # override it here rather than threading it through every function.
    if students_per_year_level is not None:
        roster_mod.STUDENTS_PER_YEAR_LEVEL = students_per_year_level

    roster = build_roster(seed=seed)

    written: dict[str, Path] = {}
    warehouse = write_warehouse(roster, np.random.default_rng(seed + WAREHOUSE_SEED_OFFSET), out_dir)
    written.update({f"warehouse_{k}": v for k, v in warehouse.items()})

    vendor_rng = np.random.default_rng(seed + VENDOR_SEED_OFFSET)
    written.update({f"vendor_y{k}": v for k, v in emit_vendor(roster, vendor_rng, out_dir).items()})
    written.update({f"vendor_writing_{k}": v for k, v in emit_writing(roster, vendor_rng, out_dir).items()})

    return written


def _report(written: dict[str, Path], elapsed: float) -> None:
    """Print what was written and a one-line dirtiness summary."""
    import pandas as pd

    print(f"\nWrote {len(written)} files to {written[next(iter(written))].parent}/\n")
    print(f"{'file':32s} {'rows':>10s}  {'cols':>4s}  {'size':>8s}")
    print("-" * 60)
    total_rows = total_bytes = 0
    for path in sorted(p for p in written.values()):
        frame = pd.read_csv(path, dtype=str)
        size = path.stat().st_size
        total_rows += len(frame)
        total_bytes += size
        print(f"{path.name:32s} {len(frame):>10,}  {len(frame.columns):>4}  {size / 1e6:>6.1f}MB")
    print("-" * 60)
    print(f"{'total':32s} {total_rows:>10,}  {'':>4}  {total_bytes / 1e6:>6.1f}MB")
    print(f"\nGenerated in {elapsed:.1f}s. See docs/quirks.md for the defect catalogue.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--students",
        type=int,
        default=None,
        metavar="N",
        help="total students (must divide by the 4 year levels); default ~50k",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=roster_mod.SEED,
        help="RNG seed — a different value gives a different reproducible batch",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR,
        help="output directory (default data/raw/)",
    )
    args = parser.parse_args()

    per_year_level = None
    if args.students is not None:
        n_levels = len(roster_mod.YEAR_LEVELS)
        if args.students < n_levels:
            parser.error(f"--students must be at least {n_levels}")
        per_year_level = args.students // n_levels

    start = time.time()
    written = generate(per_year_level, seed=args.seed, out_dir=args.out)
    _report(written, time.time() - start)


if __name__ == "__main__":
    main()
