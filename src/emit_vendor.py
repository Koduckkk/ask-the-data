"""Emit the vendor feed — wide, item-level, one file per year level.

This is the structurally hardest source, and the reason the cleaning pipeline
needs a real reshape rather than more string-tidying. Where the warehouse stores
one row per (student, domain) with a score column, the vendor delivers one row
per student and encodes the domain *in the column name*:

    PlatformId   Ylevel  N3Q01 N3Q02 ... R3Q01 ... L3Q01 ... L3Q26 ...
    R938...K     3        1     0         1         0         1

* ``N`` is Numeracy, ``R`` is Reading, ``L`` is the combined literacy block.
* The literacy block splits by question number, with nothing in the file to
  say so: ``L##`` questions 1-6 are Spelling, 26-31 are Grammar and
  Punctuation. A pipeline that treats ``L`` as one domain gets both wrong.
* A cell value of ``9`` is the "not attempted" sentinel, not a score of nine.
  It must recode to 0 before anything sums the row.

To get a tidy ``(student, domain, raw_score)`` out of this the pipeline has to
parse every header, map the prefix and number to a domain, recode the sentinel,
melt wide to long, and sum per domain. Only then does it line up with the
warehouse results — and the vendor's per-item sum is built to equal the roster's
raw score, so the two can actually be reconciled.

The vendor also spells the shared identifiers its own way (``PlatformId``,
``LocalId`` as an integer with the leading zero gone, ``Surname`` /
``FirstName``) and keeps clean names — so it is the trustworthy source when the
warehouse names are mojibaked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mess as M
from roster import ITEMS_PER_DOMAIN, Roster, build_roster

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The literacy block is one prefix covering two domains, partitioned by the
# question number. These ranges are the only thing that tells them apart.
SPELLING_QUESTIONS = range(1, 1 + ITEMS_PER_DOMAIN["Spelling"])          # L01..L06
GRAMMAR_QUESTIONS = range(26, 26 + ITEMS_PER_DOMAIN["Grammar and Punctuation"])  # L26..L31

# Domain -> (column prefix, question numbers). Writing is NOT here: it does not
# fit the 0/1 item pattern (it is marked against criteria, not right/wrong
# questions) and it arrives in its own files with a different shape — exactly as
# it does in the real feeds. See emit_writing().
_ITEM_LAYOUT = {
    "Numeracy": ("N", range(1, 1 + ITEMS_PER_DOMAIN["Numeracy"])),
    "Reading": ("R", range(1, 1 + ITEMS_PER_DOMAIN["Reading"])),
    "Spelling": ("L", SPELLING_QUESTIONS),
    "Grammar and Punctuation": ("L", GRAMMAR_QUESTIONS),
}

# Writing is marked against criteria, each scored on a small band. The column
# names are the vendor's, distinct again from anything in the warehouse.
WRITING_CRITERIA = (
    "wr_audience",
    "wr_text_structure",
    "wr_ideas",
    "wr_persuasive_devices",
    "wr_vocabulary",
    "wr_cohesion",
    "wr_paragraphing",
    "wr_sentence_structure",
    "wr_punctuation",
    "wr_spelling",
)

# The sentinel that means "not attempted", which must not be summed as a mark.
NOT_ATTEMPTED = 9


def _item_columns(year_level: int) -> list[tuple[str, str, int]]:
    """Every item column for a year level, as (column_name, domain, question)."""
    columns = []
    for domain, (prefix, questions) in _ITEM_LAYOUT.items():
        for q in questions:
            columns.append((f"{prefix}{year_level}Q{q:02d}", domain, q))
    return columns


def _distribute_marks(raw_score: int, n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Split a domain raw score into per-item 0/1 marks that sum back to it.

    The vendor's items must total the roster's raw score, or the two sources
    cannot be reconciled. Which specific items are correct is random; the count
    is fixed.
    """
    marks = np.zeros(n_items, dtype=int)
    correct = min(int(raw_score), n_items)
    if correct > 0:
        marks[rng.choice(n_items, size=correct, replace=False)] = 1
    return marks


def _build_year_level(roster: Roster, year_level: int, rng: np.random.Generator) -> pd.DataFrame:
    """One wide frame for a single year level."""
    students = roster.students[roster.students["year_level"] == year_level]
    results = roster.results[roster.results["year_level"] == year_level]

    # Vendor identity columns — its own spelling of the shared keys, and clean
    # names (the vendor is the source of truth when the warehouse is corrupted).
    frame = students[["student_id", "enrolment_id", "family_name", "given_name"]].rename(
        columns={
            "student_id": "PlatformId",
            "enrolment_id": "LocalId",
            "family_name": "Surname",
            "given_name": "FirstName",
        }
    ).copy()
    frame["Ylevel"] = year_level

    # LocalId as an integer with the leading zero gone — the join-key mismatch
    # against the warehouse's zero-padded string.
    frame["LocalId"] = M.strip_leading_zeros(frame["LocalId"]).astype("int64")

    # Per-domain raw scores for these students, to distribute across items.
    raw_by_student = (
        results[results["participation"] == "P"]
        .pivot_table(index="student_id", columns="domain", values="raw_score", aggfunc="first")
    )

    # Build every item column, distributing each domain's raw score across its
    # items so the per-item marks sum back to the warehouse score.
    frame = frame.reset_index(drop=True)
    ids = frame["PlatformId"].to_numpy()
    item_data = {}
    for domain, (prefix, questions) in _ITEM_LAYOUT.items():
        cols = [f"{prefix}{year_level}Q{q:02d}" for q in questions]
        n_items = len(questions)
        scores = (
            raw_by_student[domain].reindex(ids).to_numpy()
            if domain in raw_by_student
            else np.full(len(ids), np.nan)
        )
        block = np.zeros((len(ids), n_items), dtype=int)
        for i, s in enumerate(scores):
            if not np.isnan(s):
                block[i] = _distribute_marks(int(s), n_items, rng)
        for j, col in enumerate(cols):
            item_data[col] = block[:, j]

    items = pd.DataFrame(item_data)
    frame = pd.concat([frame, items], axis=1)

    # QUIRK: the "not attempted" sentinel. A fraction of item cells become 9,
    # which the pipeline must recode to 0 before summing — otherwise a single
    # not-attempted item inflates the raw score by nine.
    #
    # Only cells already scored 0 are overwritten. Flipping a scored 1 to the
    # sentinel would drop a real mark, so the vendor item-sum would fall short
    # of the warehouse raw score and the two sources could no longer be
    # reconciled to an exact match — which is the property the whole design
    # depends on. Recoding 9 -> 0 then leaves those cells exactly as they were.
    item_cols = list(items.columns)
    block = frame[item_cols].to_numpy(copy=True)  # copy-on-write hands back a view
    sentinel_mask = (rng.random(block.shape) < 0.03) & (block == 0)
    block[sentinel_mask] = NOT_ATTEMPTED
    frame[item_cols] = block

    return frame


def _distribute_criteria(raw_score: int, rng: np.random.Generator) -> np.ndarray:
    """Split a writing raw score across the criteria, each in band 0-6.

    Like the item marks, the criterion sub-scores sum to the warehouse raw
    score exactly, so the separate writing file reconciles the same way the
    paper files do.
    """
    n = len(WRITING_CRITERIA)
    scores = np.zeros(n, dtype=int)
    remaining = int(raw_score)
    order = rng.permutation(n)
    for c in order:
        if remaining <= 0:
            break
        take = min(6, remaining, int(rng.integers(0, 4)) + (remaining > n))
        scores[c] = take
        remaining -= take
    # Any leftover from the capped draws lands on criteria with room.
    while remaining > 0:
        c = order[int(rng.integers(n))]
        if scores[c] < 6:
            scores[c] += 1
            remaining -= 1
    return scores


def _build_writing(roster: Roster, year_levels: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    """Writing results for a set of year levels, one row per student."""
    mask = roster.results["year_level"].isin(year_levels) & (
        roster.results["domain"] == "Writing"
    ) & (roster.results["participation"] == "P")
    writing = roster.results[mask]

    ids = writing["student_id"].to_numpy()
    raws = writing["raw_score"].to_numpy()
    students = roster.students.set_index("student_id")

    frame = pd.DataFrame(
        {
            "PlatformId": ids,
            "Ylevel": writing["year_level"].to_numpy(),
            "Surname": students["family_name"].reindex(ids).to_numpy(),
            "FirstName": students["given_name"].reindex(ids).to_numpy(),
        }
    )
    block = np.vstack([_distribute_criteria(int(r), rng) for r in raws])
    for j, crit in enumerate(WRITING_CRITERIA):
        frame[crit] = block[:, j]
    return frame


def emit_writing(roster: Roster, rng: np.random.Generator, out_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Emit the two writing files: Y3 alone, Y5/7/9 combined.

    The split mirrors the real delivery layout, where Year 3 writing arrives in
    its own file and the upper year levels are combined in another. It gives the
    pipeline a "one domain, two files of different scope" assembly step on top
    of the wide/long reshape the paper files already require.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for label, year_levels in {"y3": (3,), "y579": (5, 7, 9)}.items():
        frame = _build_writing(roster, year_levels, rng)
        frame["Surname"] = M.inject_whitespace_and_case(frame["Surname"], rng, rate=0.03)
        frame = M.inject_duplicate_rows(frame, rng, rate=0.02)
        path = out_dir / f"vendor_writing_{label}.csv"
        frame.to_csv(path, index=False)
        written[label] = path
    return written


def emit_vendor(roster: Roster, rng: np.random.Generator, out_dir: Path = RAW_DIR) -> dict[int, Path]:
    """Emit one wide vendor CSV per year level. Returns {year_level: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for year_level in sorted(roster.results["year_level"].unique()):
        frame = _build_year_level(roster, int(year_level), rng)

        # Vendor names stay clean, but the id and name columns still pick up the
        # ordinary spreadsheet defects — whitespace and the odd placeholder.
        frame["Surname"] = M.inject_whitespace_and_case(frame["Surname"], rng, rate=0.03)
        frame = M.inject_duplicate_rows(frame, rng, rate=0.02)

        path = out_dir / f"vendor_y{int(year_level)}.csv"
        frame.to_csv(path, index=False)
        written[int(year_level)] = path
    return written


if __name__ == "__main__":
    roster = build_roster()
    rng = np.random.default_rng(23)
    for year_level, path in emit_vendor(roster, rng).items():
        size = path.stat().st_size / 1e6
        print(f"Y{year_level:<4} -> {path.name:26s} {size:6.1f} MB")
    for label, path in emit_writing(roster, rng).items():
        size = path.stat().st_size / 1e6
        print(f"{label:<6} -> {path.name:26s} {size:6.1f} MB")
