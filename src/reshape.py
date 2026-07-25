"""Structural reshape of the vendor feed (§5 of docs/quirks.md).

This is the group where cleaning stops being value-tidying and becomes genuine
transformation. The vendor delivers one wide row per student with the domain
encoded in the column name (``N3Q01``, ``R3Q07``, ``L3Q26``); getting a tidy
``(student, domain, raw_score)`` out of it requires parsing every header,
splitting the combined literacy block by question number, recoding the
"not attempted" sentinel, melting wide to long, and summing per domain.

The pay-off is that the result is *checkable*. The generator built the vendor
items to sum to the warehouse raw score exactly, so once the reshape is done
correctly the two sources reconcile 100%. ``reconcile`` turns that into an
assertion: if any step above is wrong — a mis-parsed header, a forgotten
sentinel recode, the literacy block split at the wrong number — the sums stop
matching and it fails loudly. Most pipelines cannot prove their reshape is
correct; this one can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clean import CleaningReport
from emit_vendor import NOT_ATTEMPTED, WRITING_CRITERIA
# Item-column parsing and the L-block split live in one shared module, derived
# from roster.ITEMS_PER_DOMAIN, so reshape/analysis/emit_vendor can't disagree.
from items import ITEM_RE as _ITEM_RE
from items import domain_of as _domain_of



def reshape_paper(
    vendor: pd.DataFrame,
    id_column: str = "PlatformId",
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """Turn one wide vendor paper frame into tidy (student, domain, raw_score).

    Parses the item headers, recodes the ``9`` sentinel to 0, melts to long,
    maps each column to its domain, and sums. Returns a long frame with columns
    ``[id_column, domain, raw_score]``.
    """
    # Drop duplicate student rows before summing. The vendor feed re-delivers a
    # fraction of rows exactly; summing a student's items twice would double
    # their raw score. Deduping on the id keeps one row per student.
    before = len(vendor)
    vendor = vendor.drop_duplicates(subset=id_column, keep="first")
    deduped = before - len(vendor)

    item_cols = [c for c in vendor.columns if _ITEM_RE.match(c)]
    domain_by_col = {c: _domain_of(c) for c in item_cols}
    domain_by_col = {c: d for c, d in domain_by_col.items() if d is not None}

    long = vendor.melt(
        id_vars=[id_column],
        value_vars=list(domain_by_col),
        var_name="_col",
        value_name="_mark",
    )
    # Recode the not-attempted sentinel BEFORE summing — a 9 left in inflates the
    # raw score by nine.
    sentinels = (long["_mark"] == NOT_ATTEMPTED).sum()
    long["_mark"] = long["_mark"].replace(NOT_ATTEMPTED, 0)
    long["domain"] = long["_col"].map(domain_by_col)

    tidy = (
        long.groupby([id_column, "domain"])["_mark"]
        .sum()
        .reset_index()
        .rename(columns={"_mark": "raw_score"})
    )
    if report is not None:
        report.record("reshape_paper.dedup", id_column, deduped, "duplicate student rows")
        report.record("reshape_paper.recode_sentinel", "vendor items", sentinels, "9 -> 0")
        report.record("reshape_paper.melt", "vendor", len(tidy), "wide -> long")
    return tidy


def reshape_writing(
    writing: pd.DataFrame,
    id_column: str = "PlatformId",
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """Sum the writing criterion sub-scores into a single writing raw score.

    Writing arrives in its own files with one column per marking criterion.
    Summing the ``wr_*`` columns gives the domain raw score, comparable with the
    paper domains and the warehouse.
    """
    # Drop duplicate student rows before summing, as for the paper files.
    writing = writing.drop_duplicates(subset=id_column, keep="first")
    criteria = [c for c in WRITING_CRITERIA if c in writing.columns]
    tidy = writing[[id_column]].copy()
    tidy["domain"] = "Writing"
    tidy["raw_score"] = writing[criteria].sum(axis=1)
    tidy = tidy.groupby([id_column, "domain"], as_index=False)["raw_score"].first()
    if report is not None:
        report.record("reshape_writing", "wr_* criteria", len(tidy), "summed to raw_score")
    return tidy


def zero_refused_scores(
    results: pd.DataFrame,
    participation: pd.DataFrame,
    results_key: str = "student_id",
    part_key: str = "student_id",
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """Zero the score of any student the participation table records as refused.

    The refused-but-attempted contradiction: a student coded ``R`` in the
    participation table nonetheless carries a plausible score in results. The
    participation code is authoritative — a refusal scores zero — so the score
    is the artefact and must be overridden, not believed.

    This is the one defect invisible in a single file. The score is well-formed
    and in range; only the disagreement between the two tables reveals it, which
    is exactly why the join-then-override has to be explicit.

    Expects already-canonicalised ``participation_code`` and ``domain`` on both
    sides so the join lines up.
    """
    refused = participation[participation["participation_code"] == "R"][
        [part_key, "domain"]
    ].drop_duplicates()
    refused = refused.rename(columns={part_key: results_key})
    refused["_refused"] = True

    merged = results.merge(refused, on=[results_key, "domain"], how="left")
    is_refused = merged["_refused"].fillna(False)
    # Only count rows where a non-zero score is actually being overridden.
    overridden = (is_refused & merged["raw_score"].fillna(0).ne(0)).sum()

    merged.loc[is_refused, ["raw_score", "scaled_score"]] = 0
    out = merged.drop(columns="_refused")
    if report is not None:
        report.record(
            "zero_refused_scores", "raw_score", overridden, "R overrides score -> 0"
        )
    return out


def reconcile(
    vendor_long: pd.DataFrame,
    warehouse: pd.DataFrame,
    vendor_id: str = "PlatformId",
    warehouse_id: str = "student_id",
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Check the reshaped vendor totals against the warehouse raw scores.

    This is the correctness proof for the whole reshape. The generator made the
    vendor items sum to the warehouse raw score exactly, so a correct reshape
    reconciles fully. The function returns the mismatched rows; an empty result
    means the reshape reproduced every total.

    Callers assert ``reconcile(...).empty`` — a non-empty result is a bug in the
    reshape (or in the data), surfaced loudly rather than silently averaged away.
    """
    merged = vendor_long.merge(
        warehouse.rename(columns={warehouse_id: vendor_id}),
        on=[vendor_id, "domain"],
        suffixes=("_vendor", "_warehouse"),
    )
    diff = (merged["raw_score_vendor"] - merged["raw_score_warehouse"]).abs()
    return merged[diff > tolerance]


if __name__ == "__main__":
    from pathlib import Path

    import roster as roster_mod
    from emit_vendor import emit_vendor, emit_writing
    from emit_warehouse import RAW_DIR
    from roster import build_roster

    report = CleaningReport()

    # Reshape all four paper files and both writing files.
    paper_parts, writing_parts = [], []
    for yl in (3, 5, 7, 9):
        v = pd.read_csv(RAW_DIR / f"vendor_y{yl}.csv")
        paper_parts.append(reshape_paper(v, report=report))
    for label in ("y3", "y579"):
        w = pd.read_csv(RAW_DIR / f"vendor_writing_{label}.csv")
        writing_parts.append(reshape_writing(w, report=report))

    vendor_long = pd.concat(paper_parts + writing_parts, ignore_index=True)
    vendor_long = vendor_long.drop_duplicates(["PlatformId", "domain"])

    # The warehouse "truth" to reconcile against: the roster's participated raw
    # scores. (In the full pipeline this comes from the cleaned results table.)
    roster = build_roster(seed=roster_mod.SEED)
    truth = roster.results[roster.results["participation"] == "P"][
        ["student_id", "domain", "raw_score"]
    ]

    mismatches = reconcile(vendor_long, truth)
    print(report.summary())
    print()
    print(f"Reconciliation: {len(mismatches):,} mismatches out of {len(vendor_long):,} "
          f"(student, domain) pairs")
    print("PASS — vendor reshape reproduces every warehouse total"
          if mismatches.empty else "FAIL — reshape does not reconcile")
