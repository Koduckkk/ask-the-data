"""Tests for the structural reshape (§5 of docs/quirks.md).

The centrepiece is the reconciliation test: reshape the real vendor feed and
assert it reproduces every warehouse total. That single assertion proves the
header parsing, the L-block split, the sentinel recode, the dedup and the sum
are all correct at once — if any were wrong, the totals would diverge.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import roster as roster_mod
import reshape as R
from emit_warehouse import RAW_DIR
from roster import build_roster


# --- header -> domain mapping ------------------------------------------------


@pytest.mark.parametrize(
    "column,domain",
    [
        ("N3Q01", "Numeracy"),
        ("R3Q08", "Reading"),
        ("L3Q01", "Spelling"),               # L in 1..6
        ("L3Q06", "Spelling"),
        ("L3Q26", "Grammar and Punctuation"),  # L in 26..31
        ("L3Q31", "Grammar and Punctuation"),
        ("PlatformId", None),                 # not an item column
        ("Ylevel", None),
    ],
)
def test_domain_of_column(column, domain):
    assert R._domain_of(column) == domain


def test_literacy_block_splits_by_question_number():
    # The whole subtlety: same L prefix, different domain by number.
    assert R._domain_of("L5Q03") == "Spelling"
    assert R._domain_of("L5Q29") == "Grammar and Punctuation"


# --- reshape mechanics on a tiny frame ---------------------------------------


def test_reshape_paper_sums_and_recodes_sentinel():
    wide = pd.DataFrame(
        {
            "PlatformId": ["S1"],
            "Ylevel": [3],
            "N3Q01": [1], "N3Q02": [9], "N3Q03": [1],   # the 9 must recode to 0
            "R3Q01": [1], "R3Q02": [0],
        }
    )
    tidy = R.reshape_paper(wide)
    num = tidy[tidy["domain"] == "Numeracy"]["raw_score"].iloc[0]
    read = tidy[tidy["domain"] == "Reading"]["raw_score"].iloc[0]
    assert num == 2   # 1 + (9->0) + 1
    assert read == 1


def test_reshape_paper_dedups_before_summing():
    # A duplicated student row must not double the raw score.
    wide = pd.DataFrame(
        {"PlatformId": ["S1", "S1"], "N3Q01": [1, 1], "N3Q02": [1, 1]}
    )
    tidy = R.reshape_paper(wide)
    assert tidy[tidy["domain"] == "Numeracy"]["raw_score"].iloc[0] == 2


def test_reshape_writing_sums_criteria():
    w = pd.DataFrame(
        {"PlatformId": ["S1"], "wr_audience": [3], "wr_ideas": [2], "wr_spelling": [1]}
    )
    tidy = R.reshape_writing(w)
    assert tidy["raw_score"].iloc[0] == 6
    assert tidy["domain"].iloc[0] == "Writing"


# --- the reconciliation proof (the headline) ---------------------------------


def test_paper_reshape_reconciles_exactly():
    """The mechanically-scored paper domains reconcile 100% with the warehouse.

    Numeracy/Reading/Spelling/Grammar are 0/1 item-scored, not human-marked, so
    their item sums must equal the warehouse raw score exactly. This is the proof
    of correctness for the structural reshape.
    """
    parts = [R.reshape_paper(pd.read_csv(RAW_DIR / f"vendor_y{yl}.csv")) for yl in (3, 5, 7, 9)]
    vendor_long = pd.concat(parts, ignore_index=True).drop_duplicates(["PlatformId", "domain"])

    roster = build_roster(seed=roster_mod.SEED)
    truth = roster.results[roster.results["participation"] == "P"][
        ["student_id", "domain", "raw_score"]
    ]

    mismatches = R.reconcile(vendor_long, truth)
    assert mismatches.empty, (
        f"{len(mismatches)} paper pairs did not reconcile — the reshape is wrong"
    )


def test_writing_reconciles_up_to_marker_bias():
    """Writing is human-marked, so it reconciles only up to marker severity.

    Unlike the mechanical paper domains, writing carries a marker effect: a harsh
    or lenient marker shifts the score away from the student's true level. So most
    writing scripts still match, but a fraction (those marked by biased markers)
    deviate — and that deviation is the signal the marker-anomaly model detects.
    """
    parts = [
        R.reshape_writing(pd.read_csv(RAW_DIR / f"vendor_writing_{label}.csv"))
        for label in ("y3", "y579")
    ]
    writing_long = pd.concat(parts, ignore_index=True).drop_duplicates(["PlatformId", "domain"])

    roster = build_roster(seed=roster_mod.SEED)
    truth = roster.results[
        (roster.results["participation"] == "P") & (roster.results["domain"] == "Writing")
    ][["student_id", "domain", "raw_score"]]

    mismatches = R.reconcile(writing_long, truth)
    frac = len(mismatches) / len(writing_long)
    # Most scripts reconcile; a minority (biased markers ~2 of 10) deviate.
    assert 0.05 < frac < 0.5, f"unexpected writing mismatch fraction: {frac:.2f}"


def test_reconcile_detects_a_deliberate_error():
    # Prove the assertion actually bites: corrupt one total and expect a catch.
    vendor = pd.DataFrame(
        {"PlatformId": ["S1", "S2"], "domain": ["Numeracy", "Numeracy"],
         "raw_score": [5, 6]}
    )
    truth = pd.DataFrame(
        {"student_id": ["S1", "S2"], "domain": ["Numeracy", "Numeracy"],
         "raw_score": [5, 99]}   # S2 deliberately wrong
    )
    mismatches = R.reconcile(vendor, truth)
    assert len(mismatches) == 1
    assert mismatches["PlatformId"].iloc[0] == "S2"


# --- refused-but-attempted (cross-table) -------------------------------------


def test_zero_refused_scores_overrides_the_score():
    results = pd.DataFrame(
        {
            "student_id": ["S1", "S2", "S3"],
            "domain": ["Reading", "Reading", "Numeracy"],
            "raw_score": [10.0, 20.0, 15.0],
            "scaled_score": [400.0, 450.0, 420.0],
        }
    )
    participation = pd.DataFrame(
        {
            "student_id": ["S1", "S2", "S3"],
            "domain": ["Reading", "Reading", "Numeracy"],
            "participation_code": ["P", "R", "P"],   # S2 refused
        }
    )
    out = R.zero_refused_scores(results, participation)
    s2 = out[out["student_id"] == "S2"].iloc[0]
    assert s2["raw_score"] == 0 and s2["scaled_score"] == 0
    # The participants are untouched.
    assert out[out["student_id"] == "S1"]["raw_score"].iloc[0] == 10.0


def test_zero_refused_counts_only_real_overrides():
    report = R.CleaningReport()
    results = pd.DataFrame(
        {"student_id": ["S1"], "domain": ["Reading"], "raw_score": [12.0],
         "scaled_score": [400.0]}
    )
    participation = pd.DataFrame(
        {"student_id": ["S1"], "domain": ["Reading"], "participation_code": ["R"]}
    )
    R.zero_refused_scores(results, participation, report=report)
    assert report.to_frame()["changed"].iloc[0] == 1
