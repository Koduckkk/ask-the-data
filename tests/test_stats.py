"""Tests for the statistical inference layer.

Uses the real database (built by a fixture), so the tests exercise the same
path the page does. scipy is required; importorskip keeps the core suite
independent of it.
"""

import sys
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

pytest.importorskip("scipy")

import load as L
import stats as S


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    L.build_database(path)
    connection = duckdb.connect(str(path), read_only=True)
    yield connection
    connection.close()


# --- gender gap + CI ---------------------------------------------------------


def test_gender_gap_returns_a_ci_and_verdict(con):
    gap = S.gender_gap("Numeracy", 9, con=con)
    assert gap.ci_low < gap.ci_high
    # The difference sits inside its own CI.
    assert gap.ci_low <= gap.difference <= gap.ci_high
    assert gap.interpretation  # a plain-English read, not empty


def test_significance_matches_whether_ci_excludes_zero(con):
    gap = S.gender_gap("Reading", 9, con=con)
    excludes_zero = not (gap.ci_low <= 0 <= gap.ci_high)
    assert gap.is_significant == excludes_zero


def test_interpretation_names_noise_when_ci_includes_zero(con):
    # For a domain with no built-in effect, the honest read mentions noise /
    # no real difference when the interval spans zero.
    gap = S.gender_gap("Writing", 9, con=con)
    if not gap.is_significant:
        assert "noise" in gap.interpretation.lower() or "no real" in gap.interpretation.lower()


# --- school ranking with small-n flagging ------------------------------------


def test_reliable_schools_meet_the_threshold(con):
    reliable, _flagged = S.school_ranking("Numeracy", con=con)
    assert (reliable["n"] >= S.MIN_RELIABLE_N).all()
    # Ranked descending by mean.
    assert reliable["mean"].is_monotonic_decreasing


def test_flagged_schools_are_below_threshold(con):
    _reliable, flagged = S.school_ranking("Numeracy", con=con)
    if not flagged.empty:
        assert (flagged["n"] < S.MIN_RELIABLE_N).all()


def test_small_schools_are_withheld_not_ranked(con):
    # A tiny school must never appear in the reliable ranking.
    reliable, flagged = S.school_ranking("Numeracy", con=con)
    overlap = set(reliable["school"]) & set(flagged["school"])
    assert not overlap


# --- cleaning -> inference impact --------------------------------------------


def test_sentinel_impact_is_grounded_and_nonempty():
    note = S.sentinel_impact()
    assert note
    # It states a real bias magnitude, not a placeholder.
    assert "points" in note and "%" in note
