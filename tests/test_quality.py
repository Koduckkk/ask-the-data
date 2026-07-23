"""Tests for the data-quality showcase (before/after + refused contradiction)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import clean as C
import quality as Q


def test_pairs_shows_only_changed_values():
    raw = pd.Series(["MÃ¼ller", "Smith", "Zhang"])
    cleaned = pd.Series(["Müller", "Smith", "Zhang"])
    out = Q._pairs(raw, cleaned)
    assert list(out["before"]) == ["MÃ¼ller"]
    assert list(out["after"]) == ["Müller"]


def test_pairs_renders_null_as_change():
    # A value cleaned to null is a real change and must be shown, not dropped.
    raw = pd.Series(["N/A", "Ava"])
    cleaned = pd.Series([pd.NA, "Ava"], dtype="string")
    out = Q._pairs(raw, cleaned)
    assert list(out["before"]) == ["N/A"]
    assert list(out["after"]) == ["(null)"]


def test_before_after_examples_are_real_and_nonempty():
    stories = Q.before_after_examples(limit=3)
    assert len(stories) >= 6  # most defect types found examples
    for s in stories:
        assert not s.examples.empty
        assert list(s.examples.columns) == ["before", "after"]
        # Every shown pair genuinely differs.
        assert (s.examples["before"] != s.examples["after"]).all()


def test_before_after_names_a_real_clean_function():
    import clean as C

    for s in Q.before_after_examples(limit=2):
        assert s.function.startswith("clean.")
        # The named function actually exists in clean.py.
        assert hasattr(C, s.function.split(".", 1)[1])


def test_refused_but_attempted_surfaces_the_contradiction():
    out = Q.refused_but_attempted(limit=5)
    assert not out.empty
    assert (out["participation_code"] == "R (refused)").all()
    assert (out["corrected_score"] == "0").all()
    # The score being corrected is a plausible one, not a sentinel.
    assert not out["score_in_results"].isin(["999.0", "-1.0", "9999.0"]).any()


# --- regression: the date bug the showcase caught ----------------------------


def test_iso_dates_are_not_mangled():
    # 2016-10-08 is already ISO (8 October). Day-first parsing would flip it to
    # 2016-08-10 (10 August). It must pass through unchanged.
    out = C.parse_dates(pd.Series(["2016-10-08", "2016-03-14"]))
    assert list(out) == ["2016-10-08", "2016-03-14"]


def test_regional_dates_still_parse_day_first():
    out = C.parse_dates(pd.Series(["14/03/2016", "03/04/2016", "14-Mar-16"]))
    assert list(out) == ["2016-03-14", "2016-04-03", "2016-03-14"]
