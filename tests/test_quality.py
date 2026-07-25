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


# --- LLM report summary (demo path — no key needed) --------------------------


def test_demo_summary_is_grounded_on_real_counts(monkeypatch):
    # The key property: the demo summary states the ACTUAL report totals, not a
    # fabricated placeholder. Feed a known report and assert the numbers appear.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASK_THE_DATA_MODE", raising=False)
    report = pd.DataFrame(
        {
            "rule": ["canonicalise_code", "recode_sentinels", "zero_refused_scores"],
            "column": ["domain", "raw_score", "raw_score"],
            "changed": [1000, 50, 7],
            "detail": ["", "", ""],
        }
    )
    summary, mode = Q.summarise_report(report)
    assert mode == "demo"
    # The real figures — total and the specific counts — appear verbatim.
    assert "1,057" in summary   # total 1000 + 50 + 7
    assert "1,000" in summary   # largest category
    assert "50" in summary and "7" in summary


def test_demo_summary_describes_the_actual_top_rule(monkeypatch):
    # The prose for the top category must match whatever rule is actually top —
    # never a fixed "coded values" claim pinned onto e.g. parse_dates.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASK_THE_DATA_MODE", raising=False)
    report = pd.DataFrame(
        {
            "rule": ["parse_dates", "canonicalise_code"],
            "column": ["birth_date", "domain"],
            "changed": [99999, 10],
            "detail": ["", ""],
        }
    )
    summary, _ = Q.summarise_report(report)
    # It describes date parsing, not "coded values", for the top category.
    top_clause = summary.split(". It also")[0]
    assert "date" in top_clause.lower()
    assert "coded values" not in top_clause


def test_summarise_report_uses_query_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("ASK_THE_DATA_MODE", "demo")  # forced demo overrides key
    report = pd.DataFrame(
        {"rule": ["canonicalise_code"], "column": ["d"], "changed": [5], "detail": [""]}
    )
    _summary, mode = Q.summarise_report(report)
    assert mode == "demo"


def test_regional_dates_still_parse_day_first():
    out = C.parse_dates(pd.Series(["14/03/2016", "03/04/2016", "14-Mar-16"]))
    assert list(out) == ["2016-03-14", "2016-04-03", "2016-03-14"]
