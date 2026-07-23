"""Tests for the presentation helpers — humanised names and chart choice."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import display as D


# --- humanise ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,pretty",
    [
        ("avg_numeracy", "Average Numeracy"),
        ("avg_scaled_score", "Average Scaled Score"),
        ("scaled_score", "Scaled Score"),
        ("year_level", "Year Level"),
        ("school_name", "School Name"),
        ("n", "Count"),
        ("student_id", "Student ID"),
    ],
)
def test_humanise(raw, pretty):
    assert D.humanise(raw) == pretty


def test_humanise_columns_renames_all():
    frame = pd.DataFrame({"avg_numeracy": [1.0], "n": [10]})
    out = D.humanise_columns(frame)
    assert list(out.columns) == ["Average Numeracy", "Count"]
    # Original frame untouched.
    assert list(frame.columns) == ["avg_numeracy", "n"]


# --- choose_chart ------------------------------------------------------------


def test_category_and_measure_is_a_bar():
    frame = pd.DataFrame({"gender": ["M", "F"], "avg_scaled_score": [1.0, 2.0], "n": [5, 6]})
    spec = D.choose_chart(frame)
    assert spec.kind == "bar"
    assert spec.label == "gender"
    # Prefers the average over the count column.
    assert spec.value == "avg_scaled_score"


def test_year_level_is_a_line():
    frame = pd.DataFrame({"year_level": [3, 5, 7, 9], "avg_writing": [1.0, 2.0, 3.0, 4.0]})
    spec = D.choose_chart(frame)
    assert spec.kind == "line"
    assert spec.label == "year_level"


def test_single_column_falls_back_to_table():
    assert D.choose_chart(pd.DataFrame({"x": [1, 2]})).kind == "table"


def test_empty_falls_back_to_table():
    assert D.choose_chart(pd.DataFrame()).kind == "table"


def test_too_many_rows_falls_back_to_table():
    frame = pd.DataFrame({"label": [str(i) for i in range(40)], "v": range(40)})
    assert D.choose_chart(frame).kind == "table"


def test_no_label_column_falls_back_to_table():
    # All-numeric with no ordinal hint — nothing sensible to put on the x-axis.
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert D.choose_chart(frame).kind == "table"
