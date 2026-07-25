"""Tests for the schema-drift detector.

The key behaviours: value overlap is the dominant signal (a rename is caught
even when the name barely matches), decoy columns don't false-match, and the
detector suggests rather than silently deciding.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import schema_match as SM


def _ids(n=500):
    return pd.Series([f"R{i:09d}K" for i in range(n)])


def test_present_column_is_not_flagged():
    ref = _ids()
    source = pd.DataFrame({"platform_student_id": ref, "other": ["x"] * len(ref)})
    report = SM.detect_drift(source, "student_id", "platform_student_id", ref)
    assert report.present
    assert report.suggestion is None  # nothing to suggest — it's there


def test_renamed_column_is_suggested_by_value_overlap():
    ref = _ids()
    # Column renamed AND token-reordered, but same values.
    source = pd.DataFrame({"student_platform_id": ref, "surname": ["Smith"] * len(ref)})
    report = SM.detect_drift(source, "student_id", "platform_student_id", ref)
    assert not report.present
    assert report.suggestion == "student_platform_id"


def test_value_overlap_beats_a_weak_name_match():
    # A name that barely resembles the target still wins on 100% value overlap —
    # the data is the evidence, not the label.
    ref = _ids()
    source = pd.DataFrame({"pupil_ref_xyz": ref, "code": ["P"] * len(ref)})
    report = SM.detect_drift(source, "student_id", "platform_student_id", ref)
    assert report.suggestion == "pupil_ref_xyz"
    top = report.scores[0]
    assert top.value_overlap == 1.0
    assert top.name_similarity < 0.5  # the name alone would not have found it


def test_decoy_columns_do_not_false_match():
    ref = _ids()
    source = pd.DataFrame(
        {"gender": ["M", "F"] * (len(ref) // 2), "year": [9] * len(ref)}
    )
    report = SM.detect_drift(source, "student_id", "platform_student_id", ref)
    # No column shares the ids, so nothing crosses the threshold.
    assert report.suggestion is None


def test_name_similarity_is_token_aware():
    # Reordered tokens should still score high.
    assert SM._name_similarity("platform_student_id", "student_platform_id") > 0.6
    assert SM._name_similarity("student_id", "gender") < 0.3


def test_format_fingerprint_captures_shape():
    assert SM._format_fingerprint(pd.Series(["R938098930K", "T111222333Z"])) == "A999999999A"
    assert SM._format_fingerprint(pd.Series(["0173501", "0044120"])) == "9999999"
