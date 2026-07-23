"""Tests for the text-defect cleaning rules (§1 of docs/quirks.md).

Each rule gets the dirty-in / clean-out treatment: a known defect goes in, the
exact repair comes out. The mojibake tests also assert the honest failure —
that lossily corrupted values are left alone rather than guessed at.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clean as C
import mess as M


# --- repair_mojibake ---------------------------------------------------------


def test_repair_mojibake_reverses_double_encoding():
    dirty = pd.Series(["MÃ¼ller", "RenÃ©e", "JosÃ©", "FranÃ§ois"])
    clean = C.repair_mojibake(dirty)
    assert list(clean) == ["Müller", "Renée", "José", "François"]


def test_repair_mojibake_leaves_lossy_corruption_alone():
    # "?" and the replacement character carry no information to recover from,
    # so the rule must not touch them — inventing a letter would be worse.
    dirty = pd.Series(["Jos?", "O�Brien", "Zo?"])
    assert list(C.repair_mojibake(dirty)) == ["Jos?", "O�Brien", "Zo?"]


def test_repair_mojibake_leaves_clean_values_unchanged():
    clean = pd.Series(["Smith", "Zoë", "O'Brien"])
    assert list(C.repair_mojibake(clean)) == ["Smith", "Zoë", "O'Brien"]


# --- blank_placeholders ------------------------------------------------------


def test_blank_placeholders_nulls_all_known_fillers():
    dirty = pd.Series(["Ava", "N/A", "n/a", "-", "unknown", ".", "null", "NA"])
    cleaned = C.blank_placeholders(dirty)
    assert cleaned.iloc[0] == "Ava"
    assert cleaned.iloc[1:].isna().all()


def test_blank_placeholders_is_case_insensitive_and_trims():
    dirty = pd.Series([" N/A ", "NULL", "Unknown"])
    assert C.blank_placeholders(dirty).isna().all()


def test_blank_placeholders_keeps_real_values():
    real = pd.Series(["Noah", "Zoe", "Mia"])
    assert list(C.blank_placeholders(real)) == ["Noah", "Zoe", "Mia"]


# --- strip_junk_characters ---------------------------------------------------


def test_strip_junk_removes_only_junk():
    dirty = pd.Series(["A^li", "Kelly>", "Zhang{", "Smith#"])
    assert list(C.strip_junk_characters(dirty)) == ["Ali", "Kelly", "Zhang", "Smith"]


def test_strip_junk_preserves_apostrophe_hyphen_accent():
    keep = pd.Series(["O'Brien", "Smith-Jones", "Renée"])
    assert list(C.strip_junk_characters(keep)) == ["O'Brien", "Smith-Jones", "Renée"]


# --- normalise_whitespace_case -----------------------------------------------


def test_normalise_collapses_padding_and_case():
    dirty = pd.Series([" p ", "P", "  MALE", "female "])
    assert list(C.normalise_whitespace_case(dirty, case="upper")) == [
        "P", "P", "MALE", "FEMALE",
    ]


def test_normalise_trim_only_leaves_case():
    dirty = pd.Series([" Mixed ", "CASE"])
    assert list(C.normalise_whitespace_case(dirty)) == ["Mixed", "CASE"]


# --- roundtrip: inject then clean recovers the original ----------------------


def test_junk_roundtrip_recovers_original():
    rng = np.random.default_rng(1)
    original = pd.Series(["Smith", "Nguyen", "Patel", "Kelly", "Zhang"] * 20)
    dirtied = M.inject_junk_characters(original, rng, rate=0.5)
    recovered = C.strip_junk_characters(dirtied)
    assert list(recovered) == list(original)


def test_placeholder_roundtrip_nulls_what_was_injected():
    rng = np.random.default_rng(2)
    original = pd.Series(["Ava", "Noah", "Mia"] * 30)
    dirtied = M.inject_null_placeholders(original, rng, rate=0.3)
    cleaned = C.blank_placeholders(dirtied)
    # Every injected placeholder is now null; every surviving value is original.
    injected = dirtied != original
    assert cleaned[injected].isna().all()
    assert (cleaned[~injected] == original[~injected]).all()


# --- the report is populated -------------------------------------------------


def test_rules_record_to_report():
    report = C.CleaningReport()
    C.strip_junk_characters(pd.Series(["A^li", "Bo"]), report, "name")
    C.blank_placeholders(pd.Series(["N/A", "Bo"]), report, "name")
    frame = report.to_frame()
    assert set(frame["rule"]) == {"strip_junk_characters", "blank_placeholders"}
    assert frame["changed"].sum() == 2
