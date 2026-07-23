"""Tests for the identifier and row cleaning rules (§2 of docs/quirks.md)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clean as C
import mess as M


# --- normalise_id ------------------------------------------------------------


def test_normalise_id_strips_leading_zeros():
    dirty = pd.Series(["0173501", "0044120", "9876543"])
    assert list(C.normalise_id(dirty)) == ["173501", "44120", "9876543"]


def test_normalise_id_all_zeros_survives():
    # A value of all zeros must not collapse to an empty string.
    assert C.normalise_id(pd.Series(["0000"])).iloc[0] == "0"


def test_normalise_id_makes_the_two_systems_join():
    # The whole reason the rule exists: warehouse zero-padded string vs vendor
    # stripped integer. Normalising both sides must make them meet.
    warehouse = pd.Series(["0173501", "0044120", "0500000"])
    vendor = pd.Series(["173501", "44120", "500000"])  # stored as int, zeros gone
    assert list(C.normalise_id(warehouse)) == list(C.normalise_id(vendor.astype(str)))


def test_normalise_id_roundtrip_against_injector():
    # mess.strip_leading_zeros is the vendor-side transform; normalise_id must
    # bring a zero-padded warehouse id to the same canonical form.
    padded = pd.Series(["0173501", "0044120", "0007000"])
    vendor_side = M.strip_leading_zeros(padded)
    assert list(C.normalise_id(padded)) == list(C.normalise_id(vendor_side))


# --- flag_unmatched_ids ------------------------------------------------------


def test_flag_unmatched_ids_marks_only_orphans():
    frame = pd.DataFrame({"local_id": ["100", "200", "999", "300"]})
    reference = {"100", "200", "300"}
    out = C.flag_unmatched_ids(frame, "local_id", reference)
    assert list(out["id_unmatched"]) == [False, False, True, False]


def test_flag_unmatched_does_not_repair():
    # The typo'd id must survive untouched — flagged, not fixed.
    frame = pd.DataFrame({"local_id": ["135"]})  # a transposition of 153, say
    out = C.flag_unmatched_ids(frame, "local_id", {"153"})
    assert out["local_id"].iloc[0] == "135"
    assert out["id_unmatched"].iloc[0]


# --- drop_exact_duplicates ---------------------------------------------------


def test_drop_exact_duplicates_removes_copies():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = C.drop_exact_duplicates(frame)
    assert len(out) == 2


def test_drop_exact_duplicates_counts_removed():
    report = C.CleaningReport()
    frame = pd.DataFrame({"a": [1, 1, 1, 2]})
    C.drop_exact_duplicates(frame, report, "t")
    assert report.to_frame()["changed"].iloc[0] == 2


def test_drop_exact_duplicates_roundtrip():
    rng = np.random.default_rng(3)
    original = pd.DataFrame({"k": range(100), "v": list("abcd") * 25})
    dirtied = M.inject_duplicate_rows(original, rng, rate=0.2)
    assert len(dirtied) > len(original)
    cleaned = C.drop_exact_duplicates(dirtied)
    assert len(cleaned) == len(original)


# --- resolve_conflicting_duplicates ------------------------------------------


def test_resolve_conflicting_keeps_one_per_key():
    frame = pd.DataFrame(
        {"key": ["A", "A", "B"], "updated": [1, 2, 1], "school": ["x", "y", "z"]}
    )
    out = C.resolve_conflicting_duplicates(frame, "key", prefer="updated")
    assert len(out) == 2
    # A keeps the row with the larger 'updated' (2 -> school y).
    assert out[out["key"] == "A"]["school"].iloc[0] == "y"


def test_resolve_conflicting_keep_rule_is_directional():
    frame = pd.DataFrame({"key": ["A", "A"], "updated": [1, 2], "v": ["lo", "hi"]})
    keep_max = C.resolve_conflicting_duplicates(frame, "key", "updated", ascending=False)
    keep_min = C.resolve_conflicting_duplicates(frame, "key", "updated", ascending=True)
    assert keep_max["v"].iloc[0] == "hi"
    assert keep_min["v"].iloc[0] == "lo"
