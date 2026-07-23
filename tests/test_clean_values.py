"""Tests for the value cleaning rules (§3 of docs/quirks.md)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clean as C
import mess as M


# --- recode_sentinels --------------------------------------------------------


def test_recode_sentinels_nulls_them():
    dirty = pd.Series(["27", "999", "15", "-1", "9999"])
    out = C.recode_sentinels(dirty)
    assert out.iloc[0] == "27" and out.iloc[2] == "15"
    assert out.iloc[[1, 3, 4]].isna().all()


def test_recode_sentinels_handles_float_spelling():
    # A CSV round-trip turns 999 into "999.0".
    assert C.recode_sentinels(pd.Series(["999.0", "-1.0"])).isna().all()


def test_recode_sentinels_does_not_touch_real_scores():
    real = pd.Series(["0", "40", "500.5"])
    assert list(C.recode_sentinels(real)) == ["0", "40", "500.5"]


# --- coerce_numeric ----------------------------------------------------------


def test_coerce_numeric_drops_text():
    dirty = pd.Series(["380.0", "absent", "412.3", "exempt"])
    out = C.coerce_numeric(dirty)
    assert out.iloc[0] == 380.0 and out.iloc[2] == 412.3
    assert out.iloc[[1, 3]].isna().all()


def test_coerce_numeric_column_is_actually_numeric():
    out = C.coerce_numeric(pd.Series(["1", "2", "x"]))
    assert pd.api.types.is_numeric_dtype(out)


# --- canonicalise_code -------------------------------------------------------


@pytest.mark.parametrize(
    "vocab,variants,canonical",
    [
        ("domain", ["NUM", "Maths", "numeracy", "Numeracy "], "Numeracy"),
        ("gender", ["M", "m", "Male", "MALE", "1"], "M"),
        ("participation", ["R", "Refused", " refused "], "R"),
    ],
)
def test_canonicalise_maps_every_variant(vocab, variants, canonical):
    out = C.canonicalise_code(pd.Series(variants), vocab)
    assert (out == canonical).all()


def test_canonicalise_leaves_unknown_values():
    out = C.canonicalise_code(pd.Series(["Numeracy", "Astronomy"]), "domain")
    assert list(out) == ["Numeracy", "Astronomy"]


def test_canonicalise_roundtrip_against_injector():
    # Inject every variant at full rate, then canonicalise: everything must
    # return to the canonical value. This is the proof the cleaner covers the
    # injector's whole vocabulary.
    rng = np.random.default_rng(5)
    original = pd.Series(["Reading", "Numeracy", "Spelling",
                          "Grammar and Punctuation", "Writing"] * 40)
    dirtied = M.inject_code_variants(original, rng, "domain", rate=1.0)
    recovered = C.canonicalise_code(dirtied, "domain")
    assert list(recovered) == list(original)


def test_canonicalise_map_built_from_injector_table():
    # Guard against drift: every variant mess can inject must be known to the
    # cleaner. If someone adds a variant to CODE_VARIANTS, this catches it.
    from mess import CODE_VARIANTS

    for vocab in CODE_VARIANTS:
        mapping = C._canonical_map(vocab)
        for canonical, variants in CODE_VARIANTS[vocab].items():
            for v in variants:
                assert mapping[v.strip().lower()] == canonical


# --- parse_year_level --------------------------------------------------------


def test_parse_year_level_unifies_int_and_string():
    dirty = pd.Series(["3", "Year 3", "9", "Year 9"])
    assert list(C.parse_year_level(dirty)) == [3, 3, 9, 9]


# --- parse_dates -------------------------------------------------------------


def test_parse_dates_handles_all_formats():
    dirty = pd.Series(["2016-03-14", "14/03/2016", "14-Mar-16"])
    out = C.parse_dates(dirty)
    assert (out == "2016-03-14").all()


def test_parse_dates_day_first_convention():
    # 03/04/2016 under day-first is 3 April, not 4 March.
    assert C.parse_dates(pd.Series(["03/04/2016"])).iloc[0] == "2016-04-03"


def test_parse_dates_unparseable_becomes_null():
    out = C.parse_dates(pd.Series(["2016-03-14", "not-a-date"]))
    assert out.iloc[0] == "2016-03-14"
    assert pd.isna(out.iloc[1])
