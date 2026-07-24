"""Tests for the IRT analysis module.

The 2PL fit is via girth; these tests check the data-prep and packaging around
it — the response matrix is clean 0/1, the sentinel is recoded, and the fit
returns sensible item parameters. Marked so the suite can skip if girth is
absent, keeping the core pipeline tests independent of the analysis dependency.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

girth = pytest.importorskip("girth")  # skip this file if girth isn't installed

import analysis as A


def test_response_matrix_is_binary_no_sentinel():
    m = A._response_matrix(3, "Numeracy")
    values = set(np.unique(m.to_numpy()))
    assert values <= {0, 1}  # the 9 sentinel has been recoded to 0
    assert m.shape[1] == 8   # 8 numeracy items


def test_domain_of_maps_prefix_and_range():
    assert A._domain_of("N3Q01") == "Numeracy"
    assert A._domain_of("R3Q08") == "Reading"
    assert A._domain_of("L3Q03") == "Spelling"                 # L in 1..6
    assert A._domain_of("L3Q28") == "Grammar and Punctuation"  # L in 26..31
    assert A._domain_of("PlatformId") is None


def test_fit_2pl_returns_one_param_per_item():
    result = A.fit_2pl(3, "Numeracy")
    assert result.domain == "Numeracy"
    assert len(result.items) == 8
    assert list(result.items.columns) == ["item", "difficulty", "discrimination"]
    assert result.n_persons > 1000
    # Discrimination should be positive for well-behaved items.
    assert (result.items["discrimination"] > 0).all()


def test_available_fits_cover_the_item_domains():
    fits = A.available_fits()
    domains = {d for _yl, d in fits}
    assert domains == set(A.ITEM_DOMAINS)
    assert all(yl in A.YEAR_LEVELS for yl, _d in fits)
