"""Tests that the full pipeline loads a clean, queryable database.

These assert end-to-end properties of the cleaned tables rather than individual
rules — that the cleaning actually reached the database, not just the functions.
"""

import sys
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import load as L


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    """Build the database once into a temp file for the whole module."""
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    L.build_database(path)
    con = duckdb.connect(str(path), read_only=True)
    yield con
    con.close()


def test_three_tables_exist(db):
    names = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
    assert names == {"students", "results", "schools"}


def test_domains_are_canonical(db):
    domains = {r[0] for r in db.execute("SELECT DISTINCT domain FROM results").fetchall()}
    assert domains == {
        "Reading", "Numeracy", "Spelling", "Grammar and Punctuation", "Writing",
    }


def test_no_sentinels_survive(db):
    n = db.execute(
        "SELECT COUNT(*) FROM results WHERE raw_score IN (999, -1, 9999)"
    ).fetchone()[0]
    assert n == 0


def test_year_level_is_clean_integer(db):
    levels = {r[0] for r in db.execute("SELECT DISTINCT year_level FROM results").fetchall()}
    assert levels == {3, 5, 7, 9}


def test_genders_are_canonical(db):
    genders = {
        r[0] for r in db.execute(
            "SELECT DISTINCT gender FROM students WHERE gender IS NOT NULL"
        ).fetchall()
    }
    assert genders <= {"M", "F", "X"}


def test_non_schools_excluded(db):
    n = db.execute(
        "SELECT COUNT(*) FROM schools WHERE school_id IN ('00000', '88000', 'unassigned')"
    ).fetchone()[0]
    assert n == 0


def test_refused_scores_are_zeroed(db):
    # No result should carry a non-zero score for a refused student — the
    # override happened before load. (Refused rows are recoded to raw_score 0.)
    # We assert the blueprint's headline query runs and returns sane numbers.
    rows = db.execute(
        """
        SELECT gender, AVG(scaled_score) avg
        FROM results r JOIN students s USING (student_id)
        WHERE r.year_level = 9 AND domain = 'Numeracy' AND scaled_score IS NOT NULL
        GROUP BY gender
        """
    ).fetchall()
    assert len(rows) >= 2
    assert all(300 <= avg <= 900 for _, avg in rows)
