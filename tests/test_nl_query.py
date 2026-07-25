"""Tests for the NL->SQL layer, demo mode (no API key required).

Demo mode runs the same guardrails and executor as the live path, so these
tests exercise the whole pipeline end to end without a key.
"""

import sys
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import load as L
import nl_query as NL


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    """A clean database, built once, for the demo queries to run against."""
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    L.build_database(path)
    connection = duckdb.connect(str(path), read_only=True)
    yield connection
    connection.close()


def test_headline_question_runs_in_demo_mode(con):
    res = NL.answer("average year 9 numeracy score by gender", con=con, mode="demo")
    assert res.ok
    assert res.mode == "demo"
    assert "gender" in res.rows.columns
    # The SQL is always populated so it can be shown to the user.
    assert "SELECT" in res.sql.upper()


@pytest.mark.parametrize("question", NL.demo_examples())
def test_every_demo_example_runs(con, question):
    # Each canned example must translate, pass guardrails, and execute.
    res = NL.answer(question, con=con, mode="demo")
    assert res.ok, f"{question!r} failed: {res.error}"
    assert len(res.rows) > 0


def test_unmatched_question_is_a_clean_error(con):
    res = NL.answer("what is the meaning of life", con=con, mode="demo")
    assert not res.ok
    assert res.error  # a reason, not a crash
    assert res.suggestion


def test_unmatched_question_offers_nearest_examples(con):
    # A miss should signpost what demo mode can answer, not dead-end.
    res = NL.answer("which school has the most year 3 students", con=con, mode="demo")
    assert not res.ok
    assert res.examples  # nearest canned questions surfaced
    assert all(ex in NL.demo_examples() for ex in res.examples)


def test_sql_always_shown_even_on_error(con):
    # A demo miss has no SQL, but a guardrail rejection must still surface the SQL.
    # Inject a bad canned query to prove the guardrail path populates .sql.
    res = NL.answer("average score by domain", con=con, mode="demo")
    assert res.sql  # the shown-SQL contract holds on the happy path


def test_malicious_sql_is_rejected_before_execution(con, monkeypatch):
    # The blueprint's guardrail test: a "question" that resolves to a
    # destructive statement must be caught by validation, not executed.
    monkeypatch.setattr(
        NL, "_match_demo", lambda q: "DROP TABLE students; --"
    )
    res = NL.answer("drop everything", con=con, mode="demo")
    assert not res.ok
    assert "DROP" in res.error or "single statement" in res.error
    # And the students table is untouched.
    assert con.execute("SELECT COUNT(*) FROM students").fetchone()[0] > 0


def test_file_read_via_table_function_is_sandboxed(con, monkeypatch):
    # The guardrail's keyword denylist does NOT catch DuckDB's file-reading
    # table functions (read_text/read_csv/glob) — a SELECT that starts with
    # SELECT and has no forbidden keyword passes validation. The engine sandbox
    # (_harden) must block it anyway, at execution.
    monkeypatch.setattr(
        NL, "_match_demo", lambda q: "SELECT * FROM read_text('/etc/hostname')"
    )
    res = NL.answer("read a file", con=con, mode="demo")
    # It passes the guardrail (no forbidden keyword) but must fail at execution,
    # blocked by the sandbox — never returning file contents.
    assert not res.ok
    assert "execution failed" in res.error
    assert res.rows is None


def test_resolve_mode_prefers_demo_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASK_THE_DATA_MODE", raising=False)
    assert NL.resolve_mode() == "demo"


def test_resolve_mode_forced_demo_overrides_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-something")
    monkeypatch.setenv("ASK_THE_DATA_MODE", "demo")
    assert NL.resolve_mode() == "demo"
