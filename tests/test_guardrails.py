"""Tests for the SQL guardrails — the safety boundary before execution.

The malicious-input cases are the point: a prototype that shows a visible
rejection of INSERT/DROP/multi-statement injection is a loud signal that the
LLM's output is treated as untrusted.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import guardrails as G


# --- queries that must pass ---------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM students",
        "select gender, avg(scaled_score) from results group by gender",
        "SELECT * FROM results WHERE year_level = 9;",  # trailing ; is fine
        "WITH t AS (SELECT * FROM results) SELECT * FROM t",  # CTE
        "SELECT * FROM students -- a trailing comment\n",
    ],
)
def test_valid_selects_pass(sql):
    assert G.validate_sql(sql).ok


# --- queries that must be rejected --------------------------------------------


@pytest.mark.parametrize(
    "sql,needle",
    [
        ("INSERT INTO students VALUES (1)", "INSERT"),
        ("UPDATE results SET scaled_score = 0", "UPDATE"),
        ("DELETE FROM students", "DELETE"),
        ("DROP TABLE students", "DROP"),
        ("ALTER TABLE students ADD COLUMN x INT", "ALTER"),
        ("ATTACH 'evil.db' AS evil", "ATTACH"),
        ("PRAGMA table_info('students')", "PRAGMA"),
        ("COPY students TO '/tmp/out.csv'", "COPY"),
    ],
)
def test_write_and_ddl_rejected(sql, needle):
    result = G.validate_sql(sql)
    assert not result.ok
    assert needle in result.reason


def test_multiple_statements_rejected():
    result = G.validate_sql("SELECT * FROM students; DROP TABLE students")
    assert not result.ok
    assert "single statement" in result.reason


def test_injection_hidden_in_comment_rejected():
    # A comment must not be able to smuggle a second statement past the check.
    sql = "SELECT * FROM students; -- ok\nDROP TABLE students"
    assert not G.validate_sql(sql).ok


def test_non_select_start_rejected():
    assert not G.validate_sql("EXPLAIN SELECT * FROM students").ok


def test_empty_rejected():
    assert not G.validate_sql("").ok
    assert not G.validate_sql("   ").ok


def test_markdown_fence_is_unwrapped():
    # Models sometimes wrap SQL in ```sql fences despite instructions.
    result = G.validate_sql("```sql\nSELECT * FROM students\n```")
    assert result.ok
    assert "SELECT" in result.sql.upper()


# --- row limit ----------------------------------------------------------------


def test_row_limit_wraps_query():
    wrapped = G.enforce_row_limit("SELECT * FROM students", limit=50)
    assert "LIMIT 50" in wrapped
    assert "SELECT * FROM students" in wrapped


def test_row_limit_caps_even_with_inner_limit():
    # The outer cap applies regardless of what the inner query requested.
    wrapped = G.enforce_row_limit("SELECT * FROM students LIMIT 999999", limit=100)
    assert wrapped.rstrip().endswith("LIMIT 100")
