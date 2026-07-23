"""Natural-language querying — translate a plain-English question to SQL, run it.

This is the one place the LLM appears in the whole project, and it does exactly
one thing: turn a question into SQL. Everything around it is deterministic —
the SQL is validated by ``guardrails`` before execution, run read-only against
the DuckDB built by ``load.py``, and **always shown to the user alongside the
results**. The tool treats the model as a translator whose work is displayed
for human verification.

Two modes, chosen automatically:

* **LLM mode** — when ``ANTHROPIC_API_KEY`` is set, the question plus the schema
  doc go to the model, which returns SQL only.
* **Demo mode** — when no key is present (or ``ASK_THE_DATA_MODE=demo``), the
  question is matched against a small set of canned question -> SQL pairs, so
  an assessor without a key can still run the entire pipeline end to end.

Either way the SQL passes through the same guardrails and the same executor, so
the demo path exercises exactly the code the live path does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from guardrails import enforce_row_limit, validate_sql

DB_PATH = Path(__file__).resolve().parent.parent / "ask_the_data.duckdb"
SCHEMA_DOC = Path(__file__).resolve().parent.parent / "docs" / "schema.md"

MODEL = "claude-opus-4-8"


@dataclass
class QueryResult:
    """The full record of answering one question — SQL always included."""

    question: str
    sql: str = ""
    rows: pd.DataFrame | None = None
    mode: str = ""                 # "llm" or "demo"
    error: str = ""                # set when generation/validation/execution failed
    suggestion: str = ""           # what the user might try instead
    examples: list[str] = field(default_factory=list)  # nearest demo questions on a miss

    @property
    def ok(self) -> bool:
        return self.error == "" and self.rows is not None


# --- demo mode ---------------------------------------------------------------

# Canned question -> SQL pairs for keyless running. Chosen to cover the headline
# question patterns and to exercise every table and join the schema supports.
# Matching is on normalised keywords, so light rewording still hits.
# Ordered so the first examples make the most interesting charts — a rising
# trend over year levels, then a ranked set of schools — rather than a flat
# by-gender breakdown that (correctly) shows no effect and reads as a dull chart.
DEMO_QUERIES: list[tuple[frozenset[str], str, str]] = [
    (
        frozenset({"average", "writing", "year", "level"}),
        "average writing score by year level",
        """
        SELECT year_level, ROUND(AVG(scaled_score), 1) AS avg_writing, COUNT(*) AS n
        FROM results WHERE domain = 'Writing' AND scaled_score IS NOT NULL
        GROUP BY year_level ORDER BY year_level
        """,
    ),
    (
        frozenset({"top", "schools", "average", "numeracy"}),
        "top 10 schools by average numeracy score",
        """
        SELECT sc.school_name, ROUND(AVG(r.scaled_score), 1) AS avg_numeracy, COUNT(*) AS n
        FROM results r
        JOIN schools sc ON r.school_id = sc.school_id
        WHERE r.domain = 'Numeracy' AND r.scaled_score IS NOT NULL
        GROUP BY sc.school_name HAVING COUNT(*) >= 20
        ORDER BY avg_numeracy DESC LIMIT 10
        """,
    ),
    (
        frozenset({"average", "year", "9", "numeracy", "gender"}),
        "average year 9 numeracy score by gender",
        """
        SELECT s.gender, ROUND(AVG(r.scaled_score), 1) AS avg_scaled_score, COUNT(*) AS n
        FROM results r JOIN students s USING (student_id)
        WHERE r.year_level = 9 AND r.domain = 'Numeracy' AND r.scaled_score IS NOT NULL
        GROUP BY s.gender ORDER BY s.gender
        """,
    ),
    (
        frozenset({"average", "score", "domain"}),
        "average scaled score by domain",
        """
        SELECT domain, ROUND(AVG(scaled_score), 1) AS avg_scaled_score, COUNT(*) AS n
        FROM results WHERE scaled_score IS NOT NULL
        GROUP BY domain ORDER BY avg_scaled_score DESC
        """,
    ),
    (
        frozenset({"proficiency", "reading", "year", "5"}),
        "proficiency band distribution for year 5 reading",
        """
        SELECT proficiency, COUNT(*) AS students
        FROM results
        WHERE year_level = 5 AND domain = 'Reading' AND proficiency IS NOT NULL
        GROUP BY proficiency ORDER BY students DESC
        """,
    ),
    (
        frozenset({"how", "many", "students", "sector"}),
        "how many students sat, by school sector",
        """
        SELECT sc.sector, COUNT(DISTINCT r.student_id) AS students
        FROM results r JOIN schools sc ON r.school_id = sc.school_id
        WHERE r.scaled_score IS NOT NULL
        GROUP BY sc.sector ORDER BY students DESC
        """,
    ),
]


def _normalise(text: str) -> frozenset[str]:
    """Reduce a question to a set of lower-cased word tokens for matching."""
    import re

    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def demo_examples() -> list[str]:
    """The canned questions, for display when no key is present."""
    return [example for _keys, example, _sql in DEMO_QUERIES]


def _match_demo(question: str) -> str | None:
    """Pick the canned SQL whose keyword set best overlaps the question."""
    tokens = _normalise(question)
    best_sql, best_overlap = None, 0
    for keys, _example, sql in DEMO_QUERIES:
        overlap = len(keys & tokens)
        # Require a real overlap, and most of the pattern's keywords to be present.
        if overlap > best_overlap and overlap >= max(2, len(keys) - 1):
            best_sql, best_overlap = sql, overlap
    return best_sql


def nearest_examples(question: str, n: int = 3) -> list[str]:
    """The canned questions most similar to one that did not match.

    When a question misses in demo mode, showing the closest examples turns a
    dead end into a signpost — the user sees what demo mode *can* answer and how
    close their phrasing was.
    """
    tokens = _normalise(question)
    ranked = sorted(
        DEMO_QUERIES,
        key=lambda row: len(row[0] & tokens),
        reverse=True,
    )
    return [example for _keys, example, _sql in ranked[:n]]


# --- llm mode ----------------------------------------------------------------

_SYSTEM = """You translate a question about assessment data into a single \
DuckDB SQL query. Reply with SQL only — no prose, no markdown fences, no \
explanation. Rules:
- Output exactly one SELECT statement.
- Use only the tables and columns in the provided schema.
- Never write to the database (no INSERT/UPDATE/DELETE/DDL).
- Prefer readable aggregates; round averages to one decimal place.
- When a score is requested, exclude rows where the score is NULL."""


def _generate_sql_llm(question: str, schema: str) -> str:
    """Ask the model for SQL. Assumes a key is present (checked by the caller)."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Schema:\n\n{schema}\n\nQuestion: {question}\n\nSQL:",
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text").strip()


# --- mode selection ----------------------------------------------------------


def resolve_mode() -> str:
    """Decide which mode to run in: 'demo' unless a key is present and not forced off."""
    if os.environ.get("ASK_THE_DATA_MODE", "").lower() == "demo":
        return "demo"
    return "llm" if os.environ.get("ANTHROPIC_API_KEY") else "demo"


# --- the public entry point --------------------------------------------------


def answer(
    question: str,
    con: duckdb.DuckDBPyConnection | None = None,
    schema: str | None = None,
    mode: str | None = None,
    row_limit: int = 1000,
) -> QueryResult:
    """Answer a question end to end: generate SQL, guard it, run it, return both.

    The SQL is always populated on the result — even on error — so the caller
    can display the model's work regardless of outcome.
    """
    mode = mode or resolve_mode()
    result = QueryResult(question=question, mode=mode)

    # 1. Generate the candidate SQL.
    if mode == "demo":
        candidate = _match_demo(question)
        if candidate is None:
            result.error = "Demo mode can only answer its example questions."
            result.suggestion = (
                "Set ANTHROPIC_API_KEY for free-form questions, or try one of these:"
            )
            result.examples = nearest_examples(question)
            return result
    else:
        schema = schema if schema is not None else SCHEMA_DOC.read_text()
        candidate = _generate_sql_llm(question, schema)

    # 2. Guard it before anything runs.
    validation = validate_sql(candidate)
    result.sql = validation.sql or candidate
    if not validation.ok:
        result.error = validation.reason
        result.suggestion = "Rephrase the question, or ask for a plain SELECT over the listed tables."
        return result

    # 3. Execute read-only, row-limited.
    owns_connection = con is None
    con = con or duckdb.connect(str(DB_PATH), read_only=True)
    try:
        result.rows = con.execute(enforce_row_limit(result.sql, row_limit)).fetchdf()
    except Exception as exc:  # a valid-looking query can still fail at runtime
        result.error = f"execution failed: {exc}"
        result.suggestion = "The query was well-formed but did not run — check column names against the schema."
    finally:
        if owns_connection:
            con.close()
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "average writing score by year level"
    res = answer(q)
    print(f"[{res.mode} mode] {res.question}\n")
    print(f"SQL:\n{res.sql}\n")
    if res.ok:
        print(res.rows.to_string(index=False))
    else:
        print(f"Error: {res.error}\n{res.suggestion}")
        for example in res.examples:
            print(f"  - {example}")
