"""SQL guardrails — validate generated SQL before it ever executes.

The LLM is a translator, not a trusted author. Whatever it produces is checked
here first, and only a single read-only SELECT survives. This is the safety
boundary between "the model wrote some SQL" and "we ran it against the database".

The checks are deliberately conservative — reject anything that isn't obviously
a single, read-only query:

* strip comments (a `--` or `/* */` can hide a second statement)
* exactly one statement (no `;`-separated piggybacking)
* must start with SELECT or WITH (a CTE that resolves to a SELECT)
* reject any write/DDL/attach/pragma keyword anywhere

A rejection is not a crash: ``validate_sql`` returns a reason so the caller can
show the SQL, the reason, and a suggestion to rephrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Keywords that must never appear in a query we are willing to run. Matched as
# whole words, case-insensitively. ATTACH/PRAGMA/COPY/INSTALL/LOAD are DuckDB
# escape hatches to the filesystem or other databases, so they are blocked too.
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "create", "alter", "truncate",
    "replace", "merge", "grant", "revoke", "attach", "detach", "pragma",
    "copy", "install", "load", "export", "import", "call", "set",
)
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE
)

# A statement we are willing to run starts with SELECT, or WITH (a common-table
# expression that ultimately produces a SELECT).
_ALLOWED_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

# Comment forms that can smuggle a second statement or hide a keyword.
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one candidate query."""

    ok: bool
    sql: str = ""          # the cleaned, runnable SQL (only when ok)
    reason: str = ""       # why it was rejected (only when not ok)


def _strip_comments(sql: str) -> str:
    """Remove SQL comments so they cannot hide a keyword or statement."""
    return _BLOCK_COMMENT_RE.sub(" ", _LINE_COMMENT_RE.sub("", sql))


def validate_sql(sql: str) -> ValidationResult:
    """Validate a candidate query. Only a single read-only SELECT passes.

    Returns a ``ValidationResult`` — never raises, never executes. The caller
    decides what to do with a rejection (show it, suggest a rephrase).
    """
    if not sql or not sql.strip():
        return ValidationResult(ok=False, reason="empty query")

    cleaned = _strip_comments(sql).strip()

    # Some models wrap SQL in ```sql fences despite being asked not to; unwrap.
    cleaned = re.sub(r"^```(?:sql)?|```$", "", cleaned, flags=re.IGNORECASE).strip()

    # Trailing semicolon is fine; an *internal* one means multiple statements.
    without_trailing = cleaned.rstrip(";").strip()
    if ";" in without_trailing:
        return ValidationResult(
            ok=False, reason="only a single statement is allowed"
        )

    # Check forbidden keywords before the SELECT-start check, so a write or DDL
    # attempt is rejected with a specific reason ("'DROP' is not permitted")
    # rather than the generic "only SELECT allowed" — more useful to show, and
    # it names exactly what was blocked.
    forbidden = _FORBIDDEN_RE.search(without_trailing)
    if forbidden:
        return ValidationResult(
            ok=False,
            reason=f"'{forbidden.group(1).upper()}' is not permitted (read-only access)",
        )

    if not _ALLOWED_START_RE.match(without_trailing):
        return ValidationResult(
            ok=False, reason="only SELECT queries are allowed"
        )

    return ValidationResult(ok=True, sql=without_trailing)


def enforce_row_limit(sql: str, limit: int = 1000) -> str:
    """Wrap a validated query so it can never return an unbounded result set.

    Rather than parse and edit the query's own LIMIT, the validated SELECT is
    nested in an outer ``SELECT * FROM (...) LIMIT n``. That caps the rows a
    demo returns regardless of what the inner query asks for, without changing
    its meaning.
    """
    inner = sql.rstrip(";").strip()
    return f"SELECT * FROM (\n{inner}\n) AS _capped LIMIT {limit}"
