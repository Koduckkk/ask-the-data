"""Presentation helpers — humanise column names and pick a chart.

Kept separate from ``nl_query`` on purpose: the query layer deals in machine
column names (``avg_numeracy``) because those are what a human verifies against
the schema and the shown SQL. Prettifying happens only at the *display* edge, so
the SQL stays literal while the table and chart read naturally.

Nothing here is Streamlit-specific — ``choose_chart`` returns a plain
description of what to draw, so it is testable without a running app.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Abbreviations that appear in generated/aliased column names, expanded for
# display. Applied as whole tokens after splitting on underscores.
_EXPANSIONS = {
    "avg": "Average",
    "num": "Numeracy",
    "n": "Count",
    "id": "ID",
    "dob": "Date of Birth",
    "pct": "Percent",
    "std": "Std. Dev.",
    "min": "Minimum",
    "max": "Maximum",
    "sd": "Std. Dev.",
}


def humanise(name: str) -> str:
    """Turn a column name into a display label.

    ``avg_numeracy`` -> "Average Numeracy", ``scaled_score`` -> "Scaled Score",
    ``n`` -> "Count". Known abbreviations are expanded; everything else is title-cased.
    """
    words = [_EXPANSIONS.get(part, part) for part in name.split("_") if part]
    # Title-case only the parts that weren't already expanded to a proper form.
    return " ".join(w if w[:1].isupper() else w.capitalize() for w in words)


def humanise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with display-friendly column headers."""
    return frame.rename(columns={c: humanise(c) for c in frame.columns})


@dataclass(frozen=True)
class ChartSpec:
    """What to draw for a result — or ``kind="table"`` when a chart won't help."""

    kind: str            # "bar" | "line" | "table"
    label: str = ""      # x-axis / category column (original name)
    value: str = ""      # y-axis / measured column (original name)


def chart_frame(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    """Build the humanised, indexed frame to hand to a Streamlit chart.

    Drops rows whose label is null: a missing gender is a real data point in the
    table, but a "null" bar on a chart reads as a category and is misleading. The
    table alongside still shows it.
    """
    keep = frame[[spec.label, spec.value]].copy()
    keep = keep[keep[spec.label].notna()]
    # A bar chart reads as a ranking, so sort by value. Horizontal bars render
    # bottom-to-top, so ascending puts the largest at the top. A line chart keeps
    # its natural (ordinal) order — sorting a trend by value would scramble it.
    if spec.kind == "bar":
        keep = keep.sort_values(spec.value, ascending=True)
    keep = keep.rename(columns={spec.label: humanise(spec.label), spec.value: humanise(spec.value)})
    return keep.set_index(humanise(spec.label))


# Columns whose order implies a trend, so a line reads better than bars.
_ORDINAL_HINTS = ("year", "level", "grade")


def choose_chart(frame: pd.DataFrame) -> ChartSpec:
    """Pick a sensible default chart from the result's shape.

    The heuristic matches the shape the demo (and typical aggregate) queries
    produce: one label column plus one or more numeric columns.

    * one label + a numeric measure, label looks ordinal (year level) -> line
    * one label + a numeric measure -> bar
    * anything else (no clear label/measure, too many rows) -> table
    """
    if frame is None or frame.empty or frame.shape[1] < 2:
        return ChartSpec(kind="table")

    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    non_numeric = [c for c in frame.columns if c not in numeric]

    # The label is the first non-numeric column, or an ordinal-looking numeric
    # one (year_level is numeric but names a category).
    label = non_numeric[0] if non_numeric else None
    if label is None:
        ordinal = [c for c in numeric if any(h in c.lower() for h in _ORDINAL_HINTS)]
        label = ordinal[0] if ordinal else None
    if label is None:
        return ChartSpec(kind="table")

    # The measure is the first numeric column that isn't the label. Prefer a
    # non-count column when several exist, so "average X" charts over "n".
    measures = [c for c in numeric if c != label]
    if not measures:
        return ChartSpec(kind="table")
    value = next((m for m in measures if m.lower() not in ("n", "count")), measures[0])

    # Too many bars is noise; a table is clearer past ~25 rows.
    if len(frame) > 25:
        return ChartSpec(kind="table")

    is_ordinal = any(h in label.lower() for h in _ORDINAL_HINTS)
    return ChartSpec(kind="line" if is_ordinal else "bar", label=label, value=value)
