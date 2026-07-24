"""Data Quality — the cleaning made visible.

The second face of the app. Where the query page shows the *answers*, this page
shows the *work*: what the raw data looked like, what the pipeline changed, and
the counts per rule. It's the most distinctive part of the project — the messy
assessment data and the judgement that cleans it — brought to the surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import load as L
import quality as Q

st.set_page_config(page_title="Data Quality — Ask the Data", page_icon="🧹", layout="wide")

st.title("🧹 Data Quality")
st.caption(
    "The raw data arrives messy. This page shows exactly what the cleaning "
    "pipeline does about it — real dirty records, their cleaned form, and a "
    "count of every change. All synthetic; the defect catalogue is in "
    "`docs/quirks.md`."
)


@st.cache_data(show_spinner="Running the cleaning pipeline…")
def _report_frame():
    _tables, report = L.run_cleaning()
    return report.to_frame()


@st.cache_data(show_spinner="Reading before/after examples…")
def _examples():
    stories = Q.before_after_examples()
    return [(s.defect, s.explanation, s.examples, s.function) for s in stories]


@st.cache_data
def _refused():
    return Q.refused_but_attempted()


# --- the cleaning report as headline numbers ---------------------------------

st.header("What the pipeline changed")
st.caption("Every rule reports what it touched — the pipeline tells you what it did.")

report = _report_frame()
# Surface the most striking rules as metric tiles.
highlights = {
    "canonicalise_code": "Codes canonicalised",
    "recode_sentinels": "Score sentinels → null",
    "repair_mojibake": "Names repaired",
    "zero_refused_scores": "Refused scores zeroed",
    "drop_exact_duplicates": "Duplicate rows dropped",
    "parse_dates": "Dates parsed to ISO",
}
tiles = (
    report[report["rule"].isin(highlights)]
    .groupby("rule")["changed"].sum()
    .reindex(highlights.keys())
    .fillna(0)
    .astype(int)
)
cols = st.columns(3)
for i, (rule, label) in enumerate(highlights.items()):
    cols[i % 3].metric(label, f"{tiles[rule]:,}")

with st.expander("Full cleaning report (every rule)"):
    st.dataframe(
        report.rename(columns={"rule": "Rule", "column": "Column", "changed": "Values changed", "detail": "Detail"}),
        use_container_width=True,
        hide_index=True,
    )

# --- LLM narrative of the same report ----------------------------------------
# The AI only phrases the real counts above; the deterministic report is the
# source of truth. Live with a key, a labelled demo summary of the same figures
# without one — the same pattern the query page uses.

summary, summary_mode = Q.summarise_report(report)
label = "AI summary" if summary_mode == "llm" else "AI summary (demo — pre-written from the same counts)"
st.info(f"**{label}**\n\n{summary}")
st.caption(
    "The AI only phrases the deterministic counts above — it never counts or "
    "estimates. With an API key this is generated live; without one, a "
    "pre-written summary of the same real figures is shown."
)

# --- the signature defect ----------------------------------------------------

st.header("The one no single file reveals")
st.markdown(
    "Some students are recorded as **refused** in the participation table but "
    "still carry a plausible score in the results table. The score looks "
    "completely legitimate — only the *disagreement between two tables* exposes "
    "it. The participation code wins: a refusal scores **zero**."
)
st.code("reshape.zero_refused_scores()", language="python")
refused = _refused()
st.dataframe(refused, use_container_width=True, hide_index=True)
st.caption(
    "These scores are well-formed and in range. Nothing in the results file "
    "alone would flag them — this is a cross-table judgement, not a formatting fix."
)

# --- before / after per defect -----------------------------------------------

st.header("Before / after, by defect")
st.caption(
    "Real dirty values from the raw data, and exactly what the rule turns them "
    "into. Each transformation names the `clean.py` function that produced it — "
    "every change traces to a named, tested rule you can open."
)

for defect, explanation, examples, function in _examples():
    st.subheader(defect)
    # The function chip is the QA hook: the reviewer can go read the exact rule.
    st.caption(explanation)
    st.code(f"{function}()", language="python")
    st.dataframe(
        examples.rename(columns={"before": "Before (raw)", "after": "After (cleaned)"}),
        use_container_width=True,
        hide_index=True,
    )
