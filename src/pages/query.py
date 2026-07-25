"""Ask the Data — query messy assessment data in plain English.

The interactive query page. A question goes in, the LLM (or a canned demo
answer) turns it into SQL, and the generated SQL sits beside the result so a
human can verify what the model wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import display as D
import nl_query as NL

mode = NL.resolve_mode()

# --- sidebar: what mode we're in and why -------------------------------------

with st.sidebar:
    st.header("Ask the Data")
    st.caption(
        "Query messy assessment data in plain English. The generated SQL is "
        "always shown so you can verify the answer."
    )
    if mode == "demo":
        st.info(
            "**Demo mode** — no API key detected. Answering from a set of "
            "canned questions. Set `ANTHROPIC_API_KEY` for free-form queries."
        )
    else:
        st.success("**Live mode** — questions are translated to SQL by the LLM.")

    st.divider()
    st.caption("Example questions")
    for example in NL.demo_examples():
        if st.button(example, width="stretch"):
            st.session_state["question"] = example

    st.divider()
    st.caption(
        "All data is synthetic. Cleaning and reshaping happen upstream in the "
        "pipeline; this page only queries the already-clean database."
    )

# --- main: ask, then show SQL beside results ---------------------------------

st.title("🔎 Ask the Data")

question = st.text_input(
    "Ask a question about the assessment data",
    key="question",
    placeholder="e.g. average writing score by year level",
)

if question:
    with st.spinner("Translating and running…"):
        result = NL.answer(question)

    left, right = st.columns([3, 2])

    with right:
        st.subheader("Generated SQL")
        if result.sql:
            st.code(result.sql, language="sql")
        else:
            st.caption("No SQL was generated for this question.")
        st.caption(f"Mode: {result.mode}")

    with left:
        st.subheader("Result")
        if result.ok:
            # Chart first when the shape suits one, then the table. Columns are
            # humanised for display only — the SQL panel keeps the literal names.
            spec = D.choose_chart(result.rows)
            pretty = D.humanise_columns(result.rows)
            if spec.kind != "table":
                chart_df = D.chart_frame(result.rows, spec)
                if spec.kind == "line":
                    st.line_chart(chart_df)
                else:
                    # Horizontal bars keep category labels readable.
                    st.bar_chart(chart_df, horizontal=True)
            st.dataframe(pretty, width="stretch", hide_index=True)
            st.caption(f"{len(result.rows):,} row(s)")
        else:
            st.warning(result.error)
            if result.suggestion:
                st.caption(result.suggestion)
            # On a demo miss, offer the closest example questions as buttons.
            for example in result.examples:
                if st.button(example, key=f"near-{example}", width="stretch"):
                    st.session_state["question"] = example
                    st.rerun()
else:
    st.caption("Type a question above, or pick an example from the sidebar.")
