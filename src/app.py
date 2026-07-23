"""Ask the Data — a minimal interface over the NL->SQL layer.

Deliberately thin. The interesting code is in nl_query.py and guardrails.py;
this page just wires a text box to answer() and lays the result out so the
design signature is visible: the generated SQL sits next to the results, always,
so a human can verify what the model wrote.

    streamlit run src/app.py

Runs in demo mode with no API key (a set of canned questions), or translates
free-form questions with a key present. The mode is shown in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


def _require_streamlit_run() -> None:
    """Fail fast with one clear hint if run as `python app.py`.

    A Streamlit script only works under `streamlit run`, which installs a
    ScriptRunContext. Run with plain `python` there is none, and every Streamlit
    call emits a "missing ScriptRunContext" warning — dozens of them, burying the
    real problem. Detect the missing context and print a single instruction
    instead.
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        print(
            "This is a Streamlit app — run it with:\n\n"
            "    streamlit run src/app.py\n\n"
            "(plain `python src/app.py` does not start the server).",
            file=sys.stderr,
        )
        raise SystemExit(1)


_require_streamlit_run()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nl_query as NL

st.set_page_config(page_title="Ask the Data", page_icon="🔎", layout="wide")

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
        # Clicking an example fills the box (works in either mode).
        if st.button(example, use_container_width=True):
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
    placeholder="e.g. average year 9 numeracy score by gender",
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
            st.dataframe(result.rows, use_container_width=True, hide_index=True)
            st.caption(f"{len(result.rows):,} row(s)")
        else:
            st.warning(result.error)
            if result.suggestion:
                st.caption(result.suggestion)
            # On a demo miss, offer the closest example questions as buttons.
            for example in result.examples:
                if st.button(example, key=f"near-{example}", use_container_width=True):
                    st.session_state["question"] = example
                    st.rerun()
else:
    st.caption("Type a question above, or pick an example from the sidebar.")
