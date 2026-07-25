"""Ask the Data — multipage app entry point (navigation controller).

Defines the page order explicitly with st.navigation, so the sidebar reads as a
narrative: the data-science work first (anomaly detection, IRT, inference), then
the interactive query page, then the cleaning showcase that underpins it all.

    streamlit run src/app.py
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
    real problem. Detect the missing context and print a single instruction.
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

st.set_page_config(page_title="Ask the Data", page_icon="🔎", layout="wide")

_PAGES = Path(__file__).resolve().parent / "pages"

# Order is the narrative — data science first for a statistics panel, then the
# interactive query, then the cleaning that makes it all possible.
pages = [
    st.Page(_PAGES / "anomaly.py", title="Anomaly Detection", icon="🚩"),
    st.Page(_PAGES / "irt.py", title="IRT Analysis", icon="📐"),
    st.Page(_PAGES / "insights.py", title="Statistical Insights", icon="📊"),
    st.Page(_PAGES / "query.py", title="Ask the Data", icon="🔎", default=True),
    st.Page(_PAGES / "data_quality.py", title="Data Quality", icon="🧹"),
]

# One global data disclaimer in the sidebar, said once for the whole app — so
# individual pages need only their own page-specific interpretive notes, not a
# repeated "this is synthetic" warning.
with st.sidebar:
    st.caption(
        "ℹ️ **All data is synthetic**, generated from a seed — no real assessment "
        "records. The analyses demonstrate the *method*; they recover the "
        "generator's structure, not real-world truth."
    )

st.navigation(pages).run()
