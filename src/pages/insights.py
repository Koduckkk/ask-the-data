"""Statistical Insights — inference, not just description.

The layer that asks 'is this signal or noise?'. A group gap comes with a
confidence interval and a plain-English verdict, and a note connects a cleaning
step to its inferential consequence. (School and marker anomaly detection live
on their own page.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stats as S


st.title("📊 Statistical Insights")
st.caption(
    "Describing the data is easy; judging it is the point. This page adds the "
    "inferential reasoning a data scientist brings — confidence intervals and "
    "signal-vs-noise."
)

st.warning(
    "**Synthetic data — this demonstrates the method.** No gender effect is "
    "built into the data, so the correct answer is usually 'no real gap', and "
    "finding that (with a CI that includes zero) is exactly what good inference "
    "should do.",
    icon="⚠️",
)


@st.cache_resource
def _con():
    import duckdb
    return duckdb.connect(str(S.DB_PATH), read_only=True)


con = _con()

# --- is the gap real or noise? -----------------------------------------------

st.header("Is the gender gap real, or noise?")
st.caption(
    "The question a panel actually cares about — not 'what is the gap' but "
    "'can we trust it'. Pick a domain and year level; the gap comes with a 95% "
    "confidence interval and a verdict."
)

c1, c2 = st.columns(2)
domain = c1.selectbox(
    "Domain", ["Numeracy", "Reading", "Spelling", "Grammar and Punctuation", "Writing"]
)
year_level = c2.selectbox("Year level", [3, 5, 7, 9], index=3)

gap = S.gender_gap(domain, year_level, con=con)

g1, g2, g3 = st.columns(3)
g1.metric(f"Female mean (n={gap.n_a:,})", f"{gap.mean_a:.1f}")
g2.metric(f"Male mean (n={gap.n_b:,})", f"{gap.mean_b:.1f}")
g3.metric("Gap (F − M)", f"{gap.difference:+.1f}")

if gap.is_significant:
    st.error(
        f"**Statistically detectable.** 95% CI [{gap.ci_low:.1f}, {gap.ci_high:.1f}] "
        f"excludes zero.",
        icon="📈",
    )
else:
    st.success(
        f"**Consistent with no real difference.** 95% CI "
        f"[{gap.ci_low:.1f}, {gap.ci_high:.1f}] includes zero.",
        icon="✅",
    )
st.write(gap.interpretation)

# --- cleaning connects to inference ------------------------------------------

st.header("Why cleaning is an inference problem")


@st.cache_data
def _impact():
    return S.sentinel_impact()


st.info(_impact())
st.caption(
    "This is the link between the pipeline and the statistics: a cleaning step "
    "isn't cosmetic — skipping it biases every estimate that follows."
)
