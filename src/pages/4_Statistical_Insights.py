"""Statistical Insights — inference, not just description.

The layer that asks 'is this signal or noise?'. A group gap comes with a
confidence interval and a plain-English verdict; school rankings refuse to
trust schools with too few students; and a note connects a cleaning step to its
inferential consequence. This is the data-scientist judgement on top of the
data-engineering pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stats as S

st.set_page_config(page_title="Statistical Insights — Ask the Data", page_icon="📊", layout="wide")

st.title("📊 Statistical Insights")
st.caption(
    "Describing the data is easy; judging it is the point. This page adds the "
    "inferential reasoning a data scientist brings — confidence intervals, "
    "signal-vs-noise, and skepticism about small samples."
)

st.warning(
    "**Synthetic data — this demonstrates the method.** No gender effect is "
    "built into the data, so the correct answer is usually 'no real gap', and "
    "finding that (with a CI that includes zero) is exactly what good inference "
    "should do. The small-sample school flagging is the real signal.",
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

# --- schools: rank the reliable, flag the rest -------------------------------

st.header("School ranking — with sample-size skepticism")
st.markdown(
    "Ranking schools by average score is where naïve analysis goes wrong: a "
    "school with a handful of students can top the table on noise alone. This "
    f"ranks only schools with at least **{S.MIN_RELIABLE_N} students**, reports "
    "each average's **standard error** (how precisely it's pinned down), and "
    "**withholds** the small ones as unreliable rather than ranking them."
)

reliable, flagged = S.school_ranking(domain, con=con)

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Reliable schools ({len(reliable)})")
    st.dataframe(
        reliable.rename(
            columns={
                "school": "School",
                "n": "Students",
                "mean": "Mean score",
                "std_error": "Std. error",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Standard error grows as the student count shrinks — a smaller school's "
        "average is less certain even when it is above the reliability threshold."
    )
with right:
    st.subheader(f"Flagged: too few students ({len(flagged)})")
    if flagged.empty:
        st.caption("No schools fell below the threshold in this domain.")
    else:
        st.dataframe(
            flagged.rename(
                columns={"school": "School", "n": "Students", "mean": "Mean (unreliable)"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Withheld from the ranking. With this few students, the average is "
            "too unstable to compare — ranking them would be reading noise."
        )

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
