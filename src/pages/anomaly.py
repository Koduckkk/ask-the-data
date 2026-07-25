"""Anomaly Detection — what needs a second look.

The operational question an assessment QA team asks: which schools and markers
are behaving unusually? School averages are stabilised with shrinkage (small
schools distrusted in proportion to their size), and markers are screened for
severity after controlling for the ability of the students they marked — output
as a ranked review queue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stats as S


st.title("🚩 Anomaly Detection")
st.caption(
    "Not 'what is the number' but 'what looks wrong and needs investigating'. "
    "School effects with sample-size skepticism, and a marker-severity review "
    "queue that controls for student ability."
)

with st.expander("How this works — shrinkage and marker screening"):
    st.markdown(
        "**In plain terms.** Two things on this page look unusual and need a "
        "second look: **schools** and **markers**. The hard part is telling a "
        "*real* signal from *noise* — and both methods here are built to do "
        "exactly that."
    )
    st.markdown(
        "**Schools — don't trust a tiny school's average.** A school with 8 "
        "students can top the table by luck alone. Instead of trusting each "
        "school's raw average, we **pull it toward the state average** — a lot "
        "for small schools (their average is mostly luck), barely at all for "
        "big schools (their average is solid). Big schools keep their number; "
        "small schools have to earn theirs. It's a smooth, principled version "
        "of 'ignore schools with too few students'."
    )
    st.markdown(
        "**Markers — is this marker harsh, or did they just get weak students?** "
        "Writing is marked by people, and some mark harder than others. But a "
        "low average might just mean that marker happened to get weaker students. "
        "So we first **predict** each student's writing score from how they did "
        "in the *other* subjects, then look at the gap. A marker whose scripts "
        "sit consistently below (or above) that prediction — across many "
        "scripts — is the one to review. The output is a ranked queue: *give me "
        "the top N markers to investigate*."
    )

    st.markdown("---")
    st.markdown(
        "**For the technical reader.** Schools use **empirical-Bayes shrinkage "
        "(partial pooling)**. Each school's shrunk estimate is a reliability-"
        "weighted blend of its own mean and the grand mean:"
    )
    st.latex(
        r"\hat{\mu}_s = w_s\,\bar{x}_s + (1 - w_s)\,\mu, \qquad "
        r"w_s = \frac{\tau^2}{\tau^2 + \sigma^2 / n_s}"
    )
    st.markdown(
        r"where $\tau^2$ is the between-school variance, $\sigma^2$ the "
        r"within-school variance, and $n_s$ the school's size. A large $n_s$ "
        r"drives $w_s \to 1$ (keep your own mean); a small $n_s$ drives "
        r"$w_s \to 0$ (pulled to $\mu$). This is exactly how value-added / "
        r"school-effect models work."
    )
    st.markdown(
        "Markers use **covariate-adjusted residuals**. Each script's residual "
        "removes student ability by subtracting an expectation built from the "
        "student's other-domain mean:"
    )
    st.latex(
        r"r_{i} = \text{writing}_i - \mathbb{E}[\text{writing}\mid "
        r"\text{other-domain ability}_i]"
    )
    st.markdown(
        "Averaging $r_i$ by marker isolates the **marker effect** (student "
        "ability is differenced out). Markers are then flagged by a **robust "
        "peer-relative** distance — median/MAD across markers, not distance "
        "from zero — because with thousands of scripts per marker even a "
        "trivial residual is 'significant' yet meaningless. Same logic as the "
        "school model: *explain what you can, flag what's left.*"
    )
    st.caption("See `src/stats.py` — `school_effects()` and `marker_anomalies()`.")


@st.cache_resource
def _con():
    import duckdb
    return duckdb.connect(str(S.DB_PATH), read_only=True)


con = _con()

domain = st.selectbox(
    "Domain", ["Numeracy", "Reading", "Spelling", "Grammar and Punctuation", "Writing"]
)

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
                "school": "School", "n": "Students",
                "mean": "Mean score", "std_error": "Std. error",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Standard error grows as the student count shrinks — a smaller school's "
        "average is less certain even when above the reliability threshold."
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
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Withheld from the ranking. With this few students, the average is "
            "too unstable to compare — ranking them would be reading noise."
        )

# --- school effects with shrinkage -------------------------------------------

st.header("School effects — the principled version of 'distrust small schools'")
st.markdown(
    f"The flag above uses a hard cutoff (n < {S.MIN_RELIABLE_N}). The "
    "field-standard upgrade is "
    "**partial pooling** (empirical-Bayes shrinkage): every school's estimate is "
    "pulled toward the overall mean by a weight that depends on its sample size. "
    "A big school barely moves; a tiny school is pulled most of the way in, "
    "because its own average is mostly noise. The same judgement, applied "
    "smoothly and grounded in the between- vs within-school variance."
)


@st.cache_data
def _effects(domain: str):
    return S.school_effects(domain)


effects = _effects(domain)
st.dataframe(
    effects.rename(
        columns={
            "school": "School", "n": "Students", "raw_mean": "Raw mean",
            "shrunk_estimate": "Shrunk estimate", "reliability": "Reliability",
            "pulled_by": "Pulled by",
        }
    ),
    width="stretch",
    hide_index=True,
)
st.caption(
    "See the small schools: their raw mean is pulled several points toward the "
    "centre (low reliability), while large schools keep their own estimate "
    "(reliability ≈ 1). This is how value-added / school-effect models work."
)

# --- marker anomaly detection ------------------------------------------------

st.header("Marker anomalies — a review queue")
st.markdown(
    "Writing is human-marked, so a marker's severity is a real bias — but a "
    "harsh score might just mean a weak cohort. To separate the two, each "
    "script's writing score is compared to what the student's **other-domain "
    "ability** predicts; the residual removes student ability, and averaging by "
    "marker isolates the **marker effect**. Markers are ranked by a normalised "
    "severity score — a review team works down the queue: *give me the top N "
    "markers to investigate*."
)


@st.cache_data
def _markers():
    return S.marker_anomalies()


markers = _markers()
n_flagged = int((markers["flag"] != "ok").sum())
st.error(
    f"**{n_flagged} markers flagged** for review — scoring anomalously relative "
    f"to their peers, after controlling for student ability.",
    icon="🚩",
)
st.dataframe(
    markers.rename(
        columns={
            "review_rank": "Rank", "marker": "Marker", "n_scripts": "Scripts",
            "mean_residual": "Mean residual", "severity_score": "Severity",
            "flag": "Flag",
        }
    )[["Rank", "Marker", "Scripts", "Mean residual", "Severity", "Flag"]],
    width="stretch",
    hide_index=True,
)
st.caption(
    "Severity is a robust (median/MAD) distance from the marker peer group, not "
    "from zero — because with thousands of scripts per marker, a trivial residual "
    "is 'statistically significant' yet meaningless. The two genuinely biased "
    "markers rank far above the rest; the fair markers cluster near zero."
)
