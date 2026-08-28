"""
NetSage AI — single-page Streamlit dashboard.

    streamlit run app.py

Four tabs on one page:
    Overview       metrics, issue-type / OSI-layer mix, AI vs human agreement
    Case Browser   pick a case, run the rule checker and the AI live, review the answer
    Review Log     the full accept / edit / reject audit trail
    Responsible AI documented cases where the AI was wrong and a human corrected it
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.ai_engine import DiagnosisEngine, PROVIDERS
from src.review_manager import ReviewManager, ReviewStore
from src.rule_checker import RuleChecker

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "data" / "cases.csv"
OUTPUTS_PATH = ROOT / "data" / "ai_outputs.csv"
REVIEWS_PATH = ROOT / "data" / "reviews.csv"
RAI_PATH = ROOT / "data" / "responsible_ai_log.csv"

SEVERITY_COLOR = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Info": "⚪"}
DECISION_ICON = {"Accepted": "✅", "Edited": "✏️", "Rejected": "❌", "Not reviewed": "⏳"}
CONCEPTS = ["vlan", "trunking", "inter_vlan_routing", "addressing", "dhcp", "dns",
            "static_routing", "ospf", "nat", "acl", "wireless"]
LAYERS = ["L1", "L2", "L3", "L4", "L7"]

st.set_page_config(page_title="NetSage AI", page_icon="🛰️", layout="wide")


# ======================================================================================
# Data loading
# ======================================================================================

@st.cache_data(show_spinner=False)
def load_cases() -> List[Dict[str, Any]]:
    if not CASES_PATH.exists():
        return []
    with open(CASES_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_outputs() -> List[Dict[str, Any]]:
    if not OUTPUTS_PATH.exists():
        return []
    with open(OUTPUTS_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_rai() -> pd.DataFrame:
    if not RAI_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RAI_PATH)


def counts_frame(mapping: Dict[str, int], order: List[str], label: str) -> pd.DataFrame:
    keys = [k for k in order if k in mapping] + [k for k in mapping if k not in order]
    return pd.DataFrame({label: [mapping.get(k, 0) for k in keys]}, index=keys)


cases = load_cases()
manager = ReviewManager(ReviewStore(REVIEWS_PATH))

if not cases:
    st.error(f"No cases found. Expected {CASES_PATH}. Run this app from the netsage_ai/ folder.")
    st.stop()

case_by_id = {c["case_id"]: c for c in cases}


# ======================================================================================
# Sidebar
# ======================================================================================

with st.sidebar:
    st.title("🛰️ NetSage AI")
    st.caption("AI-assisted network troubleshooting with a mandatory human gate.")

    provider = st.selectbox(
        "AI provider",
        sorted(PROVIDERS),
        index=sorted(PROVIDERS).index("mock"),
        help="`mock` is the offline zero-cost engine and needs no key. The others read their "
             "credentials from .env.",
    )
    engine = DiagnosisEngine.from_env(provider)
    if provider == "mock":
        st.success("Offline mock engine — no API key, no network, no cost.")
    elif engine.provider.is_configured:
        st.success(f"{provider} / {engine.provider.model} — ready")
    else:
        st.warning(
            f"**{provider} is not configured.**\n\n"
            f"Add the key to `.env` (copy `.env.example`), then reload. Until then every call "
            f"silently falls back to the mock engine and is flagged as such."
        )

    reviewer = st.text_input("Reviewer name", value="reviewer",
                             help="Recorded against every decision in data/reviews.csv")

    st.divider()
    st.metric("Cases in dataset", len(cases))
    outputs = load_outputs()
    st.metric("Cases diagnosed", len(outputs))
    st.metric("Cases reviewed", len(manager.store))

    st.divider()
    if st.button("▶️ Run pipeline on all cases", use_container_width=True,
                 help="Runs the rule checker and the selected provider over every case and "
                      "rewrites data/ai_outputs.csv."):
        bar = st.progress(0.0, text="Starting…")
        rows = []
        for i, case in enumerate(cases, 1):
            res = engine.diagnose(case)
            rows.append(res.to_row(case))
            bar.progress(i / len(cases), text=f"{case['case_id']} — {res.diagnosis.concept_tag}")
        with open(OUTPUTS_PATH, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL,
                               lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        bar.empty()
        st.success(f"Diagnosed {len(rows)} cases. Reload the Overview tab.")
        st.rerun()

    with st.expander("Danger zone"):
        if st.button("Clear all review decisions", use_container_width=True):
            manager.store.clear()
            st.warning("Review log emptied.")
            st.rerun()

    st.divider()
    st.caption(
        "Pipeline order: deterministic checks → LLM diagnosis → human review. "
        "The checker runs first and its findings are injected into the prompt as verified "
        "evidence, which is what stops the model contradicting the CLI output."
    )


# ======================================================================================
# Tabs
# ======================================================================================

tab_overview, tab_browser, tab_reviews, tab_rai = st.tabs(
    ["📊 Overview", "🔎 Case Browser & Live Diagnosis", "📝 Review Log", "⚖️ Responsible AI"]
)


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------

with tab_overview:
    outputs = load_outputs()
    m = manager.metrics(cases, ai_rows=outputs)

    st.subheader("Diagnostic accuracy and human oversight")

    if not outputs:
        st.info("No AI outputs yet. Use **Run pipeline on all cases** in the sidebar, or run "
                "`python run_pipeline.py --auto-review` from a terminal.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cases", m.total_cases)
    c2.metric("Rule coverage", f"{m.rule_coverage_rate:.0%}",
              help="Share of cases where the deterministic checker found at least one fault "
                   "without any AI involvement.")
    c3.metric("Auto-graded accuracy", f"{m.auto_accuracy:.0%}",
              help="AI concept tag matches the known answer AND every cited evidence line "
                   "really appears in the CLI output.")
    c4.metric("Evidence grounded", f"{m.grounding_rate:.0%}",
              help="Anti-hallucination gate: every quoted line was verified against the case "
                   "output. An ungrounded answer can never be scored correct.")
    c5.metric("Human agreement", f"{m.agreement_rate:.0%}",
              help="Share of reviewed cases a human accepted with no edits.")

    if m.reviewed:
        st.caption(
            f"Of {m.reviewed} reviewed cases: **{m.accepted} accepted**, **{m.edited} edited**, "
            f"**{m.rejected} rejected** — {m.usable_rate:.0%} were usable as a starting point. "
            f"Mean root-cause text overlap with the verified answer is {m.mean_keyword_overlap:.2f}, "
            f"which is the honest measure of how much rewriting the reviewer still had to do."
        )

    if outputs and all(str(r.get("provider")) == "mock" for r in outputs):
        st.info(
            "These numbers come from the **offline mock engine**, which reads the same "
            "deterministic findings the rule checker produces. They confirm the pipeline is "
            "wired correctly; they are not an LLM benchmark. Re-run with Groq or Gemini in the "
            "sidebar to get a real model accuracy figure to compare against.",
            icon="ℹ️",
        )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**Cases by issue type**")
        st.bar_chart(counts_frame(m.by_concept, CONCEPTS, "cases"), height=290)
    with right:
        st.markdown("**Cases by OSI layer**")
        st.bar_chart(counts_frame(m.by_layer, LAYERS, "cases"), height=290)

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("**Cases by severity**")
        st.bar_chart(counts_frame(m.by_severity, ["Critical", "High", "Medium", "Low"], "cases"),
                     height=290)
    with right2:
        st.markdown("**Human decisions**")
        st.bar_chart(
            counts_frame(
                {"Accepted": m.accepted, "Edited": m.edited, "Rejected": m.rejected,
                 "Not reviewed": max(m.total_cases - m.reviewed, 0)},
                ["Accepted", "Edited", "Rejected", "Not reviewed"], "cases"),
            height=290,
        )

    if m.disagreement_by_concept:
        st.markdown("**Where the AI needed correcting, by issue type**")
        st.caption("Edited or rejected diagnoses. Tall bars mark the subsystems where the "
                   "assistant is least trustworthy and human review matters most.")
        st.bar_chart(counts_frame(m.disagreement_by_concept, CONCEPTS, "corrections"), height=250)

    if outputs:
        st.divider()
        st.markdown("**All cases**")
        merged = pd.DataFrame(manager.merged_table(cases, outputs))
        merged.insert(0, "", merged["decision"].map(lambda d: DECISION_ICON.get(d, "")))
        st.dataframe(
            merged[["", "case_id", "title", "concept_tag", "osi_layer", "severity",
                    "ai_concept_tag", "ai_confidence", "ai_correct", "evidence_grounded",
                    "decision", "reviewer"]],
            use_container_width=True, hide_index=True, height=420,
        )
        st.download_button("⬇️ Download merged results as CSV",
                           merged.to_csv(index=False).encode("utf-8"),
                           "netsage_results.csv", "text/csv")


# --------------------------------------------------------------------------------------
# Case browser + live diagnosis + review
# --------------------------------------------------------------------------------------

with tab_browser:
    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        concept_filter = st.multiselect("Filter by issue type", CONCEPTS, default=[])
    with fcol2:
        layer_filter = st.multiselect("OSI layer", LAYERS, default=[])
    with fcol3:
        sev_filter = st.multiselect("Severity", ["Critical", "High", "Medium", "Low"], default=[])

    visible = [
        c for c in cases
        if (not concept_filter or c["concept_tag"] in concept_filter)
        and (not layer_filter or c["osi_layer"] in layer_filter)
        and (not sev_filter or c["severity"] in sev_filter)
    ]
    if not visible:
        st.warning("No cases match those filters.")
        st.stop()

    latest = manager.store.latest()
    labels = {
        f"{DECISION_ICON.get(latest.get(c['case_id'], {}).get('decision', 'Not reviewed'), '⏳')} "
        f"{c['case_id']} — {c['title']}": c["case_id"]
        for c in visible
    }
    chosen_label = st.selectbox(f"Case ({len(visible)} shown)", list(labels))
    case = case_by_id[labels[chosen_label]]

    st.markdown(f"### {case['case_id']} — {case['title']}")
    b1, b2, b3 = st.columns(3)
    b1.markdown(f"**Issue type**  \n`{case['concept_tag']}`")
    b2.markdown(f"**OSI layer**  \n`{case['osi_layer']}`")
    b3.markdown(f"**Severity**  \n{SEVERITY_COLOR.get(case['severity'], '')} {case['severity']}")

    st.markdown("**Symptom**")
    st.info(case["symptom"])
    st.markdown("**Topology note**")
    st.caption(case["topology_note"])

    with st.expander("📟 CLI evidence (show-command output)", expanded=True):
        st.code(case["show_outputs"], language="text")

    st.divider()
    st.markdown("#### 1 · Deterministic rule checker")
    st.caption("Runs first, with no AI involved. Its findings are injected into the prompt as "
               "verified ground truth the model is not allowed to contradict.")

    findings = [f for f in RuleChecker().run(case) if f.check_id != "CHECK_ERROR"]
    if not findings:
        st.warning("No deterministic finding. This case relies entirely on the AI — treat the "
                   "answer with more suspicion and confirm it with the suggested command.")
    for f in findings:
        with st.container(border=True):
            st.markdown(f"{SEVERITY_COLOR.get(f.severity, '')} **{f.check_id}** — {f.title}  "
                        f"`{f.osi_layer}` `{f.concept_tag}`")
            st.write(f.detail)
            if f.evidence:
                st.code("\n".join(f.evidence), language="text")

    st.divider()
    st.markdown("#### 2 · AI diagnosis")

    key = f"diag::{case['case_id']}::{provider}"
    run_col, clear_col = st.columns([1, 4])
    with run_col:
        if st.button("🤖 Diagnose this case", type="primary", use_container_width=True):
            with st.spinner(f"Asking {provider}…"):
                st.session_state[key] = engine.diagnose(case, findings)
    with clear_col:
        if key in st.session_state and st.button("Clear result"):
            del st.session_state[key]
            st.rerun()

    result = st.session_state.get(key)
    if result is None:
        st.info("Press **Diagnose this case** to run the pipeline on this case.")
    else:
        d = result.diagnosis
        if result.fell_back_to_mock:
            st.error(f"The **{result.provider}** call failed and the answer below came from the "
                     f"offline mock engine instead.\n\n`{result.error}`")
        if result.parse_repaired:
            st.warning("The model's response was not clean JSON and had to be repaired before "
                       "parsing. Logged in the output CSV as `parse_repaired`.")

        conf_color = {"high": "🟢", "medium": "🟡", "low": "🟠"}[d.confidence]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Confidence", f"{conf_color} {d.confidence}")
        k2.metric("OSI layer", d.osi_layer)
        k3.metric("Issue type", d.concept_tag)
        k4.metric("Latency", f"{result.latency_ms} ms")

        st.markdown("**Root cause**")
        st.success(d.root_cause)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Evidence cited**")
            grounded = result.evidence_is_grounded(case)
            if grounded:
                st.caption("✅ Every line below was verified against the CLI output above.")
            else:
                st.caption("⚠️ At least one cited line does **not** appear in the CLI output. "
                           "Treat this answer as unverified.")
            st.code("\n".join(d.evidence) or "(none)", language="text")
        with e2:
            st.markdown("**Next command to run**")
            st.code(d.next_command or "(none)", language="text")
            st.markdown("**Risk note**")
            st.caption(d.risk_note)

        st.markdown("**Fix steps**")
        for i, step in enumerate(d.fix_steps, 1):
            st.markdown(f"{i}. {step}")

        with st.expander("Compare with the lab's verified answer"):
            st.markdown(f"**Expected fault** — {case['expected_fault']}")
            st.markdown(f"**Expected root cause** — {case['expected_root_cause']}")
            st.markdown(f"**Expected next command** — `{case['expected_next_command']}`")
            st.markdown(f"**Expected fix** — {case['expected_fix_steps']}")
            g = result.grade(case)
            st.json(g)

        with st.expander("Raw model response"):
            st.code(result.raw_response or "(empty)", language="json")

        # ------------------------------------------------------------------------------
        st.divider()
        st.markdown("#### 3 · Human review — required before any fix is applied")
        st.caption("Nothing above is a fix until a person signs it off. Every decision is "
                   "appended to data/reviews.csv with a timestamp and your name.")

        decision = st.radio(
            "Decision", ["Accepted", "Edited", "Rejected"], horizontal=True,
            captions=["Correct as written — apply it",
                      "Right direction, needed correcting",
                      "Wrong — replaced entirely"],
            key=f"dec::{case['case_id']}",
        )
        needs_correction = decision in ("Edited", "Rejected")
        with st.form(f"review::{case['case_id']}", border=True):
            corrected = st.text_area(
                "Corrected root cause" + (" (required)" if needs_correction else " (optional)"),
                value="" if needs_correction else d.root_cause, height=110,
            )
            cc1, cc2, cc3 = st.columns(3)
            corrected_concept = cc1.selectbox(
                "Corrected issue type", CONCEPTS,
                index=CONCEPTS.index(d.concept_tag) if d.concept_tag in CONCEPTS else 0)
            corrected_layer = cc2.selectbox(
                "Corrected OSI layer", LAYERS,
                index=LAYERS.index(d.osi_layer) if d.osi_layer in LAYERS else 2)
            corrected_cmd = cc3.text_input("Corrected next command", value=d.next_command)
            notes = st.text_area(
                "Reviewer notes",
                placeholder="What did the AI get wrong, and how did you know? This is the text "
                            "that makes the Responsible AI log useful to the next person.",
                height=80,
            )
            submitted = st.form_submit_button(f"Save review as **{decision}**", type="primary")

        if submitted:
            if needs_correction and not corrected.strip():
                st.error("A corrected root cause is required when editing or rejecting.")
            else:
                kwargs = dict(reviewer=reviewer, notes=notes)
                if decision == "Accepted":
                    manager.accept(case["case_id"], result, **kwargs)
                elif decision == "Edited":
                    manager.edit(case["case_id"], result, corrected.strip(),
                                 corrected_concept, corrected_layer, corrected_cmd, **kwargs)
                else:
                    manager.reject(case["case_id"], result, corrected.strip(),
                                   corrected_concept, corrected_layer, corrected_cmd, **kwargs)
                st.success(f"Saved **{decision}** for {case['case_id']} by {reviewer}.")
                st.rerun()

    prior = manager.store.for_case(case["case_id"])
    if prior:
        with st.expander(f"Review history for {case['case_id']} ({len(prior)} decision(s))"):
            st.dataframe(
                pd.DataFrame(prior)[["timestamp", "reviewer", "decision",
                                     "corrected_root_cause", "notes"]],
                use_container_width=True, hide_index=True,
            )


# --------------------------------------------------------------------------------------
# Review log
# --------------------------------------------------------------------------------------

with tab_reviews:
    st.subheader("Reviewer audit trail")
    rows = manager.store.all_rows()
    if not rows:
        st.info("No reviews recorded yet. Review a case in the **Case Browser** tab, or seed the "
                "log with `python run_pipeline.py --auto-review`.")
    else:
        df = pd.DataFrame(rows)
        st.caption(f"{len(df)} decisions across {df['case_id'].nunique()} cases. "
                   f"The most recent decision per case is the one that counts.")
        pick = st.multiselect("Filter by decision", ["Accepted", "Edited", "Rejected"], default=[])
        view = df[df["decision"].isin(pick)] if pick else df
        st.dataframe(
            view[["timestamp", "case_id", "reviewer", "decision", "ai_concept_tag",
                  "corrected_concept_tag", "corrected_root_cause", "notes", "provider"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True, hide_index=True, height=460,
        )
        st.download_button("⬇️ Download review log", df.to_csv(index=False).encode("utf-8"),
                           "reviews.csv", "text/csv")


# --------------------------------------------------------------------------------------
# Responsible AI
# --------------------------------------------------------------------------------------

with tab_rai:
    st.subheader("Responsible AI log — cases where the AI was wrong")
    st.caption(
        "Every entry is a failure actually observed while building this pipeline, with the exact "
        "flaw, the evidence it misused, the human correction, and the guardrail added afterwards. "
        "Two entries are left deliberately unfixed because they are judgement calls a human "
        "should keep making."
    )
    rai = load_rai()
    if rai.empty:
        st.info(f"No log found at {RAI_PATH}.")
    else:
        counts = rai["reviewer_decision"].value_counts().to_dict()
        r1, r2, r3 = st.columns(3)
        r1.metric("Documented failures", len(rai))
        r2.metric("Rejected outright", counts.get("Rejected", 0))
        r3.metric("Corrected by editing", counts.get("Edited", 0))

        for _, row in rai.iterrows():
            icon = "❌" if row["reviewer_decision"] == "Rejected" else "✏️"
            with st.expander(f"{icon} {row['log_id']} · {row['case_id']} — {row['failure_type']}"):
                st.markdown("**What the assistant said**")
                st.warning(row["ai_output_summary"])
                st.markdown("**The exact flaw**")
                st.write(row["exact_flaw"])
                st.markdown("**Evidence it misused**")
                st.code(str(row["evidence_the_ai_misused"]), language="text")
                st.markdown("**Human correction**")
                st.success(row["human_correction"])
                st.markdown("**Guardrail added**")
                st.info(row["guardrail_added"])
                s1, s2, s3 = st.columns(3)
                s1.caption(f"Decision: **{row['reviewer_decision']}**")
                s2.caption(f"Provider: {row['provider']}")
                s3.caption(f"Status: {row['status']}")

        with st.expander("Raw log table"):
            st.dataframe(rai, use_container_width=True, hide_index=True)
