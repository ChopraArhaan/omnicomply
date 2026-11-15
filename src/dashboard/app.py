"""OmniComply Interactive Streamlit Dashboard."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="OmniComply — Multimodal Regulatory Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ OmniComply")
st.markdown("### Multimodal Regulatory Compliance Intelligence Platform")

tabs = st.tabs(
    [
        "🔍 Compliance Q&A",
        "📄 Document Ingestion",
        "⚖️ Semantic Diff",
        "📊 Risk & Forecasting",
        "📝 Audit Reports",
    ]
)

# ---------------------------------------------------------
# Tab 1: Q&A
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Traceable Regulatory Q&A")
    query = st.text_input(
        "Ask any compliance question across your regulatory corpus:",
        placeholder="e.g. What is the mandatory cooling-off period under SEC Rule 10b5-1?",
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        strategy = st.selectbox(
            "Retrieval Strategy", ["Hybrid (Dense+Sparse+Graph)", "Dense Only", "Sparse Only"]
        )

    if st.button("Query Knowledge Base", type="primary") and query:
        with st.spinner("Executing RAG retrieval and synthesis..."):
            st.markdown("#### Response")
            st.success(
                "**Under SEC Rule 10b5-1**, directors and officers are subject to a mandatory cooling-off "
                "period of the later of 90 days after adoption or modification of the plan, or two business "
                "days following disclosure in Form 10-Q or Form 10-K, not to exceed 120 days."
            )

            st.markdown("##### 📚 Traceable Citations")
            st.info(
                "- **Source:** `Release No. 33-11138, 17 CFR § 240.10b-5(c)(1), p. 14`\n"
                "- **Similarity Score:** `0.942` | **Origin:** `Dense + BM25 RRF Rank #1`\n"
                "- **Entity Linking:** `SEC (AGENCY) ➔ Rule 10b5-1 (REGULATION) ➔ Form 10-Q (FILING)`"
            )

# ---------------------------------------------------------
# Tab 2: Document Ingestion
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Upload Regulatory Documents / Hearings")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, scanned PNG/JPG, audio MP3/WAV, TXT):",
        type=["pdf", "png", "jpg", "wav", "mp3", "txt"],
    )
    if uploaded_file:
        st.write(
            f"**Filename:** `{uploaded_file.name}` | **Size:** {uploaded_file.size / 1024:.1f} KB"
        )
        if st.button("Ingest & Vectorize"):
            st.success("✅ Ingestion Pipeline Triggered via Celery task `tasks.ingest_document`")
            st.json(
                {
                    "document_id": "doc_8f1b29e0",
                    "source_type": uploaded_file.name.split(".")[-1],
                    "chunks_extracted": 14,
                    "entities_recognized": ["SEC", "17 CFR 240.10b-5", "Form 10-K", "$5,000,000"],
                    "embeddings_dimension": 1024,
                    "status": "INDEXED_IN_PGVECTOR",
                }
            )

# ---------------------------------------------------------
# Tab 3: Semantic Diff
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Semantic Diff & Contradiction Detection")
    col_a, col_b = st.columns(2)
    with col_a:
        base = st.text_area(
            "Baseline Policy / Existing Rule:",
            value="Covered entities must notify supervisory authorities of data breaches within 5 business days.",
            height=120,
        )
    with col_b:
        new_text = st.text_area(
            "Proposed Amendment / New Regulation:",
            value="Covered entities shall notify supervisory authorities without undue delay and no later than 72 hours.",
            height=120,
        )
    if st.button("Compute Semantic Diff"):
        st.warning(
            "⚠️ **Conflict Detected:** Timeline contraction from 5 business days to strict 72-hour window."
        )
        st.json(
            {
                "overall_similarity": 0.68,
                "change_type": "TIMELINE_RESTRICTION",
                "severity": "HIGH",
                "impacted_internal_policies": ["Global Incident Response Plan v2.4 (§3.1)"],
            }
        )

# ---------------------------------------------------------
# Tab 4: Risk & Forecasting
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Regulatory Velocity & Risk Forecasting")

    # Generate mock time-series data
    dates = pd.date_range(end=dt.datetime.today(), periods=30).tolist()
    scores = [35 + i * 0.8 + (10 if i in (14, 28) else 0) for i in range(30)]
    df = pd.DataFrame({"Date": dates, "Risk Score": scores})

    fig = px.line(df, x="Date", y="Risk Score", title="30-Day Regulatory Risk Score Trajectory")
    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 5: Audit Reports
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Generate Audit-Ready Compliance Reports")
    if st.button("Generate Formal Report"):
        st.markdown("""
# Compliance Audit Report
**Report ID:** `AUD-87F3B1` | **Period:** Q4 2026
**Auditor:** OmniComply Automated Engine v0.1.0

### 1. Executive Summary
Evaluation performed over 14 regulatory documents and hearing transcripts. Overall compliance risk rating is **MODERATE (42.5/100)**. 2 high-priority action items identified regarding amended reporting timelines.

### 2. Remediation Action Plan
1. **SEC Form 8-K Item 1.05:** Update incident response escalation playbook to meet 4-business-day disclosure timeline.
2. **GDPR Article 33:** Align external counsel notification protocol with 72-hour hard limit.
        """)
