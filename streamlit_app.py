import json
import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional
from snowflake.snowpark.context import get_active_session

# ---------------------------------------------------------------------------
# Connection — works in both warehouse runtime (SiS) and local dev
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_session():
    try:
        return get_active_session()
    except Exception:
        import os
        conn_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME") or "default"
        from snowflake.snowpark import Session
        return Session.builder.configs({"connection_name": conn_name}).create()

session = _get_session()

DB = "HEALTHCARE_COPILOT"
RAW = f"{DB}.RAW_DATA"
ANA = f"{DB}.ANALYTICS"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def run_query(sql: str) -> pd.DataFrame:
    return session.sql(sql).to_pandas()

def search_documents(query: str, limit: int = 5, filter_obj: Optional[dict] = None):
    """Call Cortex Search Service and return results."""
    svc_name = f"{ANA}.CLINICAL_DOC_SEARCH"
    cols = ["DOC_ID", "PATIENT_ID", "DOC_TYPE", "TITLE", "AUTHOR", "CREATED_DATE", "CONTENT"]
    payload = {
        "query": query,
        "columns": cols,
        "limit": limit,
    }
    if filter_obj:
        payload["filter"] = filter_obj
    payload_str = json.dumps(payload).replace("'", "''")
    result = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
            '{svc_name}',
            '{payload_str}'
        ) AS RESULT
    """).collect()
    raw = result[0]["RESULT"]
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    return parsed.get("results", [])

def ask_copilot(question: str, context_docs: list) -> str:
    """Send question + retrieved context to Cortex COMPLETE and get a cited answer."""
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        context_text += f"\n\n--- SOURCE [{i}]: {doc.get('TITLE', 'Untitled')} (Type: {doc.get('DOC_TYPE', 'unknown')}) ---\n"
        content = doc.get("CONTENT", "")
        context_text += content[:3000]

    prompt = f"""You are a clinical copilot for healthcare professionals. Answer the following question using ONLY the provided source documents. 

RULES:
- Cite your sources using [1], [2], etc. corresponding to the SOURCE numbers.
- If the sources do not contain enough information to fully answer, say so explicitly.
- Be precise and clinical. Use standard medical terminology.
- Never make up information not found in the sources.
- Structure your answer with clear sections if the question is complex.

QUESTION: {question}

CONTEXT DOCUMENTS:
{context_text}

ANSWER (with citations):"""

    escaped = prompt.replace("'", "''").replace("\\", "\\\\")
    result = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            '{escaped}'
        ) AS RESPONSE
    """).collect()
    return result[0]["RESPONSE"]


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .risk-high { background-color: #FFEBEE; border-left: 4px solid #E53935; padding: 10px; border-radius: 4px; margin: 4px 0; }
    .risk-medium { background-color: #FFF8E1; border-left: 4px solid #FFC107; padding: 10px; border-radius: 4px; margin: 4px 0; }
    .risk-low { background-color: #E8F5E9; border-left: 4px solid #43A047; padding: 10px; border-radius: 4px; margin: 4px 0; }
    .metric-card { background-color: #F8F9FA; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #E0E0E0; }
    .source-card { background-color: #F5F5F5; border: 1px solid #E0E0E0; border-radius: 6px; padding: 12px; margin: 8px 0; }
    div[data-testid="stTabs"] button { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("Healthcare Copilot")
st.caption("Patient 360 · Risk Stratification · Evidence-Based Q&A")

# ---------------------------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Patient 360", "Risk Dashboard", "Copilot Q&A"])

# ========================== TAB 1: PATIENT 360 =============================
with tab1:
    patients_df = run_query(f"SELECT PATIENT_ID, FIRST_NAME || ' ' || LAST_NAME AS NAME FROM {RAW}.PATIENTS ORDER BY LAST_NAME, FIRST_NAME")
    options = {row["PATIENT_ID"]: f"{row['NAME']} ({row['PATIENT_ID']})" for _, row in patients_df.iterrows()}

    selected_pid = st.selectbox("Select Patient", options.keys(), format_func=lambda x: options[x], key="p360_patient")

    if selected_pid:
        # Demographics
        demo = run_query(f"""
            SELECT p.*, r.RISK_SCORE, r.RISK_TIER, r.RISK_FACTORS_SUMMARY
            FROM {RAW}.PATIENTS p
            LEFT JOIN {ANA}.PATIENT_RISK_SCORES r ON p.PATIENT_ID = r.PATIENT_ID
            WHERE p.PATIENT_ID = '{selected_pid}'
        """)
        if not demo.empty:
            d = demo.iloc[0]
            risk_tier = d.get("RISK_TIER", "Unknown")
            risk_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk_tier, "risk-low")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Name", f"{d['FIRST_NAME']} {d['LAST_NAME']}")
            col2.metric("Age / Gender", f"{pd.Timestamp.now().year - d['DATE_OF_BIRTH'].year} / {d['GENDER']}")
            col3.metric("Insurance", d["INSURANCE_TYPE"])
            col4.metric("Risk Score", f"{int(d.get('RISK_SCORE', 0))} / 100", delta=risk_tier, delta_color="inverse" if risk_tier == "High" else "normal")

            if d.get("RISK_FACTORS_SUMMARY"):
                st.markdown(f'<div class="{risk_class}"><strong>Risk Factors:</strong> {d["RISK_FACTORS_SUMMARY"]}</div>', unsafe_allow_html=True)

        # Sub-sections
        sec1, sec2 = st.columns(2)

        with sec1:
            st.subheader("Active Conditions")
            conds = run_query(f"""
                SELECT ICD10_CODE, DESCRIPTION, ONSET_DATE, STATUS, SEVERITY
                FROM {RAW}.CONDITIONS WHERE PATIENT_ID = '{selected_pid}' AND STATUS = 'Active'
                ORDER BY ONSET_DATE DESC
            """)
            if conds.empty:
                st.info("No active conditions on record.")
            else:
                st.dataframe(conds)

        with sec2:
            st.subheader("Current Medications")
            meds = run_query(f"""
                SELECT DRUG_NAME, DOSAGE, FREQUENCY, PRESCRIBER, START_DATE
                FROM {RAW}.MEDICATIONS WHERE PATIENT_ID = '{selected_pid}' AND STATUS = 'Active'
                ORDER BY START_DATE DESC
            """)
            if meds.empty:
                st.info("No active medications on record.")
            else:
                st.dataframe(meds)

        # Encounter timeline
        st.subheader("Encounter Timeline")
        encounters = run_query(f"""
            SELECT ENCOUNTER_DATE, ENCOUNTER_TYPE, DEPARTMENT, PROVIDER_NAME, CHIEF_COMPLAINT, DISPOSITION
            FROM {RAW}.ENCOUNTERS WHERE PATIENT_ID = '{selected_pid}'
            ORDER BY ENCOUNTER_DATE DESC
        """)
        if encounters.empty:
            st.info("No encounters on record.")
        else:
            enc_chart = encounters.copy()
            enc_chart["ENCOUNTER_DATE"] = pd.to_datetime(enc_chart["ENCOUNTER_DATE"])
            color_map = {
                "Office Visit": "#4CAF50", "Emergency": "#E53935", "Inpatient": "#FF9800",
                "Telehealth": "#2196F3", "Lab Only": "#9C27B0", "Urgent Care": "#FF5722",
            }
            enc_scatter = alt.Chart(enc_chart).mark_circle(size=120).encode(
                x=alt.X("ENCOUNTER_DATE:T", title="Date"),
                y=alt.Y("ENCOUNTER_TYPE:N", title=""),
                color=alt.Color("ENCOUNTER_TYPE:N", scale=alt.Scale(
                    domain=list(color_map.keys()), range=list(color_map.values())
                ), legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["ENCOUNTER_DATE:T", "ENCOUNTER_TYPE:N", "DEPARTMENT:N", "CHIEF_COMPLAINT:N", "DISPOSITION:N"],
            ).properties(height=280)
            st.altair_chart(enc_scatter, use_container_width=True)
            with st.expander("View encounter details"):
                st.dataframe(encounters)

        # Labs & Claims side by side
        lab_col, claim_col = st.columns(2)
        with lab_col:
            st.subheader("Recent Lab Results")
            labs = run_query(f"""
                SELECT TEST_NAME, RESULT_VALUE, UNIT, REFERENCE_LOW, REFERENCE_HIGH, IS_ABNORMAL, COLLECTED_DATE
                FROM {RAW}.LAB_RESULTS WHERE PATIENT_ID = '{selected_pid}'
                ORDER BY COLLECTED_DATE DESC LIMIT 20
            """)
            if labs.empty:
                st.info("No lab results on record.")
            else:
                st.dataframe(labs)

        with claim_col:
            st.subheader("Claims Summary")
            claims_summary = run_query(f"""
                SELECT STATUS, COUNT(*) AS CLAIM_COUNT, SUM(AMOUNT_BILLED) AS TOTAL_BILLED, SUM(AMOUNT_PAID) AS TOTAL_PAID
                FROM {RAW}.CLAIMS WHERE PATIENT_ID = '{selected_pid}'
                GROUP BY STATUS ORDER BY CLAIM_COUNT DESC
            """)
            if claims_summary.empty:
                st.info("No claims on record.")
            else:
                st.dataframe(claims_summary)
                total_billed = claims_summary["TOTAL_BILLED"].sum()
                total_paid = claims_summary["TOTAL_PAID"].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Billed", f"${total_billed:,.0f}")
                c2.metric("Total Paid", f"${total_paid:,.0f}")
                c3.metric("Outstanding", f"${total_billed - total_paid:,.0f}")


# ========================== TAB 2: RISK DASHBOARD ==========================
with tab2:
    risk_df = run_query(f"""
        SELECT * FROM {ANA}.PATIENT_RISK_SCORES ORDER BY RISK_SCORE DESC
    """)

    if risk_df.empty:
        st.warning("No risk score data available.")
    else:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        high_count = len(risk_df[risk_df["RISK_TIER"] == "High"])
        med_count = len(risk_df[risk_df["RISK_TIER"] == "Medium"])
        low_count = len(risk_df[risk_df["RISK_TIER"] == "Low"])
        avg_score = risk_df["RISK_SCORE"].mean()
        k1.metric("High Risk Patients", high_count)
        k2.metric("Medium Risk", med_count)
        k3.metric("Low Risk", low_count)
        k4.metric("Avg Risk Score", f"{avg_score:.0f}")

        # Charts row
        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Risk Score Distribution")
            hist_chart = alt.Chart(risk_df).mark_bar(color="#0068C9").encode(
                x=alt.X("RISK_SCORE:Q", bin=alt.Bin(maxbins=20), title="Risk Score"),
                y=alt.Y("count()", title="Patient Count"),
            ).properties(height=320)
            st.altair_chart(hist_chart, use_container_width=True)

        with ch2:
            st.subheader("Risk Tier by Insurance")
            tier_ins = risk_df.groupby(["INSURANCE_TYPE", "RISK_TIER"]).size().reset_index(name="COUNT")
            color_map_tier = {"High": "#E53935", "Medium": "#FFC107", "Low": "#43A047"}
            tier_chart = alt.Chart(tier_ins).mark_bar().encode(
                x=alt.X("INSURANCE_TYPE:N", title="Insurance Type"),
                y=alt.Y("COUNT:Q", title="Patient Count", stack=True),
                color=alt.Color("RISK_TIER:N", scale=alt.Scale(
                    domain=list(color_map_tier.keys()), range=list(color_map_tier.values())
                )),
            ).properties(height=320)
            st.altair_chart(tier_chart, use_container_width=True)

        # Top risk patients table
        st.subheader("Highest Risk Patients")
        top_risk = risk_df.head(15)[["PATIENT_ID", "PATIENT_NAME", "AGE", "GENDER", "INSURANCE_TYPE",
                                      "RISK_SCORE", "RISK_TIER", "CHRONIC_CONDITION_COUNT",
                                      "ED_VISITS_LAST_6MO", "ACTIVE_MEDICATION_COUNT",
                                      "ABNORMAL_LABS_LAST_6MO", "RISK_FACTORS_SUMMARY"]]
        st.dataframe(top_risk)

        # Risk factor breakdown
        st.subheader("Risk Factor Analysis")
        st.caption("How each factor contributes across the patient population")
        factor_data = pd.DataFrame({
            "Factor": ["Chronic Conditions", "ED Visits (6mo)", "Inpatient Admits (12mo)",
                       "Polypharmacy (5+ meds)", "Abnormal Labs (6mo)", "Age 65+"],
            "Affected Patients": [
                len(risk_df[risk_df["CHRONIC_CONDITION_COUNT"] >= 3]),
                len(risk_df[risk_df["ED_VISITS_LAST_6MO"] >= 1]),
                len(risk_df[risk_df["INPATIENT_ADMITS_LAST_12MO"] >= 1]),
                len(risk_df[risk_df["ACTIVE_MEDICATION_COUNT"] >= 5]),
                len(risk_df[risk_df["ABNORMAL_LABS_LAST_6MO"] >= 2]),
                len(risk_df[risk_df["AGE"] >= 65]),
            ]
        })
        factor_chart = alt.Chart(factor_data).mark_bar(color="#0068C9").encode(
            x=alt.X("Factor:N", sort=None, title=""),
            y=alt.Y("Affected Patients:Q", title="Number of Patients"),
        ).properties(height=300)
        st.altair_chart(factor_chart, use_container_width=True)


# ========================== TAB 3: COPILOT Q&A =============================
with tab3:
    st.subheader("Ask the Healthcare Copilot")
    st.caption("Ask clinical, regulatory, or patient-specific questions. Answers are grounded in clinical documents with cited sources.")

    example_questions = [
        "What are the FDA requirements for prescribing opioid analgesics?",
        "What is the sepsis management protocol and antibiotic selection?",
        "What are the monitoring requirements for SGLT2 inhibitors in CKD patients?",
        "What are the HIPAA requirements for patient data de-identification?",
        "What are the high-risk drug interactions with Warfarin?",
        "What is the DKA management protocol for insulin and potassium?",
    ]

    with st.expander("Example questions you can ask", expanded=False):
        for q in example_questions:
            if st.button(q, key=f"ex_{hash(q)}"):
                st.session_state["copilot_question"] = q

    # Optional patient context filter
    with st.sidebar:
        st.header("Search Filters")
        filter_patient = st.checkbox("Filter to specific patient", value=False)
        filter_pid = None
        if filter_patient:
            p_opts = {row["PATIENT_ID"]: f"{row['NAME']} ({row['PATIENT_ID']})" for _, row in patients_df.iterrows()}
            filter_pid = st.selectbox("Patient", p_opts.keys(), format_func=lambda x: p_opts[x], key="qa_patient_filter")

        filter_doc_type = st.multiselect("Document Types", ["clinical_note", "discharge_summary", "drug_safety_bulletin", "regulatory_guideline", "care_protocol"], default=[])
        num_sources = st.slider("Number of sources to retrieve", 3, 10, 5)

    question = st.text_area("Your question:", value=st.session_state.get("copilot_question", ""), height=80, key="qa_input", placeholder="e.g., What are the monitoring requirements for patients on anticoagulation therapy?")

    if st.button("Ask Copilot"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching clinical knowledge base..."):
                # Build filter
                search_filter = None
                filter_conditions = {}
                if filter_pid:
                    filter_conditions["@or"] = [
                        {"@eq": {"PATIENT_ID": filter_pid}},
                        {"@eq": {"PATIENT_ID": ""}}
                    ]
                if filter_doc_type:
                    filter_conditions["@in"] = {"DOC_TYPE": filter_doc_type}
                if filter_conditions:
                    if len(filter_conditions) > 1:
                        search_filter = {"@and": [{k: v} for k, v in filter_conditions.items()]}
                    else:
                        search_filter = filter_conditions

                docs = search_documents(question, limit=num_sources, filter_obj=search_filter)

            if not docs:
                st.warning("No relevant documents found. Try broadening your search filters.")
            else:
                with st.spinner("Generating evidence-based answer..."):
                    answer = ask_copilot(question, docs)

                st.markdown("### Answer")
                st.markdown(answer)

                st.markdown("---")
                st.markdown("### Source Documents")
                for i, doc in enumerate(docs, 1):
                    doc_type_label = doc.get("DOC_TYPE", "unknown").replace("_", " ").title()
                    patient_label = f" | Patient: {doc['PATIENT_ID']}" if doc.get("PATIENT_ID") else ""
                    with st.expander(f"[{i}] {doc.get('TITLE', 'Untitled')} ({doc_type_label}{patient_label})"):
                        st.caption(f"Author: {doc.get('AUTHOR', 'Unknown')} | Date: {doc.get('CREATED_DATE', 'N/A')}")
                        st.text(doc.get("CONTENT", "No content available.")[:4000])
