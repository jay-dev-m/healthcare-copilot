# Healthcare Copilot

Patient 360, Risk Stratification, and Evidence-Based Q&A — built with Streamlit on Snowflake.

## Overview

Healthcare Copilot is a clinical decision-support app that unifies structured EHR data (patients, encounters, conditions, medications, labs, claims) with unstructured clinical documents (notes, discharge summaries, regulatory guidelines) into a single interface. Answers are grounded in source documents with numbered citations.

## Features

### Patient 360
- Patient search by name/ID
- Demographics with risk score at a glance
- Active conditions and current medications
- Interactive encounter timeline (Altair scatter chart by type)
- Recent lab results with abnormal flagging
- Claims summary with billed/paid/outstanding totals

### Risk Dashboard
- KPI cards: high/medium/low risk patient counts and average score
- Risk score distribution histogram
- Risk tier breakdown by insurance type (stacked bar)
- Top-15 highest risk patients table
- Risk factor analysis showing how many patients are affected by each factor (chronic conditions, ED visits, polypharmacy, abnormal labs, age 65+)

### Copilot Q&A
- Free-text clinical question input with example prompts
- Semantic search via **Cortex Search Service** over clinical documents
- Evidence-based answers via **Cortex COMPLETE** (llama3.1-70b) with numbered source citations
- Expandable source document excerpts showing author, date, and content
- Sidebar filters: patient scope, document type, number of sources

## Architecture

```
Streamlit App (warehouse runtime)
    |
    +-- Structured queries --> HEALTHCARE_COPILOT.RAW_DATA tables
    |                          (PATIENTS, ENCOUNTERS, CONDITIONS,
    |                           MEDICATIONS, LAB_RESULTS, CLAIMS)
    |
    +-- Risk view ----------> HEALTHCARE_COPILOT.ANALYTICS.PATIENT_RISK_SCORES
    |
    +-- Cortex Search ------> HEALTHCARE_COPILOT.ANALYTICS.CLINICAL_DOC_SEARCH
    |                          (semantic retrieval over CLINICAL_DOCUMENTS)
    |
    +-- Cortex COMPLETE ----> llama3.1-70b (cited answer generation)
```

## Data Model

**Database**: `HEALTHCARE_COPILOT`

| Schema | Object | Type | Description |
|--------|--------|------|-------------|
| RAW_DATA | PATIENTS | Table | 100 synthetic patients with demographics |
| RAW_DATA | ENCOUNTERS | Table | 800 encounters (office, ED, inpatient, telehealth) |
| RAW_DATA | CONDITIONS | Table | 500 conditions with ICD-10 codes |
| RAW_DATA | MEDICATIONS | Table | 600 medication records |
| RAW_DATA | LAB_RESULTS | Table | 900 lab results with reference ranges |
| RAW_DATA | CLAIMS | Table | 700 claims with billing amounts |
| RAW_DATA | CLINICAL_DOCUMENTS | Table | 25 clinical notes, discharge summaries, guidelines |
| ANALYTICS | PATIENT_RISK_SCORES | View | Composite risk score from multiple factors |
| ANALYTICS | CLINICAL_DOC_SEARCH | Cortex Search | Semantic search over clinical documents |

All data is fully synthetic.

## Deployment

The app is deployed to **Snowflake Streamlit-in-Snowflake (warehouse runtime)**.

### Prerequisites

- Snowflake account with `HEALTHCARE_COPILOT` database and all tables/views populated
- Cortex Search Service `CLINICAL_DOC_SEARCH` created and active
- `GRANT USE AI FUNCTIONS ON ACCOUNT TO ROLE <your_role>` for Cortex COMPLETE

### Deploy to Snowflake

The app uses warehouse runtime (no EAI required). Files are uploaded to a stage and the Streamlit object is created via DDL:

```sql
CREATE STAGE IF NOT EXISTS HEALTHCARE_COPILOT.ANALYTICS.HEALTHCARE_COPILOT_STAGE;

-- Upload files via snow CLI:
-- PUT 'file://streamlit_app.py' @HEALTHCARE_COPILOT.ANALYTICS.HEALTHCARE_COPILOT_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT 'file://environment.yml' @HEALTHCARE_COPILOT.ANALYTICS.HEALTHCARE_COPILOT_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT 'file://.streamlit/config.toml' @HEALTHCARE_COPILOT.ANALYTICS.HEALTHCARE_COPILOT_STAGE/.streamlit/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

CREATE STREAMLIT HEALTHCARE_COPILOT.ANALYTICS.HEALTHCARE_COPILOT_APP
  ROOT_LOCATION = '@HEALTHCARE_COPILOT.ANALYTICS.HEALTHCARE_COPILOT_STAGE'
  MAIN_FILE = 'streamlit_app.py'
  QUERY_WAREHOUSE = 'COMPUTE_WH';
```

### Run locally

```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=<your_connection> python3 -m streamlit run streamlit_app.py
```

## Project Structure

```
healthcare-copilot/
  streamlit_app.py       # Main app (3 tabs)
  environment.yml        # Snowflake Anaconda channel dependencies
  snowflake.yml          # Deployment manifest
  pyproject.toml         # Local dev dependencies
  .streamlit/
    config.toml          # Theme configuration
```

## Tech Stack

- **Streamlit** — UI framework (warehouse runtime, compatible with 1.22+)
- **Snowflake Cortex Search** — semantic document retrieval
- **Snowflake Cortex COMPLETE** — LLM answer generation (llama3.1-70b)
- **Altair** — chart visualizations
- **Snowpark Python** — Snowflake session and query execution
