import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG (MATCH MAIN)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Cancer Screening Risk Assessment",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# GLOBAL STYLES (IDENTICAL)
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
    color: #111827;
}

h1, h2, h3, {
    color: #0F172A !important;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 800 !important;
    color: #003366 !important;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

.card {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid #E5E7EB;
    color: #111827 !important;
}

.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF !important;
    border-radius: 8px;
}
/* --- HIDE STREAMLIT MULTIPAGE SIDEBAR COMPLETELY --- */
section[data-testid="stSidebar"] {
    display: none !important;
}

/* Remove left padding reserved for sidebar */
.main, .block-container {
    padding-left: 2rem !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER (IDENTICAL)
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("PhaseIVAIHorizontal.png", width=180)
with col2:
    st.markdown(
        "<h1>Cancer Screening Risk Assessment</h1>"
        "<p style='color:#9CA3AF;'>Decision-support tool • Research prototype</p>",
        unsafe_allow_html=True
    )
with col3:
    st.image("NTUvertical.png", width=180)

st.divider()

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
if st.button("Home Page"):
    st.switch_page("UI_9.py")

# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------
st.markdown(
    """
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.25rem;
    ">
        Data Used in the System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    This section describes the data sources, feature schema, and preprocessing
    pipeline used across all models in the system. The same underlying clinical
    dataset and feature engineering strategy are shared by the federated learning
    and centralised training paradigms, ensuring methodological consistency and
    fair comparison across models.
    """
)

# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Dataset Overview</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    The system is trained using structured, tabular clinical data derived from
    electronic health records.

    **Key characteristics**
    - Unit of analysis: Individual patients  
    - Task: Binary classification (cancer screening risk)  
    - Target variable (`label`)  
        - `0` → lower screening risk  
        - `1` → higher screening risk  

    The dataset combines demographic attributes, documented clinical conditions,
    and engineered severity indicators to reflect real-world screening data.
    """
)

# ------------------------------------------------------------
# FEATURE SCHEMA
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Feature Schema</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    After preprocessing and feature engineering, each patient record is represented
    using **25 numerical features**, grouped as follows.
    """
)

st.markdown(
    """
    **Demographics**
    - Age (ordinally encoded from age bands)
    - Gender (binary encoded)

    **Smoking-related features**
    - Daily smoker flag  
    - Smoking severity score (engineered)

    **Renal health**
    - Chronic Kidney Disease (CKD) stage

    **Respiratory conditions**
    - COPD  
    - Emphysema  
    - Chronic cough  
    - Wheezing  
    - Shortness of breath  
    - Respiratory distress  
    - Composite respiratory severity score (engineered)

    **Diabetes-related complications**
    - Microalbuminuria  
    - Retinopathy (NP and P)  
    - Macular edema  
    - Renal disease  
    - Proteinuria  
    - Diabetes complication count (engineered)

    **Cardiovascular history**
    - Hypertension  
    - Stroke  
    - Ischemic heart disease  
    - Acute myocardial infarction  
    - MI history  
    - MI severity score (engineered)
    """
)

st.markdown(
    """
    Several features are intentionally engineered to capture cumulative disease
    burden and clinical severity rather than relying solely on raw binary indicators.
    """
)

# ------------------------------------------------------------
# PREPROCESSING PIPELINE
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Preprocessing Pipeline</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Both federated and centralised models use the same preprocessing pipeline.

    **Preprocessing steps**
    - Categorical attributes mapped to numeric representations  
    - Binary clinical indicators encoded as 0/1  
    - Ordinal encoding applied to age bands  
    - Construction of severity and count-based features  
    - Feature standardisation using `StandardScaler`  

    The fitted scaler is reused during inference to preserve feature alignment
    between training and deployment.
    """
)

# ------------------------------------------------------------
# FEDERATED LEARNING DATA USAGE
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Federated Learning Data Usage</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    For federated learning models, the dataset is partitioned across multiple
    clients, simulating independent healthcare institutions.

    **Key points**
    - Each client contains patient-level records only  
    - No raw data are shared between clients  
    - Feature schema and preprocessing are identical across all clients  
    - Differences arise only from how data are distributed across clients  

    Client-level data distribution characteristics are described on the
    **Models** page and are not repeated here.
    """
)

# ------------------------------------------------------------
# CENTRALISED TRAINING DATA USAGE
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Centralised Training Data Usage</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    For the centralised model, all available data are aggregated into a single
    dataset prior to training.

    **Key points**
    - No client boundaries are preserved  
    - A single, globally consistent data distribution is used  
    - Identical features and preprocessing pipeline are applied  
    - Serves as a performance reference under ideal data-sharing conditions  

    This model represents an upper bound on achievable performance when privacy
    and decentralisation constraints are removed.
    """
)

# ------------------------------------------------------------
# SYNTHETIC DATA AUGMENTATION
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Synthetic Data Augmentation</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    To address class imbalance and improve model robustness, synthetic patient
    records were incorporated during training.

    **Purpose**
    - Improve representation of minority class  
    - Stabilise training under class imbalance  
    - Support federated learning where local datasets may be limited  

    **Validation**
    - Distributional similarity checks  
    - Correlation structure preservation  
    - Membership inference risk assessment  
    - Nearest-neighbour distance analysis  

    Only validated synthetic samples were retained for model training.
    """
)

# ------------------------------------------------------------
# ETHICS & PRIVACY
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Ethical and Privacy Considerations</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    - No patient-identifiable information is used  
    - Federated learning prevents raw data sharing  
    - Synthetic data further reduces re-identification risk  

    The data handling strategy aligns with privacy-preserving machine learning
    practices suitable for healthcare research prototyping.
    """
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">Summary</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    In summary, all models in the system:
    - Use the same clinically grounded feature set  
    - Share an identical preprocessing pipeline  
    - Differ only in how training data are distributed and accessed  

    This design ensures that observed performance differences reflect training
    strategy rather than differences in data representation.
    """
)
# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.image("NTUhorizontal.png", width=180)
st.markdown("Phase IV AI Project • Nottingham Trent University")

