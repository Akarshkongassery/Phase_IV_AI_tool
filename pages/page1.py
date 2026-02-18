import streamlit as st
import matplotlib.pyplot as plt  

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
# MODELS USED – CONTENT
# ------------------------------------------------------------

# Overview


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
        Models Used in the system
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """


    This application supports **two federated learning (FL) models** trained across 5 clients with
    different client data distribution assumptions.   Both models share the same neural network architecture, optimisation strategy,
    and feature engineering pipeline. The **only difference** between them lies in how training data is distributed
    across participating clients.
    """
)

# Model 1 – IID

st.markdown(
    """
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.25rem;
    ">
        Model 1: Federated Learning (IID)
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    In the IID (Independent and Identically Distributed) setting, each client
    receives a statistically similar subset of the global dataset.

    **Key characteristics**
    - Comparable sample sizes across clients  
    - Similar class distributions per client  
    - Lower data heterogeneity  
    - Commonly used as a baseline in federated learning research  

    The visualisations below illustrate client-level data distributions
    under the IID training assumption.
    """
)


# ------------------------------------------------------------
# IID CLIENT DISTRIBUTION – PIE CHARTS (HARDCODED)
# ------------------------------------------------------------

st.markdown("#### Client-level class distribution (IID)")

# Hardcoded client statistics
iid_client_data = {
    "Client 0": {"label_0": 2934, "label_1": 944},
    "Client 1": {"label_0": 2919, "label_1": 959},
    "Client 2": {"label_0": 2935, "label_1": 943},
    "Client 3": {"label_0": 2913, "label_1": 965},
    "Client 4": {"label_0": 2957, "label_1": 920},
}

cols = st.columns(5)

for col, (client, counts) in zip(cols, iid_client_data.items()):
    with col:
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.pie(
            [counts["label_0"], counts["label_1"]],
            labels=["Label 0", "Label 1"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#93C5FD", "#FCA5A5"],
            textprops={"fontsize": 8}
        )
        ax.set_title(client, fontsize=9)
        ax.axis("equal")
        st.pyplot(fig)


# Model 2 – non-IID
# st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.25rem;
    ">
        Model 2: Federated Learning (non-IID)
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """


    In the non-IID setting, client datasets are heterogeneous, reflecting
    real-world clinical data distributions where data availability and
    class prevalence vary across institutions.

    **Key characteristics**
    - Uneven class distributions across clients  
    - Variable sample sizes  
    - Client-specific data biases  
    - More realistic but more challenging training scenario  

    The visualisations below demonstrate how data heterogeneity manifests
    across participating clients.
    """
)
# st.markdown("</div>", unsafe_allow_html=True)

st.markdown("#### Client-level class distribution (non-IID)")

# Hardcoded client statistics

iid_client_data = {
    "Client 0": {"label_0": 3989, "label_1": 1006},
    "Client 1": {"label_0": 5894, "label_1": 1093},
    "Client 2": {"label_0": 1121, "label_1": 2350},
    "Client 3": {"label_0": 8, "label_1": 248},
    "Client 4": {"label_0": 3646, "label_1": 33},
}



cols = st.columns(5)

for col, (client, counts) in zip(cols, iid_client_data.items()):
    with col:
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.pie(
            [counts["label_0"], counts["label_1"]],
            labels=["Label 0", "Label 1"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#93C5FD", "#FCA5A5"],
            textprops={"fontsize": 8}
        )
        ax.set_title(client, fontsize=9)
        ax.axis("equal")
        st.pyplot(fig)


## Model 3 centralised


# st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.25rem;
    ">
        Model 3: Centralised Model 
    </div>
    """,
    unsafe_allow_html=True
)




# Hardcoded client statistics

st.markdown(
    """
In the centralised learning setting, all available data are aggregated into a single dataset and used to train a global model without client-level separation.

Key characteristics

- Unified dataset without client boundaries  
- Globally consistent class distribution  
- No inter-client heterogeneity  
- Simplified training and optimisation process  

This model serves as a performance reference, representing an upper bound under ideal data-sharing conditions.
"""
)


# ------------------------------------------------------------
# FOOTER (IDENTICAL)
# ------------------------------------------------------------
st.divider()
st.image("NTUhorizontal.png", width=180)
st.markdown("Phase IV AI Project • Nottingham Trent University")
