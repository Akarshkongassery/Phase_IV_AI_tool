# ============================================================
#  CANCER RISK SCREENING – NTU-FAITHFUL STREAMLIT UI
#  (COSMETICS UNCHANGED • FEATURE LOGIC ALIGNED)
# ============================================================

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
# BUNDLE_PATH = "models/cancer_risk_model_bundle.pkl"
# BUNDLE_PATH = "models/cancer_risk_model_v2.pkl"
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Cancer Screening Risk Assessment",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# GLOBAL STYLES (UNCHANGED)
# ------------------------------------------------------------
st.markdown("""
<style>
/* 1. BASE APP & BACKGROUND */
.stApp {
    background-color: #F8FAFC;
    color: #111827;
}

/* 2. TITLES & HEADERS */
h1, h2, h3, {
    color: #0F172A !important;
}
.section-header {
    font-size: 1.25rem;
    font-weight: 800 !important; /* Extra Bold */
    color: #003366 !important;    /* Dark Blue */
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* 3. CARD STYLING */
.card {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: .05rem;
    margin-bottom: 1rem;
    border: 1px solid #E5E7EB;
    color: #111827 !important;
}



/* 4. RESULT & METRIC VISIBILITY (THE FIX) */
[data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetricLabel"] p {
    color: #6B7280 !important;
}

/* Fix for the "Routine screening recommended" boxes */
.stAlert {
    background-color: #F0FDF4 !important; /* Light green bg */
    color: #166534 !important;            /* Dark green text */
    border: 1px solid #BBF7D0 !important;
}

/* Fix for the Warning boxes if risk is high */
[data-testid="stNotificationContentWarning"] {
    background-color: #FFF7ED !important;
    color: #9A3412 !important;
}

/* 5. DROPDOWN FIX */
div[data-baseweb="popover"] {
    background-color: white !important;
}

div[role="option"] {
    background-color: white !important;
    color: #111827 !important;
}

div[role="option"]:hover {
    background-color: #F1F5F9 !important;
    color: #2563EB !important;
}

div[role="option"][aria-selected="true"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* 6. WIDGET LABELS & CHECKBOXES */
label[data-testid="stWidgetLabel"] p, .stCheckbox label p {
    color: #111827 !important;
}

/* 7. BUTTONS */
.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF !important;
    border-radius: 8px;
}
.stButton > button {
    color: #FFFFFF !important;
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
# HEADER
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("PhaseIVAIHorizontal.png", width=180)
with col2:
    st.markdown("<h1>Cancer Screening Risk Assessment</h1>"
                "<p style='color:#9CA3AF;'>Decision-support tool • Research prototype</p>",
                unsafe_allow_html=True)
with col3:
    st.image("NTUvertical.png", width=180)

st.divider()


col1, col3 = st.columns(2)

with col1:
    if st.button("Models Used", key="models_used_btn", use_container_width=True):
        st.switch_page("pages/page1.py")



with col3:
    if st.button("Data", key="data_btn", use_container_width=True):
        st.switch_page("pages/page2.py")


# st.markdown("<div class='card'><div class='section-header'>Select risk model</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        color: #EF4444;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.25rem;
    ">
        Select risk model
    </div>
    """,
    unsafe_allow_html=True
)


model_choice = st.selectbox(
    "",
    [
        "Model Trained under FL-IID Schema",
        "Model Trained under FL-non IID Schema",
        "Centralised Model"
    ]
)


# st.markdown(
#     "<p style='font-size:0.9rem; font-weight:600; color:#374151;'>Model selection</p>",
#     unsafe_allow_html=True
# )

# model_choice = st.selectbox(
#     "",
#     ["Baseline model", "Enhanced model"],
#     label_visibility="collapsed"
# )


st.markdown("</div>", unsafe_allow_html=True)






# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
class FLNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# @st.cache_resource
# def load_bundle():
#     bundle = joblib.load(BUNDLE_PATH)
#     model = FLNet(len(bundle["feature_order"]))
#     model.load_state_dict(bundle["model_state_dict"])
#     model.eval()
#     return model, bundle["scaler"], bundle["calibrator"], bundle["feature_order"], bundle["threshold"]
# model, scaler, calibrator, feature_order, threshold = load_bundle()

@st.cache_resource
def load_models():
    bundle_v1 = joblib.load("models/cancer_risk_model_v3_IID.pkl")
    bundle_v2 = joblib.load("models/cancer_risk_model_v3_nonIID.pkl")
    bundle_v3 = joblib.load("models/cancer_risk_model_v3_centralised.pkl")

    model_v1 = FLNet(len(bundle_v1["feature_order"]))
    model_v1.load_state_dict(bundle_v1["model_state_dict"])
    model_v1.eval()

    model_v2 = FLNet(len(bundle_v2["feature_order"]))
    model_v2.load_state_dict(bundle_v2["model_state_dict"])
    model_v2.eval()

    model_v3 = FLNet(len(bundle_v3["feature_order"]))
    model_v3.load_state_dict(bundle_v3["model_state_dict"])
    model_v3.eval()

    return {
        "fl_iid": {
            "model": model_v1,
            "scaler": bundle_v1["scaler"],
            "calibrator": bundle_v1["calibrator"],
            "feature_order": bundle_v1["feature_order"],
            "threshold": bundle_v1["threshold"],
        },
        "fl_noniid": {
            "model": model_v2,
            "scaler": bundle_v2["scaler"],
            "calibrator": bundle_v2["calibrator"],
            "feature_order": bundle_v2["feature_order"],
            "threshold": bundle_v2["threshold"],
        },
        "centralised": {
            "model": model_v3,
            "scaler": bundle_v3["scaler"],
            "calibrator": bundle_v3["calibrator"],
            "feature_order": bundle_v3["feature_order"],
            "threshold": bundle_v3["threshold"],
        }
    }




models = load_models()




# ------------------------------------------------------------
# INPUT UI (UNCHANGED VISUALLY)
# ------------------------------------------------------------
# st.markdown("<div class='card'><div class='section-header'>Demographics</div>", unsafe_allow_html=True)
# st.markdown("<div class='card'><div class='section-header'>Input User Data</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        color: #EF4444;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.25rem;
    ">
        Input User Data
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)

age_ui = st.selectbox("Age group", ["<50", "50–70", ">70"])
gender_ui = st.selectbox("Gender", ["Male", "Female"])
st.markdown("</div>", unsafe_allow_html=True)

# st.markdown("<div class='card'><div class='section-header'>Smoking Status</div>", unsafe_allow_html=True)
# smoking_status_ui = st.selectbox("Smoking status", ["Never", "Former", "Current smoker", "Unknown"])
# daily_smoker = st.checkbox("Daily smoker")

# st.markdown("<div class='card'><div class='section-header'>Smoking Status</div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Smoking Status</div>", unsafe_allow_html=True)

smoking_status_ui = st.selectbox(
    " ",
    ["Never", "Former", "Current smoker", "Unknown"]
)

daily_smoker = st.checkbox(
    "Daily smoker",
    disabled=(smoking_status_ui == "Never")
)

st.markdown("</div>", unsafe_allow_html=True)



# st.markdown("<div class='card'><div class='section-header'>Kidney Health</div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Kidney Health</div>", unsafe_allow_html=True)
ckd_stage_ui = st.selectbox("CKD stage", ["None", "Stage 1", "Stage 2", "Stage 3"])
st.markdown("</div>", unsafe_allow_html=True)

# st.markdown("<div class='card'><div class='section-header'>Respiratory Conditions</div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Respiratory Conditions</div>", unsafe_allow_html=True)
COPD = st.checkbox("COPD")
emphysema = st.checkbox("Emphysema")
st.markdown("<div class='section-header'>Respiratory Conditions</div>", unsafe_allow_html=True)
resp_cough = st.checkbox("Chronic cough")
resp_wheezing = st.checkbox("Wheezing")
resp_dyspnea = st.checkbox("Shortness of breath")
resp_distress = st.checkbox("Respiratory distress")
st.markdown("</div>", unsafe_allow_html=True)

# st.markdown("<div class='card'><div class='section-header'>Diabetes Complications</div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Diabetes Complications</div>", unsafe_allow_html=True)
dm_microalbuminuria = st.checkbox("Microalbuminuria")
dm_retinopathy_np = st.checkbox("Retinopathy (NP)")
dm_retinopathy_p = st.checkbox("Retinopathy (P)")
dm_macular_edema = st.checkbox("Macular edema")
dm_renal_disease = st.checkbox("Renal disease")
dm_proteinuria = st.checkbox("Proteinuria")
st.markdown("</div>", unsafe_allow_html=True)

# st.markdown("<div class='card'><div class='section-header'>Cardiovascular History</div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Cardiovascular History</div>", unsafe_allow_html=True)
hypertension = st.checkbox("Hypertension")
stroke = st.checkbox("Stroke")
IHD = st.checkbox("Ischemic heart disease")
MI_acute = st.checkbox("Acute MI")
MI_history = st.checkbox("MI history")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# INTEGRATED GRADIENTS (UNCHANGED)
# ------------------------------------------------------------
def integrated_gradients(model, x, baseline=None, steps=50):
    model.eval()
    x_t = torch.tensor(x, dtype=torch.float32)

    baseline_t = torch.zeros_like(x_t) if baseline is None else torch.tensor(baseline, dtype=torch.float32)

    scaled_inputs = [
        baseline_t + (i / steps) * (x_t - baseline_t)
        for i in range(steps + 1)
    ]

    grads = []
    for s in scaled_inputs:
        s = s.unsqueeze(0)
        s.requires_grad_(True)
        model.zero_grad()
        out = torch.sigmoid(model(s))
        out.backward()
        grads.append(s.grad.detach().clone().squeeze(0))

    avg_grads = torch.stack(grads).mean(dim=0)
    ig = (x_t - baseline_t) * avg_grads
    return ig.numpy()

# ------------------------------------------------------------
# INFERENCE (NTU-FAITHFUL ENGINEERING)
# ------------------------------------------------------------

if st.button("Estimate Screening Risk", use_container_width=True):

    # SELECT ACTIVE MODEL BASED ON DROPDOWN

    if model_choice == "Federated Learning (IID)":
        active = models["fl_iid"]
    elif model_choice == "Federated Learning (Non-IID)":
        active = models["fl_noniid"]
    else:
        active = models["centralised"]


    model = active["model"]
    scaler = active["scaler"]
    calibrator = active["calibrator"]
    feature_order = active["feature_order"]
    threshold = active["threshold"]

    # 🔽 NOW PROCEED WITH EXISTING LOGIC
    age = 40 if age_ui == "<50" else 60 if age_ui == "50–70" else 80
    gender = 1 if gender_ui == "Male" else 0

    

    age = 40 if age_ui == "<50" else 60 if age_ui == "50–70" else 80
    gender = 1 if gender_ui == "Male" else 0

    smoking_status = smoking_status_ui.lower().replace(" ", "_")
    daily_smoker_flag = int(daily_smoker)

    if daily_smoker_flag == 1 and smoking_status != "never":
        smoking_severity = 3
    elif daily_smoker_flag == 1 and smoking_status == "never":
        smoking_severity = 2
    elif daily_smoker_flag == 0 and smoking_status != "never":
        smoking_severity = 1
    else:
        smoking_severity = 0

    CKD_stage = {"None": 0, "Stage 1": 1, "Stage 2": 2, "Stage 3": 3}[ckd_stage_ui]

    resp_severity = sum([resp_cough, resp_wheezing, resp_dyspnea, resp_distress])

    diabetes_complication_count = sum([
        dm_microalbuminuria, dm_retinopathy_np, dm_retinopathy_p,
        dm_macular_edema, dm_renal_disease, dm_proteinuria
    ])

    MI_severity = 2 if MI_acute else 1 if MI_history else 0

    features = {
        "age": age,
        "gender": gender,
        "smoking_severity": smoking_severity,
        "daily_smoker_flag": daily_smoker_flag,
        "CKD_stage": CKD_stage,
        "COPD": int(COPD),
        "emphysema": int(emphysema),
        "hypertension": int(hypertension),
        "stroke": int(stroke),
        "resp_cough": int(resp_cough),
        "resp_wheezing": int(resp_wheezing),
        "resp_dyspnea": int(resp_dyspnea),
        "resp_distress": int(resp_distress),
        "resp_severity": resp_severity,
        "dm_microalbuminuria": int(dm_microalbuminuria),
        "dm_retinopathy_np": int(dm_retinopathy_np),
        "dm_retinopathy_p": int(dm_retinopathy_p),
        "dm_macular_edema": int(dm_macular_edema),
        "dm_renal_disease": int(dm_renal_disease),
        "dm_proteinuria": int(dm_proteinuria),
        "diabetes_complication_count": diabetes_complication_count,
        "IHD": int(IHD),
        "MI_acute": int(MI_acute),
        "MI_history": int(MI_history),
        "MI_severity": MI_severity
    }

    x = np.array([[features[f] for f in feature_order]], dtype=float)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x_scaled, dtype=torch.float32))).item()

    calibrated_prob = calibrator.predict_proba([[prob]])[0, 1]

    # st.metric("Estimated cancer risk", f"{prob * 100:.1f}%")
    risk_label = "Mild risk" if calibrated_prob < threshold else "High risk"

    # st.metric(
    #     label="Cancer screening risk",
    #     value=risk_label
    # )

    risk_label = "Mild risk" if calibrated_prob < threshold else "High risk"
    risk_color = "#16A34A" if calibrated_prob < threshold else "#DC2626"  # green / red

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 12px;
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            text-align: center;
            margin-bottom: 1rem;
        ">
            <div style="
                font-size: 0.9rem;
                color: #6B7280;
                margin-bottom: 0.25rem;
            ">
                Cancer screening risk
            </div>
            <div style="
                font-size: 2.2rem;
                font-weight: 700;
                color: {risk_color};
            ">
                {risk_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.popover("ℹ️ Risk details"):
        st.markdown(
            f"""
            **Technical details**

            - Calibrated cancer risk estimate: **{calibrated_prob * 100:.1f}%**
          
            """
        )
          # - High-risk threshold: **{threshold * 100:.1f}%**
            

            # This value is used internally by the model but hidden from the main interface
            # to reduce misinterpretation by non-expert users.

    # st.success("Routine screening recommended" if calibrated_prob < threshold else "High screening priority")
    if calibrated_prob < threshold:
        st.success("Routine screening recommended")
    else:
        st.error("High screening priority")



    ig_vals = integrated_gradients(model, x_scaled[0])
    idx = np.argsort(np.abs(ig_vals))[::-1][:10]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(idx)), ig_vals[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_order[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Integrated Gradient attribution")
    ax.set_title("Why the model produced this score")

    st.subheader("Model explanation (Integrated Gradients)")
    st.pyplot(fig)

    with st.expander("Engineered feature vector sent to model"):
        st.json(features)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.image("NTUhorizontal.png", width=180)
st.markdown("Phase IV AI Project • Nottingham Trent University")

