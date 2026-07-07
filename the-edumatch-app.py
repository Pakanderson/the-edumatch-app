import streamlit as st
import pandas as pd
import joblib

# ==============================================================================
# 1. INITIALIZATION & SECRETS CONFIGURATION
# ==============================================================================
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Missing 'OPENAI_API_KEY' in Streamlit Secrets!")
    st.stop()


# Simulating your mock RAG Knowledge Engine placeholder
class MockRagEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def retrieve_legal_context(self, profile):
        # A simple fallback system to ensure your RAG layout doesn't crash
        if profile["Fee_Arrears"] == 1:
            return [
                {
                    "clause_id": "§ 12 Abs. 3 HochSchG",
                    "text": "Exmatriculation due to default on semester fee payments after warnings.",
                }
            ]
        elif profile["ECTS_Earned_Sem1"] + profile["ECTS_Earned_Sem2"] < 30:
            return [
                {
                    "clause_id": "§ 45 MPO",
                    "text": "Compulsory academic advisory session required due to credit deficit below baseline benchmarks.",
                }
            ]
        return []


@st.cache_resource
def load_analytics_assets():
    try:
        model = joblib.load("models/german_retention_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        kmeans = joblib.load("models/kmeans.pkl")
        x_train_columns = joblib.load("models/x_train_columns.pkl")
    except FileNotFoundError:
        # Fallback objects if models are missing during test phases
        st.warning(
            "⚠️ Warning: Pre-trained machine learning asset binaries not found under /models directory. Initializing testing fallbacks..."
        )
        model = joblib.load("models/german_retention_model.pkl") if False else None
        scaler = None
        kmeans = None
        x_train_columns = None

    return model, scaler, kmeans, x_train_columns


# Unpacking the global cached assets cleanly (FIXED COMPONENT)
german_retention_model, scaler, kmeans, X_train_cols = load_analytics_assets()

# Instantiate the engine globally (FIXED COMPONENT)
rag_engine = MockRagEngine(api_key=openai_api_key)

# ==============================================================================
# 2. STREAMLIT USER INTERFACE LAYOUT
# ==============================================================================
st.title("EduMatch: Predictive Retention Pipeline & Advisor Dashboard")
st.markdown(
    "Input a student's administrative, economic, and academic performance vectors "
    "to calculate empirical exmatriculation risks, map intervention segments, and extract "
    "matched legal regulatory provisions."
)

with st.sidebar:
    st.header("Student Parameters")
    age = st.slider("Age at Enrollment", 17, 60, 22, step=1)
    gender = st.radio("Gender Identity", ["Female", "Male"])
    displaced = st.radio("Relocated Student (Displaced Status)", ["Yes", "No"])
    bafoeg = st.radio("Federal Financial Aid Status (BAföG)", ["Yes (Recipient)", "No"])
    arrears = st.radio("Semester Tuition Fee Arrears", ["Yes (Delinquent)", "No"])

    st.subheader("Academic Milestones")
    ects_s1 = st.slider("Earned ECTS (1st Semester)", 0, 40, 30, step=5)
    grade_s1 = st.slider(
        "Grade Average Point (1st Semester)", 0.0, 20.0, 12.0, step=0.1
    )
    ects_s2 = st.slider("Earned ECTS (2nd Semester)", 0, 40, 25, step=5)
    grade_s2 = st.slider(
        "Grade Average Point (2nd Semester)", 0.0, 20.0, 11.5, step=0.1
    )


# ==============================================================================
# 3. PREDICTION & PIPELINE LOGIC
# ==============================================================================
def run_prediction():
    input_data = pd.DataFrame(
        [
            {
                "Grade_Avg_Sem1": grade_s1,
                "Grade_Avg_Sem2": grade_s2,
                "BAfoeg_Status": 1 if bafoeg == "Yes (Recipient)" else 0,
                "Fee_Arrears": 1 if arrears == "Yes (Delinquent)" else 0,
                "Displaced_Status": 1 if displaced == "Yes" else 0,
                "Gender": 1 if gender == "Male" else 0,
                "Age_At_Enrollment": age,
                "ECTS_Earned_Sem1": ects_s1,
                "ECTS_Earned_Sem2": ects_s2,
            }
        ]
    )

    # Calculate fallback values if binaries are missing in repository
    if german_retention_model is not None:
        if hasattr(german_retention_model, "predict_proba"):
            risk_prob = german_retention_model.predict_proba(input_data)[0][1]
        else:
            risk_prob = float(german_retention_model.predict(input_data)[0])
    else:
        # Mock calculation logic mirroring real model outputs for testing
        base_risk = 0.15
        if arrears == "Yes (Delinquent)":
            base_risk += 0.35
        if (ects_s1 + ects_s2) < 40:
            base_risk += 0.40
        risk_prob = min(base_risk, 0.98)

    if scaler is not None and kmeans is not None:
        scaled_input = scaler.transform(input_data)
        cluster = kmeans.predict(scaled_input)[0]
    else:
        if arrears == "Yes (Delinquent)":
            cluster = 2
        elif (ects_s1 + ects_s2) < 35:
            cluster = 1
        else:
            cluster = 0

    return risk_prob, cluster, input_data.iloc[0].to_dict()


# ==============================================================================
# 4. CONDITIONAL RENDER (TRIGGERED ON CLICK)
# ==============================================================================
if st.button("Calculate Risk & Match Regulations"):
    risk_prob, cluster_id, profile = run_prediction()

    st.divider()

    st.header("Predictive Risk Indicators")
    status = "CRITICAL RISK" if risk_prob > 0.5 else "STABLE STATUS"

    if status == "CRITICAL RISK":
        st.error(f"Operational Status: {status}")
    else:
        st.success(f"Operational Status: {status}")

    st.metric(label="Calculated Exmatriculation Probability", value=f"{risk_prob:.2%}")

    st.header("Assigned Intervention Cohort")
    cluster_mapping = {
        0: "Cluster 0: Stabilized Academic Deficit (Financial Aid Cushioned)",
        1: "Cluster 1: Acute Academic Collapse (Severe Credit Deficit / Mature Students)",
        2: "Cluster 2: Latent Socioeconomic Risk (High Grades, Unpaid Semester Fees)",
    }
    st.info(cluster_mapping.get(cluster_id, "Unknown Cluster Profile"))

    st.header("RAG: Contextualized Examination Regulations (Prüfungsordnung)")
    docs = rag_engine.retrieve_legal_context(profile)

    if docs:
        for idx, doc in enumerate(docs, 1):
            with st.expander(f"[{idx}] {doc['clause_id']}", expanded=True):
                st.markdown(f"**Regulatory Context:** {doc['text']}")
    else:
        st.write(
            "Clear Standing: No critical regulatory violations or credit deficits identified."
        )
