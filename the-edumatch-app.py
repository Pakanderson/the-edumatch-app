import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
# 📦 STEP 1: INITIALIZE PRE-TRAINED MACHINE LEARNING ASSETS
# =====================================================================
@st.cache_resource
def load_analytics_assets():
    try:
        model = joblib.load("models/german_retention_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")

        # Extract features directly from the trained model to avoid missing X_train errors
        if hasattr(model, "feature_names_in_"):
            x_train_columns = list(model.feature_names_in_)
        else:
            # Fallback explicit column layout order matching baseline training vectors
            x_train_columns = [
                "Grade_Avg_Sem1",
                "Grade_Avg_Sem2",
                "BAfoeg_Status",
                "Fee_Arrears",
                "Displaced_Status",
                "Gender",
                "Age_At_Enrollment",
                "ECTS_Earned_Sem1",
                "ECTS_Earned_Sem2",
            ]
    except FileNotFoundError:
        # Graceful fallback objects if models are missing during test phases
        model, scaler, kmeans, x_train_columns = None, None, None, None

    return model, scaler, kmeans, x_train_columns


# Unpack models globally
german_retention_model, scaler, kmeans, X_train_cols = load_analytics_assets()


# =====================================================================
# 📚 STEP 2: INITIALIZE THE SEMANTIC VECTOR RAG ENGINE
# =====================================================================
class SemanticVectorRAG:
    """Upgraded RAG Engine using mathematical vector spaces and cosine similarity
    to search legal clauses based on descriptive risk keywords.
    """

    def __init__(self):
        self.documents = [
            {
                "id": "[PO-101]",
                "title": "Academic Progression Standard",
                "text": "Gemäß § 12 der Prüfungsordnung müssen Studierende bis zum Ende des zweiten Fachsemesters mindestens 30 ECTS-Punkte erbracht haben. Bei Unterschreitung dieser Schwelle von 15 ECTS pro Semester erfolgt eine automatische Einladung zur obligatorischen Studienberatung.",
                "keywords": "low ects credit deficit academic warning failed modules progression",
            },
            {
                "id": "[PO-201]",
                "title": "Structural Leave of Absence (Urlaubssemester)",
                "text": "Laut § 18 der Immatrikulationsordnung können reife Studierende oder Erwerbstätige bei nachgewiesener Belastung ein Urlaubssemester beantragen. Während dieses Zeitraums ruhen die regulären ECTS-Fristen, wodurch eine Exmatrikulation wegen Fristüberschreitung abgewendet wird.",
                "keywords": "mature student age older working employment adjustment break pause",
            },
            {
                "id": "[PO-301]",
                "title": "BAföG Progress Verification",
                "text": "Nach § 48 BAföG ist zum Ende des 4. Fachsemesters ein positiver Leistungsnachweis (Formblatt 5) vorzulegen. Eine frühzeitige Stabilisierung der ECTS-Zahlen im 1. und 2. Semester sichert den kontinuierlichen Erhalt der Bundesförderung und minimiert das Abbruchrisiko.",
                "keywords": "bafoeg financial aid recipient state funding grant verification support",
            },
            {
                "id": "[PO-302]",
                "title": "Hardship Applications & Installment Deferrals",
                "text": "Nach § 23 der Gebührenordnung führt ein Rückstand bei den Semesterbeiträgen zur Einleitung des Exmatrikulationsverfahrens. Betroffene Studierende können beim Studierendenwerk einen Härtefallantrag stellen, um eine Stundung oder Ratenzahlung der Rückstände zu vereinbaren.",
                "keywords": "fee arrears debt unpaid tuition bursar money outstanding balance delinquency",
            },
        ]

        self.corpus = [doc["keywords"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer()
        self.vector_database = self.vectorizer.fit_transform(self.corpus)

    def query_vector_space(self, student_profile, threshold=0.15):
        query_terms = []
        if student_profile.get("ECTS_Earned_Sem2", 30) < 15:
            query_terms.append("low ects credit deficit warning failed progression")
        if (
            student_profile.get("Age_At_Enrollment", 20) > 26
            and student_profile.get("ECTS_Earned_Sem2", 30) < 5
        ):
            query_terms.append("mature student age older adjustment struggle")
        if student_profile.get("Fee_Arrears", 0) == 1:
            query_terms.append("fee arrears debt unpaid tuition outstanding balance")
        if (
            student_profile.get("BAfoeg_Status", 0) == 1
            and student_profile.get("ECTS_Earned_Sem2", 30) < 20
        ):
            query_terms.append("bafoeg financial aid funding support review")

        if not query_terms:
            return "✅ Clear Standing: No critical regulatory anomalies vector-matched."

        combined_query = " ".join(query_terms)
        query_vector = self.vectorizer.transform([combined_query])
        similarity_scores = cosine_similarity(
            query_vector, self.vector_database
        ).flatten()

        results = ""
        match_count = 1
        for idx, score in enumerate(similarity_scores):
            if score >= threshold:
                doc = self.documents[idx]
                results += f"📄 [{match_count}] {doc['id']} {doc['title']} (Vector Match Score: {score:.2f})\n"
                results += f"   Regulatory Context: {doc['text']}\n\n"
                match_count += 1

        return (
            results
            if results
            else "✅ Clear Standing: Similarity values fell below tracking threshold."
        )


@st.cache_resource
def load_semantic_rag():
    return SemanticVectorRAG()


semantic_rag_engine = load_semantic_rag()


# =====================================================================
# 🎨 STEP 3: STREAMLIT FRONT-END UI DEPLOYMENT LAYOUT
# =====================================================================
st.set_page_config(page_title="EduMatch Dashboard", page_icon="🎓", layout="wide")

st.title(
    "EduMatch: Predictive Student Retention Pipeline With Advisor Decision Dashboard"
)
st.markdown(
    "Input a student's administrative, economic, and academic performance vectors "
    "to calculate empirical exmatriculation risks, map intervention segments, and extract "
    "matched legal regulatory provisions via mathematical cosine similarity."
)

st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("👤 Student Profile Inputs")

    age = st.slider("Age at Enrollment", 17, 60, value=22, step=1)
    gender = st.radio("Gender Identity", ["Female", "Male"])
    displaced = st.radio("Relocated Student (Displaced Status)", ["Yes", "No"], index=1)
    bafoeg = st.radio(
        "Federal Financial Aid Status (BAföG)", ["Yes (Recipient)", "No"], index=1
    )
    arrears = st.radio(
        "Semester Tuition Fee Arrears", ["Yes (Delinquent)", "No"], index=1
    )

    st.markdown("**Academic Performance Data**")
    ects_s1 = st.slider("Earned ECTS (1st Semester)", 10, 40, value=30, step=5)
    grade_s1 = st.slider(
        "Grade Average Point (1st Semester)", 1.0, 5.0, value=2.0, step=0.1
    )
    ects_s2 = st.slider("Earned ECTS (2nd Semester)", 10, 40, value=25, step=5)
    grade_s2 = st.slider(
        "Grade Average Point (2nd Semester)", 1.0, 5.0, value=2.3, step=0.1
    )


# =====================================================================
# 🧭 STEP 4: RUN LIVE BACKEND INTERACTION PIPELINE
# =====================================================================
with col2:
    st.header("📊 Advisor Analytical Output")

    # Structure frontend data vectors
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

    # Force alignment with feature orders
    if X_train_cols is not None:
        input_data = input_data[X_train_cols]

    # Check if files were successfully tracked by asset cache loader
    if german_retention_model is not None:
        # Run live model evaluation pipelines
        risk_probability = german_retention_model.predict_proba(input_data)[0][1]
        scaled_input = scaler.transform(input_data)
        assigned_cluster = kmeans.predict(scaled_input)[0]

        cluster_mapping = {
            0: "Cluster 0: Stabilized Academic Deficit (Financial Aid Cushioned)",
            1: "Cluster 1: Acute Academic Collapse (Severe Credit Deficit / Mature Students)",
            2: "Cluster 2: Latent Socioeconomic Risk (High Grades, Unpaid Semester Fees)",
        }
        cluster_name = cluster_mapping.get(assigned_cluster, "Unknown Cluster Profile")

        # Execute local Semantic Vector RAG Database query lookup
        student_profile_dict = input_data.iloc[0].to_dict()
        rag_output = semantic_rag_engine.query_vector_space(student_profile_dict)

        # Render Box A: Status & Probability rounded cleanly to whole values (:0%)
        status_indicator = (
            "🔴 CRITICAL RISK" if risk_probability > 0.5 else "🟢 STABLE STATUS"
        )
        if risk_probability > 0.5:
            st.error(
                f"**Operational Status:** {status_indicator}\n\n**Calculated Exmatriculation Probability:** {risk_probability:.0%}"
            )
        else:
            st.success(
                f"**Operational Status:** {status_indicator}\n\n**Calculated Exmatriculation Probability:** {risk_probability:.0%}"
            )

        # Render Box B: Cohort Cluster Outputs
        st.info(f"**🧭 Assigned Intervention Cohort:**\n\n{cluster_name}")

        # Render Box C: Vector Search RAG text area context layout
        st.subheader("📚 Legal Compliance & Regulatory Directives")
        st.text_area(
            label="Vector-Matched Examination Regulations (Prüfungsordnung)",
            value=rag_output,
            height=250,
            disabled=True,
        )
    else:
        # Graceful fallback visual state if .pkl binaries aren't found in repository models/ folder
        st.warning(
            "⚠️ Baseline pipeline configurations missing. Please ensure your machine learning asset files (.pkl) are pushed under the /models directory to activate real-time analytics."
        )
