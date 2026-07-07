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

        if hasattr(model, "feature_names_in_"):
            x_train_columns = list(model.feature_names_in_)
        else:
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
        model, scaler, kmeans, x_train_columns = None, None, None, None

    return model, scaler, kmeans, x_train_columns


german_retention_model, scaler, kmeans, X_train_cols = load_analytics_assets()


# =====================================================================
# 📚 STEP 2: EXPANDED SEMANTIC VECTOR RAG ENGINE (WITH BOLD & COLOR MARKDOWN)
# =====================================================================
class SemanticVectorRAG:
    def __init__(self):
        self.documents = [
            {
                "id": "[PO-101]",
                "title": "Academic Progression Standard (ECTS Baseline)",
                "text": "Gemäß § 12 der Prüfungsordnung müssen Studierende bis zum Ende des zweiten Fachsemesters mindestens 30 ECTS-Punkte erbracht haben. Bei Unterschreitung dieser Schwelle von 15 ECTS pro Semester erfolgt eine automatische Einladung zur obligatorischen Studienberatung.",
                "keywords": "low ects credit deficit academic warning failed modules progression minimum credit requirements fail",
            },
            {
                "id": "[PO-102]",
                "title": "Academic Probation & Exam Limitations",
                "text": "Laut § 15 der Rahmenprüfungsordnung gefährdet ein Notendurchschnitt von schlechter als 4.0 in Hauptmodulen den Studienerfolg. Studierende geraten unter akademische Bewährung und müssen einen individuellen Studienverlaufsplan mit dem Prüfungsamt abstimmen.",
                "keywords": "bad grades high grade point average probation fail academic risk exam struggles failure",
            },
            {
                "id": "[PO-201]",
                "title": "Structural Leave of Absence (Urlaubssemester)",
                "text": "Laut § 18 der Immatrikulationsordnung können reife Studierende oder Erwerbstätige bei nachgewiesener Belastung ein Urlaubssemester beantragen. Während dieses Zeitraums ruhen die regulären ECTS-Fristen, wodurch eine Exmatrikulation wegen Fristüberschreitung abgewendet wird.",
                "keywords": "mature student age older working employment adjustment break pause balancing life study",
            },
            {
                "id": "[PO-202]",
                "title": "Integration Support for Relocated/Displaced Students",
                "text": "Nach § 5 des Landeshochschulgesetzes erhalten Studierende mit Fluchthintergrund oder Migrationshintergrund (Displaced Status) Zugang zu erweiterten Sprachzertifikatskursen und Härtefall-Fristverlängerungen bei Prüfungsfristen, um Integrationshemmnisse auszugleichen.",
                "keywords": "displaced relocated relocation student immigrant refugee language barrier adjustment structural move",
            },
            {
                "id": "[PO-301]",
                "title": "BAföG Progress Verification & Formblatt 5",
                "text": "Nach § 48 BAföG ist zum Ende des 4. Fachsemesters ein positiver Leistungsnachweis (Formblatt 5) vorzulegen. Eine frühzeitige Stabilisierung der ECTS-Zahlen im 1. und 2. Semester sichert den kontinuierlichen Erhalt der Bundesförderung und minimiert das Abbruchrisiko.",
                "keywords": "bafoeg financial aid recipient state funding grant verification support grant suspension money loss",
            },
            {
                "id": "[PO-302]",
                "title": "Hardship Applications & Installment Deferrals",
                "text": "Nach § 23 der Gebührenordnung führt ein Rückstand bei den Semesterbeiträgen zur Einleitung des Exmatrikulationsverfahrens. Betroffene Studierende können beim Studierendenwerk einen Härtefallantrag stellen, um eine Stundung oder Ratenzahlung der Rückstände zu vereinbaren.",
                "keywords": "fee arrears debt unpaid tuition bursar money outstanding balance delinquency financial distress fees unpaid",
            },
            {
                "id": "[PO-401]",
                "title": "Diversity and Gender-Specific Family Extensions",
                "text": "Gemäß § 21 der Gleichstellungsrichtlinie der Universität können Studierende mit Erziehungspflichten oder familiären Pflegeaufgaben alternative Prüfungsformen (Hausarbeiten statt Klausuren) beantragen, um geschlechtsspezifische Benachteiligungen im Studienverlauf zu reduzieren.",
                "keywords": "gender female identity equality family balancing kids parental leave balance diversity equity",
            },
            {
                "id": "[PO-501]",
                "title": "Psychosocial Counseling Intervention (Mental Health)",
                "text": "Bei extremen Diskrepanzen im Leistungsbild (z.B. hohe Noten im 1. Semester gefolgt von totalem ECTS-Einbruch im 2. Semester) verweist die Prüfungsordnung auf das psychosoziale Beratungsnetzwerk des Studierendenwerks zur Bewältigung von Prüfungsangst und Burnout.",
                "keywords": "stress anxiety mental health drop performance discrepancy collapse dropout burn out overwhelm panic",
            },
        ]

        self.corpus = [doc["keywords"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer()
        self.vector_database = self.vectorizer.fit_transform(self.corpus)

    def query_vector_space(self, student_profile, threshold=0.10):
        query_terms = []

        ects_s1 = student_profile.get("ECTS_Earned_Sem1", 30)
        ects_s2 = student_profile.get("ECTS_Earned_Sem2", 30)
        grade_s1 = student_profile.get("Grade_Avg_Sem1", 2.0)
        grade_s2 = student_profile.get("Grade_Avg_Sem2", 2.0)

        if (ects_s1 + ects_s2) < 40 or ects_s2 < 15:
            query_terms.append(
                "low ects credit deficit warning failed progression minimum credit requirements fail"
            )
        if grade_s1 > 3.5 or grade_s2 > 3.5:
            query_terms.append(
                "bad grades high grade point average probation fail academic risk exam struggles failure"
            )
        if ects_s1 >= 30 and ects_s2 <= 15:
            query_terms.append(
                "stress anxiety mental health drop performance discrepancy collapse dropout burn out overwhelm"
            )
        if student_profile.get("Age_At_Enrollment", 22) > 26:
            query_terms.append(
                "mature student age older working employment adjustment break pause balancing life study"
            )
        if student_profile.get("Fee_Arrears", 0) == 1:
            query_terms.append(
                "fee arrears debt unpaid tuition bursar money outstanding balance delinquency financial distress fees unpaid"
            )
        if student_profile.get("BAfoeg_Status", 0) == 1:
            query_terms.append(
                "bafoeg financial aid recipient state funding grant verification support grant suspension money loss"
            )
        if student_profile.get("Displaced_Status", 0) == 1:
            query_terms.append(
                "displaced relocated relocation student immigrant refugee language barrier adjustment structural move"
            )
        if student_profile.get("Gender", 0) == 0:
            query_terms.append(
                "gender female identity equality family balancing kids parental leave balance diversity equity"
            )

        if not query_terms:
            return ":green[**✅ Clear Standing: No critical regulatory anomalies vector-matched.**]"

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
                # 🎯 HIGHLIGHTED COLORING: Bold text tags and color-styled code identifiers
                results += f"📄 **[{match_count}]** :red[**{doc['id']}**] **{doc['title']}** *(Vector Match Score: {score:.2f})*\n\n"
                results += (
                    f"&nbsp;&nbsp;&nbsp;&nbsp;**Regulatory Context:** {doc['text']}\n\n"
                )
                results += "--- \n\n"
                match_count += 1

        return (
            results
            if results
            else ":green[**✅ Clear Standing: Similarity values fell below tracking threshold.**]"
        )


@st.cache_resource
def load_semantic_rag():
    return SemanticVectorRAG()


semantic_rag_engine = load_semantic_rag()


# =====================================================================
# 🎨 STEP 3: STREAMLIT FRONT-END UI LAYOUT WITH FORM SYSTEM
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
    with st.form("advisor_input_form"):
        st.header("👤 Student Profile Inputs")

        age = st.slider("Age at Enrollment", 17, 60, value=22, step=1)
        gender = st.radio("Gender Identity", ["Female", "Male"])
        displaced = st.radio(
            "Relocated Student (Displaced Status)", ["Yes", "No"], index=1
        )
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

        submitted = st.form_submit_button(
            "Calculate Risk & Match Regulations", type="primary"
        )


# =====================================================================
# 🧭 STEP 4: RUN PIPELINE CONDITIONALLY ON CLICK
# =====================================================================
with col2:
    st.header("📊 Advisor Analytical Output")

    if submitted:
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

        if X_train_cols is not None:
            input_data = input_data[X_train_cols]

        if german_retention_model is not None:
            with st.spinner(
                "Processing performance metrics through neural matrices..."
            ):
                risk_probability = german_retention_model.predict_proba(input_data)[0][
                    1
                ]
                scaled_input = scaler.transform(input_data)
                assigned_cluster = kmeans.predict(scaled_input)[0]

                cluster_mapping = {
                    0: "Cluster 0: Stabilized Academic Deficit (Financial Aid Cushioned)",
                    1: "Cluster 1: Acute Academic Collapse (Severe Credit Deficit / Mature Students)",
                    2: "Cluster 2: Latent Socioeconomic Risk (High Grades, Unpaid Semester Fees)",
                }
                cluster_name = cluster_mapping.get(
                    assigned_cluster, "Unknown Cluster Profile"
                )

                student_profile_dict = input_data.iloc[0].to_dict()
                rag_output = semantic_rag_engine.query_vector_space(
                    student_profile_dict
                )

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

            st.info(f"**🧭 Assigned Intervention Cohort:**\n\n{cluster_name}")

            # 🎯 STRUCTURAL CHANGE: Using Markdown instead of text_area to pop bold and color text formatting cleanly
            st.markdown(
                "### **Vector-Matched Examination Regulations (Prüfungsordnung)**"
            )
            st.markdown(rag_output, unsafe_allow_html=True)
        else:
            st.warning(
                "⚠️ Baseline pipeline configurations missing. Please ensure your machine learning asset files (.pkl) are pushed under the /models directory to activate real-time analytics."
            )
    else:
        st.info(
            "💡 Adjust student parameter attributes inside the sidebar panel and click **'Calculate Risk & Match Regulations'** to process diagnostic indicators."
        )
