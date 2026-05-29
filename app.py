import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("cluster_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 AI-Powered Employee Attrition Prediction System")
st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Employee Prediction",
        "Cluster Information",
        "About Project"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.header("🏢 HR Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Employees", "1470")

    with col2:
        st.metric("Attrition Rate", "16.1%")

    with col3:
        st.metric("Clusters", "3")

    st.markdown("""
    ### Project Objective

    Predict employee attrition risk using Machine Learning and
    segment employees into meaningful workforce groups.

    **Models Used**
    - Logistic Regression (Final Model)
    - Decision Tree
    - Random Forest
    - SVM
    - KNN
    - Naive Bayes
    - PCA
    - K-Means Clustering

    **Selected Production Model**
    - Balanced Logistic Regression
    """)

# =====================================================
# EMPLOYEE PREDICTION PAGE
# =====================================================
elif page == "Employee Prediction":

    st.header("🔮 Employee Attrition Prediction")

    st.info(
        "Enter employee details and click Predict."
    )

    # -------------------------------------------------
    # USER INPUTS
    # -------------------------------------------------

    age = st.slider("Age", 18, 60, 30)

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=25000,
        value=5000
    )

    total_working_years = st.slider(
        "Total Working Years",
        0,
        40,
        5
    )

    years_at_company = st.slider(
        "Years At Company",
        0,
        40,
        3
    )

    years_since_last_promotion = st.slider(
        "Years Since Last Promotion",
        0,
        15,
        1
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    distance_from_home = st.slider(
        "Distance From Home",
        1,
        30,
        5
    )

    num_companies_worked = st.slider(
        "Number of Companies Worked",
        0,
        10,
        2
    )

    # -------------------------------------------------
    # CREATE INPUT RECORD
    # -------------------------------------------------

    if st.button("Predict Attrition Risk"):

        employee = pd.DataFrame(
            np.zeros((1, len(feature_columns))),
            columns=feature_columns
        )

        # Numerical Features
        if "Age" in employee.columns:
            employee["Age"] = age

        if "MonthlyIncome" in employee.columns:
            employee["MonthlyIncome"] = monthly_income

        if "TotalWorkingYears" in employee.columns:
            employee["TotalWorkingYears"] = total_working_years

        if "YearsAtCompany" in employee.columns:
            employee["YearsAtCompany"] = years_at_company

        if "YearsSinceLastPromotion" in employee.columns:
            employee["YearsSinceLastPromotion"] = years_since_last_promotion

        if "JobSatisfaction" in employee.columns:
            employee["JobSatisfaction"] = job_satisfaction

        if "EnvironmentSatisfaction" in employee.columns:
            employee["EnvironmentSatisfaction"] = environment_satisfaction

        if "WorkLifeBalance" in employee.columns:
            employee["WorkLifeBalance"] = work_life_balance

        if "DistanceFromHome" in employee.columns:
            employee["DistanceFromHome"] = distance_from_home

        if "NumCompaniesWorked" in employee.columns:
            employee["NumCompaniesWorked"] = num_companies_worked

        # Binary Features
        if "OverTime" in employee.columns:
            employee["OverTime"] = 1 if overtime == "Yes" else 0

        # -------------------------------------------------
        # SCALE
        # -------------------------------------------------

        employee_scaled = scaler.transform(employee)

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        probability = model.predict_proba(
            employee_scaled
        )[0][1]

        prediction = model.predict(
            employee_scaled
        )[0]

        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if probability < 0.30:
            risk = "LOW"
        elif probability < 0.60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Attrition Probability",
                f"{probability*100:.2f}%"
            )

        with col2:
            st.metric(
                "Risk Level",
                risk
            )

        # -------------------------------------------------
        # GAUGE CHART
        # -------------------------------------------------

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={
                    "text": "Attrition Risk"
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # CLUSTER PREDICTION
        # -------------------------------------------------

        cluster = kmeans.predict(
            employee_scaled
        )[0]

        cluster_names = {
            0: "Core Workforce",
            1: "At-Risk Employees",
            2: "Senior Stable Employees"
        }

        st.success(
            f"Employee Segment: {cluster_names.get(cluster, 'Unknown')}"
        )

# =====================================================
# CLUSTER PAGE
# =====================================================
elif page == "Cluster Information":

    st.header("📌 Employee Segments")

    st.subheader("Cluster 0")
    st.write("""
    Core Workforce

    - Younger employees
    - Moderate experience
    - Moderate attrition risk
    """)

    st.subheader("Cluster 1")
    st.write("""
    At-Risk Employees

    - Highest attrition rate
    - Mid-career professionals
    - Need retention programs
    """)

    st.subheader("Cluster 2")
    st.write("""
    Senior Stable Employees

    - High experience
    - High salary
    - Lowest attrition risk
    """)

# =====================================================
# ABOUT PAGE
# =====================================================
else:

    st.header("ℹ About Project")

    st.markdown("""
    ### AI-Powered Employee Attrition Prediction System

    Developed using:

    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    - Plotly

    ### Machine Learning Models

    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM
    - KNN
    - Naive Bayes

    ### Dimensionality Reduction

    - PCA

    ### Clustering

    - K-Means (3 Employee Segments)

    ### Final Selected Model

    Balanced Logistic Regression

    Selected using Stratified 5-Fold Cross Validation.
    """)