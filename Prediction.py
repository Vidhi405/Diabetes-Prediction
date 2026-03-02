import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)


@st.cache_data
def load_data():
    df = pd.read_csv("D:\\ProjectsTwo\\Cleaned_DiabetesDataset.csv")
    return df


def preprocess(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df


def train_model(df):
    X = df.drop(columns=['diagnosed_diabetes'])
    y = df['diagnosed_diabetes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_test, y_pred, X.columns.tolist()


def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("Built with **scikit-learn** + **Streamlit** | Random Forest Classifier")

    raw_df = load_data()
    df = preprocess(raw_df)

    st.sidebar.header("Options")
    show_data = st.sidebar.checkbox("Show raw dataset")
    if show_data:
        st.subheader("Raw Dataset")
        st.dataframe(raw_df)

    model, X_train, X_test, y_test, y_pred, feature_names = train_model(df)

    st.header("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    accuracy = accuracy_score(y_test, y_pred)
    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    precision = precision_score(y_test, y_pred)
    col2.metric("Precision", f"{precision * 100:.2f}%")
    recall = recall_score(y_test, y_pred)
    col3.metric("Recall", f"{recall * 100:.2f}%")

    st.header("🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.header("🔍 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis', ax=ax2)
    ax2.set_title("Top 10 Most Important Features")
    st.pyplot(fig2)

    st.header("🧑‍⚕️ Predict for a New Patient")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        age = st.slider("Age", 18, 90, 45)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0)
        systolic_bp = st.slider("Systolic BP", 80, 200, 120)
        diastolic_bp = st.slider("Diastolic BP", 50, 130, 80)

    with col_b:
        cholesterol = st.slider("Total Cholesterol", 100, 400, 200)
        hdl = st.slider("HDL Cholesterol", 20, 100, 50)
        ldl = st.slider("LDL Cholesterol", 50, 300, 100)
        triglycerides = st.slider("Triglycerides", 50, 500, 150)

    with col_c:
        family_history = st.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        hypertension = st.selectbox("Hypertension History", [0, 1], format_func=lambda x: "Yes" if x else "No")
        cardiovascular = st.selectbox("Cardiovascular History", [0, 1], format_func=lambda x: "Yes" if x else "No")
        physical_activity = st.slider("Physical Activity (mins/week)", 0, 500, 150)

    processed_df = preprocess(raw_df)
    X_all = processed_df.drop(columns=['diagnosed_diabetes'])
    medians = X_all.median()

    input_data = medians.copy()
    input_data['age'] = age
    input_data['bmi'] = bmi
    input_data['systolic_bp'] = systolic_bp
    input_data['diastolic_bp'] = diastolic_bp
    input_data['cholesterol_total'] = cholesterol
    input_data['hdl_cholesterol'] = hdl
    input_data['ldl_cholesterol'] = ldl
    input_data['triglycerides'] = triglycerides
    input_data['family_history_diabetes'] = family_history
    input_data['hypertension_history'] = hypertension
    input_data['cardiovascular_history'] = cardiovascular
    input_data['physical_activity_minutes_per_week'] = physical_activity

    input_array = pd.DataFrame([input_data])
    prob = model.predict_proba(input_array)[0][1]
    prediction = model.predict(input_array)[0]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** — Model confidence: {prob * 100:.1f}%")
    else:
        st.success(f"✅ **Low Risk of Diabetes** — Model confidence: {(1 - prob) * 100:.1f}%")

    st.caption("⚠️ This is an educational ML project. Not a substitute for medical advice.")


if __name__ == "__main__":
    main()