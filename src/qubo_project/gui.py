import streamlit as st
import pandas as pd
import os

# Set up the page layout
st.set_page_config(page_title="QUBO Classification GUI", layout="wide")

st.title("Credit Risk Classification Pipeline")
st.markdown(
    "This GUI allows you to run the complete pipeline: Preprocessing, QUBO Feature Selection, Training, and Prediction.")

# --- 1. Select Dataset ---
st.header("1. Select Dataset")
dataset_path = st.text_input("Enter path to input CSV dataset:", "data/sample_test_dataset.csv")
target_col = st.text_input("Enter target column name:", "target")

if st.button("Load Dataset Preview"):
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            st.dataframe(df.head())
            st.success(f"Dataset loaded successfully. Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.error(f"File not found: {dataset_path}")

st.divider()

# --- 2. Preprocessing ---
st.header("2. Preprocessing")
min_perc_valid = st.slider("Minimum % of valid non-zero data for a column:", 0.0, 1.0, 0.05)

if st.button("Run Preprocessing"):
    st.info("Executing preprocessing...")
    # TODO: Import and call fit_normalize() from src.qubo_project.preprocessing here

    # Placeholder for UI feedback
    st.success("Preprocessing completed! Normalized dataset saved to outputs/normalized.csv")

st.divider()

# --- 3. Feature Selection (QUBO) ---
st.header("3. Feature Selection (QUBO)")
col1, col2 = st.columns(2)
with col1:
    perc_selected = st.number_input("Percentage of features to select:", min_value=0.01, max_value=1.0, value=0.20)
    allowance = st.number_input("Allowance (+/- features):", min_value=0, max_value=10, value=1)
with col2:
    perc_test = st.number_input("Percentage of test data:", min_value=0.1, max_value=0.9, value=0.30)
    alpha_computations = st.number_input("Max alpha computations:", min_value=1, max_value=500, value=100)

if st.button("Run QUBO Feature Selection"):
    st.info("Executing QUBO optimization...")
    # TODO: Import and call select_features() from src.qubo_project.feature_selection here

    # Placeholder for UI feedback
    st.success("Feature selection completed! Training and test sets reduced and saved.")

st.divider()

# --- 4. Model Training ---
st.header("4. Model Training")
classifier_choice = st.selectbox("Select Classifier:",
                                 ["random_forest", "logistic_regression", "support_vector_machine"])

if st.button("Train Model"):
    st.info(f"Training {classifier_choice}...")
    # TODO: Import and call train() from src.qubo_project.model here

    # Placeholder for UI feedback
    st.success("Model trained and saved to outputs/model.joblib!")

st.divider()

# --- 5. Predictions & Outputs ---
st.header("5. Predictions & Outputs")
if st.button("Run Predictions"):
    st.info("Generating predictions on the test set...")
    # TODO: Import and call predict() from src.qubo_project.model here

    # Placeholder for UI feedback
    st.success("Predictions generated and saved to outputs/predictions.csv!")

    st.subheader("Classification Statistics")
    st.markdown("*(Placeholder for statistics like ROC-AUC, F1-score, and confusion matrix visualization)*")