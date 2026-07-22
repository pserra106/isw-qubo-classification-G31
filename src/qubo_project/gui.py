import streamlit as st
import pandas as pd
import os
import json

from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

# Set up the page layout
st.set_page_config(page_title="QUBO Classification GUI", layout="wide")

st.title("Credit Risk Classification Pipeline")
st.markdown(
    "This GUI allows you to run the complete pipeline: Preprocessing, QUBO Feature Selection, Training, and Prediction.")

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

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
normalized_output_path = "outputs/normalized.csv"
preprocessing_json_path = "outputs/preprocessing_result.json"

if st.button("Run Preprocessing"):
    if not os.path.exists(dataset_path):
        st.error("Please load a valid dataset first.")
    else:
        with st.spinner("Executing preprocessing..."):
            try:
                fit_normalize(
                    input_csv=dataset_path,
                    target_column=target_col,
                    normalized_csv=normalized_output_path,
                    outInitalRes_json=preprocessing_json_path,
                    minPercValid=min_perc_valid
                )
                st.success(f"Preprocessing completed! Normalized dataset saved to {normalized_output_path}")

                if os.path.exists(preprocessing_json_path):
                    with open(preprocessing_json_path, "r") as f:
                        prep_stats = json.load(f)
                    st.json(prep_stats)
            except Exception as e:
                st.error(f"An error occurred during preprocessing: {e}")

st.divider()

# --- 3. Feature Selection (QUBO) ---
st.header("3. Feature Selection (QUBO)")
col1, col2 = st.columns(2)
with col1:
    perc_selected = st.number_input("Percentage of features to select:", min_value=0.01, max_value=1.0, value=0.20)
    allowance = st.number_input("Allowance (+/- features):", min_value=0, max_value=10, value=1)
    seed_val = st.number_input("Random Seed:", min_value=0, max_value=1000, value=42)
with col2:
    perc_test = st.number_input("Percentage of test data:", min_value=0.1, max_value=0.9, value=0.30)
    alpha_computations = st.number_input("Max alpha computations:", min_value=1, max_value=500, value=50)

train_reduced_path = "outputs/training_reduced.csv"
test_reduced_path = "outputs/test_reduced.csv"
optimizations_path = "outputs/optimizations.csv"
fs_json_path = "outputs/feature_selection_result.json"

if st.button("Run QUBO Feature Selection"):
    if not os.path.exists(normalized_output_path):
        st.error("Please run preprocessing first to generate the normalized dataset.")
    else:
        with st.spinner("Executing QUBO optimization (this may take a while)..."):
            try:
                select_features(
                    normalized_csv=normalized_output_path,
                    reducedTrain_csv=train_reduced_path,
                    reducedTest_csv=test_reduced_path,
                    output_ottim_csv=optimizations_path,
                    output_json=fs_json_path,
                    target_column=target_col,
                    percTest=perc_test,
                    allowance=allowance,
                    seed=seed_val,
                    percSelected=perc_selected,
                    alpha_computations=alpha_computations
                )
                st.success("Feature selection completed! Training and test sets reduced and saved.")

                if os.path.exists(fs_json_path):
                    with open(fs_json_path, "r") as f:
                        fs_stats = json.load(f)
                    st.json(fs_stats)
            except Exception as e:
                st.error(f"An error occurred during QUBO optimization: {e}")

st.divider()

# --- 4. Model Training ---
st.header("4. Model Training")
classifier_choice = st.selectbox("Select Classifier:", ["random_forest", "logistic_regression", "decision_tree", "svm"])
model_path = "outputs/model.joblib"
metrics_json_path = "outputs/training_metrics.json"

if st.button("Train Model"):
    if not os.path.exists(train_reduced_path):
        st.error("Please run feature selection first to generate the reduced training dataset.")
    else:
        with st.spinner(f"Training {classifier_choice}..."):
            try:
                train(
                    classifier=classifier_choice,
                    reducedTrain_csv=train_reduced_path,
                    target_column=target_col,
                    model_path=model_path,
                    metrics_json=metrics_json_path,
                    seed=seed_val
                )
                st.success(f"Model trained and saved to {model_path}!")

                if os.path.exists(metrics_json_path):
                    with open(metrics_json_path, "r") as f:
                        train_metrics = json.load(f)
                    st.json(train_metrics)
            except Exception as e:
                st.error(f"An error occurred during training: {e}")

st.divider()

# --- 5. Predictions & Outputs ---
st.header("5. Predictions & Outputs")
predictions_path = "outputs/predictions.csv"
stats_json_path = "outputs/classification_stats.json"

if st.button("Run Predictions"):
    if not os.path.exists(model_path) or not os.path.exists(test_reduced_path):
        st.error("Trained model or reduced test set missing. Please complete previous steps.")
    else:
        with st.spinner("Generating predictions on the test set..."):
            try:
                predict(
                    reduced_Test_csv=test_reduced_path,
                    target_column=target_col,
                    model_path=model_path,
                    predictions_csv=predictions_path,
                    classif_stats_json=stats_json_path
                )
                st.success(f"Predictions generated and saved to {predictions_path}!")

                if os.path.exists(predictions_path):
                    pred_df = pd.read_csv(predictions_path)
                    st.subheader("Predictions Preview")
                    st.dataframe(pred_df.head(10))

                if os.path.exists(stats_json_path):
                    with open(stats_json_path, "r") as f:
                        class_stats = json.load(f)
                    st.subheader("Classification Statistics")
                    st.json(class_stats)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")