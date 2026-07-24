import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import json
import streamlit as st
import pandas as pd

# Import project modules
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

# Configure page layout
st.set_page_config(
    page_title="G31 QUBO Classification Project",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ G31 QUBO Project GUI")
st.markdown("Binary classification pipeline with QUBO feature reduction[cite: 5, 6].")

# Ensure required directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Define file paths
INPUT_CSV = os.path.join("data", "input_dataset.csv")
NORMALIZED_CSV = os.path.join("outputs", "normalized.csv")
PREPROC_JSON = os.path.join("outputs", "preprocessing_result.json")
TRAIN_REDUCED_CSV = os.path.join("outputs", "training_reduced.csv")
TEST_REDUCED_CSV = os.path.join("outputs", "test_reduced.csv")
OPTIM_CSV = os.path.join("outputs", "optimizations.csv")
FEAT_SEL_JSON = os.path.join("outputs", "feature_selection_result.json")
MODEL_PATH = os.path.join("outputs", "model.joblib")
TRAIN_METRICS_JSON = os.path.join("outputs", "training_metrics.json")
PREDICTIONS_CSV = os.path.join("outputs", "predictions.csv")
CLASSIF_STATS_JSON = os.path.join("outputs", "classification_stats.json")

# --- SIDEBAR: Configuration & Controls ---
st.sidebar.header("⚙️ Pipeline Controls & Parameters")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=['csv'])

st.sidebar.subheader("Hyperparameters")
target_column = st.sidebar.text_input("Target Column Name", value="target")
min_perc_valid = st.sidebar.slider("Min % Valid Data (Preprocessing)", 0.0, 0.5, 0.05, 0.01)
perc_selected = st.sidebar.slider("Target Feature % (QUBO)", 0.05, 0.90, 0.20, 0.05)
allowance = st.sidebar.number_input("Feature Count Allowance ($\pm$)", min_value=0, max_value=10, value=1)
perc_test = st.sidebar.slider("Test Set %", 0.10, 0.50, 0.30, 0.05)
classifier_choice = st.sidebar.selectbox(
    "Classifier",
    ["random_forest", "logistic_regression", "gradient_boosting"]
)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
alpha_computations = st.sidebar.slider("Max Alpha Computations", 10, 200, 50, 10)

# --- MAIN INTERFACE ---
if uploaded_file is not None:
    # Save uploaded file locally so modules can read it via path
    with open(INPUT_CSV, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        df_preview = pd.read_csv(INPUT_CSV)
        st.success(f"Dataset successfully loaded! Shape: {df_preview.shape[0]} rows, {df_preview.shape[1]} columns.")

        with st.expander("🔍 View Raw Data Preview"):
            st.dataframe(df_preview.head(10))

        st.markdown("---")
        st.subheader("🚀 Execute Pipeline Steps")

        col1, col2 = st.columns(2)

        with col1:
            # Step 1: Preprocessing
            if st.button("1. Run Preprocessing", use_container_width=True):
                with st.spinner("Cleaning and normalizing dataset..."):
                    try:
                        fit_normalize(
                            input_csv=INPUT_CSV,
                            target_column=target_column,
                            normalized_csv=NORMALIZED_CSV,
                            outInitalRes_json=PREPROC_JSON,
                            minPercValid=min_perc_valid
                        )
                        st.success("Preprocessing completed successfully!")
                        if os.path.exists(PREPROC_JSON):
                            with open(PREPROC_JSON, "r") as jf:
                                st.json(json.load(jf))
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")

            # Step 2: Feature Selection
            if st.button("2. Run QUBO Feature Selection", use_container_width=True):
                if not os.path.exists(NORMALIZED_CSV):
                    st.warning("Please run preprocessing first.")
                else:
                    with st.spinner("Solving QUBO feature selection (varying alpha)..."):
                        try:
                            select_features(
                                normalized_csv=NORMALIZED_CSV,
                                reducedTrain_csv=TRAIN_REDUCED_CSV,
                                reducedTest_csv=TEST_REDUCED_CSV,
                                output_ottim_csv=OPTIM_CSV,
                                output_json=FEAT_SEL_JSON,
                                target_column=target_column,
                                percTest=perc_test,
                                percSelected=perc_selected,
                                allowance=allowance,
                                seed=seed,
                                alpha_computations=alpha_computations
                            )
                            st.success("Feature selection completed!")
                            if os.path.exists(FEAT_SEL_JSON):
                                with open(FEAT_SEL_JSON, "r") as jf:
                                    st.json(json.load(jf))
                        except Exception as e:
                            st.error(f"Feature selection failed: {e}")

        with col2:
            # Step 3: Training
            if st.button("3. Train Classifier", use_container_width=True):
                if not os.path.exists(TRAIN_REDUCED_CSV):
                    st.warning("Please run feature selection first.")
                else:
                    with st.spinner(f"Training {classifier_choice}..."):
                        try:
                            train(
                                classifier=classifier_choice,
                                reducedTrain_csv=TRAIN_REDUCED_CSV,
                                target_column=target_column,
                                model_path=MODEL_PATH,
                                metrics_json=TRAIN_METRICS_JSON,
                                seed=int(seed)
                            )
                            st.success("Model trained successfully!")
                            if os.path.exists(TRAIN_METRICS_JSON):
                                with open(TRAIN_METRICS_JSON, "r") as jf:
                                    st.json(json.load(jf))
                        except Exception as e:
                            st.error(f"Training failed: {e}")

            # Step 4: Prediction
            if st.button("4. Run Predictions & Evaluate", use_container_width=True):
                if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_REDUCED_CSV):
                    st.warning("Please ensure model and test dataset are available.")
                else:
                    with st.spinner("Generating predictions and computing metrics..."):
                        try:
                            predict(
                                reduced_Test_csv=TEST_REDUCED_CSV,
                                target_column=target_column,
                                model_path=MODEL_PATH,
                                predictions_csv=PREDICTIONS_CSV,
                                classif_stats_json=CLASSIF_STATS_JSON
                            )
                            st.success("Predictions generated!")
                            if os.path.exists(CLASSIF_STATS_JSON):
                                with open(CLASSIF_STATS_JSON, "r") as jf:
                                    st.json(json.load(jf))
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

        st.markdown("---")
        st.subheader("📊 Main Outputs & Downloads")

        # Display results or download buttons if files exist
        output_tabs = st.tabs(["Predictions", "Optimization Logs", "Download Files"])

        with output_tabs[0]:
            if os.path.exists(PREDICTIONS_CSV):
                df_preds = pd.read_csv(PREDICTIONS_CSV)
                st.write(f"Showing predictions (Total rows: {len(df_preds)})")
                st.dataframe(df_preds.head(100))
            else:
                st.info("No predictions generated yet.")

        with output_tabs[1]:
            if os.path.exists(OPTIM_CSV):
                df_opt = pd.read_csv(OPTIM_CSV)
                st.write("Alpha Optimization Iterations")
                st.line_chart(df_opt.set_index("alpha")["n_features"])
                st.dataframe(df_opt)
            else:
                st.info("No optimization logs available yet.")

        with output_tabs[2]:
            st.markdown("### Download Generated Artifacts")
            col_d1, col_d2, col_d3 = st.columns(3)

            with col_d1:
                if os.path.exists(NORMALIZED_CSV):
                    with open(NORMALIZED_CSV, "rb") as f:
                        st.download_button("Download Normalized CSV", f, file_name="normalized.csv", mime="text/csv")
                if os.path.exists(TRAIN_REDUCED_CSV):
                    with open(TRAIN_REDUCED_CSV, "rb") as f:
                        st.download_button("Download Training Reduced CSV", f, file_name="training_reduced.csv", mime="text/csv")

            with col_d2:
                if os.path.exists(TEST_REDUCED_CSV):
                    with open(TEST_REDUCED_CSV, "rb") as f:
                        st.download_button("Download Test Reduced CSV", f, file_name="test_reduced.csv", mime="text/csv")
                if os.path.exists(MODEL_PATH):
                    with open(MODEL_PATH, "rb") as f:
                        st.download_button("Download Model (.joblib)", f, file_name="model.joblib", mime="application/octet-stream")

            with col_d3:
                if os.path.exists(PREDICTIONS_CSV):
                    with open(PREDICTIONS_CSV, "rb") as f:
                        st.download_button("Download Predictions CSV", f, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error processing the uploaded file. Please ensure it is a valid CSV. Details: {e}")
else:
    st.warning("👈 Please upload a dataset from the sidebar to begin.")