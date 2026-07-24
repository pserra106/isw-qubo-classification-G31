import os
import json
import pytest
import pandas as pd
import numpy as np

# Import the project modules
from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features
from src.qubo_project.model import train, predict

# Define Paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_test_dataset.csv")

# Preprocessing outputs
NORMALIZED_CSV = os.path.join(OUTPUT_DIR, "normalized.csv")
PREPROC_JSON = os.path.join(OUTPUT_DIR, "preprocessing_result.json")

# Feature Selection outputs
TRAIN_REDUCED_CSV = os.path.join(OUTPUT_DIR, "training_reduced.csv")
TEST_REDUCED_CSV = os.path.join(OUTPUT_DIR, "test_reduced.csv")
OPTIM_CSV = os.path.join(OUTPUT_DIR, "optimizations.csv")
FEAT_SEL_JSON = os.path.join(OUTPUT_DIR, "feature_selection_result.json")

# Model outputs
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
TRAIN_METRICS_JSON = os.path.join(OUTPUT_DIR, "training_metrics.json")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
CLASSIF_STATS_JSON = os.path.join(OUTPUT_DIR, "classification_stats.json")

TARGET_COL = "target"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure the sample dataset exists before running tests."""
    assert os.path.exists(SAMPLE_CSV), f"Sample dataset not found at {SAMPLE_CSV}. Run generate_sample_dataset.py first."
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_preprocessing():
    # Execute Preprocessing
    fit_normalize(
        input_csv=SAMPLE_CSV,
        target_column=TARGET_COL,
        normalized_csv=NORMALIZED_CSV,
        outInitalRes_json=PREPROC_JSON,
        minPercValid=0.05
    )
    
    assert os.path.exists(NORMALIZED_CSV), "Normalized CSV was not generated."
    df = pd.read_csv(NORMALIZED_CSV)
    
    # 1. Verify that preprocessing produces only numeric columns
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric."
        
    # 2. Verify that preprocessing handles missing values
    assert df.isna().sum().sum() == 0, "Preprocessing left NaN/Missing values in the dataset."
    
    # 3. Verify that normalization produces a valid dataset
    assert np.isfinite(df.values).all(), "Normalized dataset contains infinite values."
    
    features = df.drop(columns=[TARGET_COL])
    means = features.mean()
    # z-score normalization should center means around 0
    assert np.allclose(means, 0, atol=1e-1), "Features are not correctly normalized (mean is not approx 0)."


def test_feature_selection():
    # Execute Feature Selection
    select_features(
        normalized_csv=NORMALIZED_CSV,
        reducedTrain_csv=TRAIN_REDUCED_CSV,
        reducedTest_csv=TEST_REDUCED_CSV,
        output_ottim_csv=OPTIM_CSV,
        output_json=FEAT_SEL_JSON,
        target_column=TARGET_COL,
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=5  # Keep it low to speed up automated tests
    )
    
    assert os.path.exists(FEAT_SEL_JSON), "Feature selection JSON was not generated."
    
    with open(FEAT_SEL_JSON, "r") as f:
        res = json.load(f)
        
    selected_vector = res["selected_vector"]
    n_features = res["n_features"]
    n_selected = res["n_selected"]
    
    # 4. Verify that feature selection produces a binary vector
    for val in selected_vector:
        assert val in [0, 1], f"Vector contains non-binary value: {val}"
        
    # 5. Verify that the number of selected features is approximately 20%
    target_k = round(0.20 * n_features)
    allowance = 1
    assert abs(n_selected - target_k) <= allowance, f"Expected approx {target_k} selected features, got {n_selected}."


def test_training():
    # Execute Training
    train(
        classifier="random_forest",
        reducedTrain_csv=TRAIN_REDUCED_CSV,
        target_column=TARGET_COL,
        model_path=MODEL_PATH,
        metrics_json=TRAIN_METRICS_JSON,
        seed=42
    )
    
    # 6. Verify that training produces a saved model
    assert os.path.exists(MODEL_PATH), "Trained model file (.joblib) was not saved."
    assert os.path.exists(TRAIN_METRICS_JSON), "Training metrics JSON was not saved."


def test_prediction():
    # Execute Prediction
    predict(
        reduced_Test_csv=TEST_REDUCED_CSV,
        target_column=TARGET_COL,
        model_path=MODEL_PATH,
        predictions_csv=PREDICTIONS_CSV,
        classif_stats_json=CLASSIF_STATS_JSON
    )
    
    assert os.path.exists(PREDICTIONS_CSV), "Predictions CSV was not generated."
    
    df_pred = pd.read_csv(PREDICTIONS_CSV)
    
    # 7. Verify that prediction produces a CSV file with the required columns
    expected_columns = ["row_n", "target", "prediction", "score"]
    for col in expected_columns:
        assert col in df_pred.columns, f"Required column '{col}' is missing in predictions output."