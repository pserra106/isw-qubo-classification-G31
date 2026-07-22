import pytest
import os
import pandas as pd
import numpy as np

from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features
from src.qubo_project.model import train, predict


@pytest.fixture
def dummy_dataset_path():
    """Returns the path to the mandatory test dataset."""
    return "data/sample_test_dataset.csv"


@pytest.fixture
def temp_outputs(tmp_path):
    """Creates temporary paths for pipeline outputs during testing."""
    return {
        "norm_csv": str(tmp_path / "normalized.csv"),
        "norm_json": str(tmp_path / "preprocessing_result.json"),
        "train_csv": str(tmp_path / "training_reduced.csv"),
        "test_csv": str(tmp_path / "test_reduced.csv"),
        "optim_csv": str(tmp_path / "optimizations.csv"),
        "fs_json": str(tmp_path / "feature_selection_result.json"),
        "model_path": str(tmp_path / "model.joblib"),
        "metrics_json": str(tmp_path / "training_metrics.json"),
        "preds_csv": str(tmp_path / "predictions.csv"),
        "stats_json": str(tmp_path / "classification_stats.json"),
    }


# --- 1. Preprocessing Tests ---
def test_preprocessing_produces_only_numeric_columns(dummy_dataset_path, temp_outputs):
    """1. Verify that preprocessing produces only numeric columns."""
    fit_normalize(
        input_csv=dummy_dataset_path,
        target_column="target",
        normalized_csv=temp_outputs["norm_csv"],
        outInitalRes_json=temp_outputs["norm_json"],
        minPercValid=0.05
    )
    df_norm = pd.read_csv(temp_outputs["norm_csv"])
    # Tutte le colonne devono essere di tipo numerico (int o float)
    for col in df_norm.columns:
        assert pd.api.types.is_numeric_dtype(df_norm[col]), f"La colonna {col} non è numerica."


def test_preprocessing_handles_missing_values(dummy_dataset_path, temp_outputs):
    """2. Verify that preprocessing handles missing values (NaNs)."""
    fit_normalize(
        input_csv=dummy_dataset_path,
        target_column="target",
        normalized_csv=temp_outputs["norm_csv"],
        outInitalRes_json=temp_outputs["norm_json"],
        minPercValid=0.05
    )
    df_norm = pd.read_csv(temp_outputs["norm_csv"])
    # Verifica che non ci siano valori NaN nel dataset normalizzato risultante
    assert not df_norm.isna().any().any(), "Il dataset normalizzato contiene ancora valori NaN."


def test_normalization_produces_valid_dataset(dummy_dataset_path, temp_outputs):
    """3. Verify that normalization produces a valid dataset (e.g., mean ~0, std ~1 for features)."""
    fit_normalize(
        input_csv=dummy_dataset_path,
        target_column="target",
        normalized_csv=temp_outputs["norm_csv"],
        outInitalRes_json=temp_outputs["norm_json"],
        minPercValid=0.05
    )
    df_norm = pd.read_csv(temp_outputs["norm_csv"])
    X_features = df_norm.drop(columns=["target"])

    # Per la standardizzazione z-score, la media deve essere circa 0 e la deviazione standard circa 1
    means = X_features.mean()
    stds = X_features.std()

    assert np.allclose(means, 0, atol=1e-1), "La media delle feature normalizzate non è vicina a 0."
    assert np.allclose(stds, 1, atol=1e-1) or (stds == 0).all(), "La deviazione standard non è vicina a 1."


# --- 2. Feature Selection Tests ---
def test_feature_selection_produces_binary_vector(dummy_dataset_path, temp_outputs):
    """4. Verify that feature selection produces a binary vector [0, 1, ...]."""
    # Eseguiamo prima il preprocessing necessario
    fit_normalize(dummy_dataset_path, "target", temp_outputs["norm_csv"], temp_outputs["norm_json"], 0.05)

    select_features(
        normalized_csv=temp_outputs["norm_csv"],
        reducedTrain_csv=temp_outputs["train_csv"],
        reducedTest_csv=temp_outputs["test_csv"],
        output_ottim_csv=temp_outputs["optim_csv"],
        output_json=temp_outputs["fs_json"],
        target_column="target",
        percTest=0.30,
        allowance=1,
        seed=42,
        percSelected=0.20,
        alpha_computations=5  # Ridotto a 5 per velocizzare i test automatici
    )

    import json
    with open(temp_outputs["fs_json"], 'r') as f:
        data = json.load(f)

    vector = data["selected_vector"]
    assert isinstance(vector, list), "Il vettore selezionato non è una lista."
    assert all(val in [0, 1] for val in vector), "Il vettore contiene valori diversi da 0 o 1."


def test_number_of_selected_features_is_about_20_percent(dummy_dataset_path, temp_outputs):
    """5. Verify that the number of selected features is approximately 20% (with allowance)."""
    import json
    with open(temp_outputs["fs_json"], 'r') as f:
        data = json.load(f)

    n_sel = data["n_selected"]
    target_k = data["target_k"]
    allowance = data["allowance"]

    assert abs(n_sel - target_k) <= allowance, "Il numero di feature selezionate è fuori dalla tolleranza consentita."


# --- 3. Model Tests ---
def test_training_produces_saved_model(temp_outputs):
    """6. Verify that training produces a saved model (.joblib file)."""
    train(
        classifier="random_forest",
        reducedTrain_csv=temp_outputs["train_csv"],
        target_column="target",
        model_path=temp_outputs["model_path"],
        metrics_json=temp_outputs["metrics_json"],
        seed=42
    )

    assert os.path.exists(temp_outputs["model_path"]), "Il file del modello .joblib non è stato creato."


def test_prediction_produces_csv_with_required_columns(temp_outputs):
    """7. Verify that prediction produces a CSV file with the required columns."""
    predict(
        reduced_Test_csv=temp_outputs["test_csv"],
        target_column="target",
        model_path=temp_outputs["model_path"],
        predictions_csv=temp_outputs["preds_csv"],
        classif_stats_json=temp_outputs["stats_json"]
    )

    assert os.path.exists(temp_outputs["preds_csv"]), "Il file CSV delle predizioni non è stato creato."

    df_preds = pd.read_csv(temp_outputs["preds_csv"])
    required_columns = ["row_n", "target", "prediction", "score"]

    for col in required_columns:
        assert col in df_preds.columns, f"La colonna obbligatoria '{col}' manca nel file delle predizioni."