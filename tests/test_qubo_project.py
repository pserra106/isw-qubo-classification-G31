# Test automatici per la pipeline QUBO project eseguibili con pytest

import os
import json
from pathlib import Path  # <--- AGGIUNGI QUESTA RIGA
import pandas as pd
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict


def test_pipeline_end_to_end(tmp_path):
    # 0. Creazione del piccolo dataset di esempio in data/sample_test_dataset.csv (o temporaneo)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "sample_test_dataset.csv"

    # Dataset con almeno 10% di target 1 (qui 3 su 10 = 30%)
    df_sample = pd.DataFrame({
        "feat1": [1.0, 0.0, 2.5, 0.0, 3.1, 0.5, 1.2, 0.0, 2.2, 1.0],
        "feat2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Quasi vuota / zeri
        "feat3": [1.2, 3.4, 2.1, 1.1, 4.5, 2.2, 3.1, 1.0, 2.9, 3.0],
        "target": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
    })
    df_sample.to_csv(data_path, index=False)

    norm_csv = tmp_path / "normalized.csv"
    prep_json = tmp_path / "prep.json"

    # 1. Test Preprocessing (verifica colonne numeriche e gestione nulli/zeri)
    fit_normalize(str(data_path), "target", str(norm_csv), str(prep_json), minPercValid=0.1)
    df_norm = pd.read_csv(norm_csv)
    assert all(df_norm.dtypes != 'object'), "Il preprocessing deve produrre solo colonne numeriche"

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    optim_csv = tmp_path / "optim.csv"
    fs_json = tmp_path / "fs.json"

    # 2. Test Feature Selection (verifica vettore binario e risultati QUBO)
    select_features(
        str(norm_csv), str(train_csv), str(test_csv),
        str(optim_csv), str(fs_json), "target",
        percTest=0.4, allowance=1, percSelected=0.5
    )

    with open(fs_json, 'r', encoding='utf-8') as f:
        fs_data = json.load(f)

    # Verifica che il vettore selezionato sia binario (composto solo da 0 e 1)
    selected_vector = fs_data.get("selected_vector", [])
    assert all(v in [0, 1] for v in selected_vector), "Il vettore delle feature deve essere binario"

    df_train = pd.read_csv(train_csv)
    assert "target" in df_train.columns

    model_path = tmp_path / "model.joblib"
    metrics_json = tmp_path / "metrics.json"

    # 3. Test Training (verifica salvataggio del modello)
    train("random_forest", str(train_csv), "target", str(model_path), str(metrics_json))
    assert os.path.exists(model_path), "Il training deve salvare il file binario del modello (.joblib)"

    preds_csv = tmp_path / "preds.csv"
    stats_json = tmp_path / "stats.json"

    # 4. Test Predict (verifica file CSV con colonne richieste)
    predict(str(test_csv), "target", str(model_path), str(preds_csv), str(stats_json))
    assert os.path.exists(preds_csv), "Il modulo di predizione deve generare il file CSV dei risultati"

    df_preds = pd.read_csv(preds_csv)
    expected_columns = ["row_n", "target", "prediction", "score"]
    for col in expected_columns:
        assert col in df_preds.columns, f"Il file delle predizioni deve contenere la colonna '{col}'"