# Questo modulo gestisce la lettura del dataset, la rimozione delle colonne
# con troppi valori nulli o zeri, la normalizzazione z-score e la
# suddivisione iniziale in training e test set.

import argparse
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path


def fit_normalize(
        input_csv: str,
        target_column: str,
        normalized_csv: str,
        outInitalRes_json: str,
        minPercValid: float = 0.05,
        train_csv: str = None,
        test_csv: str = None,
        test_split_perc: float = 0.2
):
    start_time = time.time()

    # 1. Lettura del dataset
    df = pd.read_csv(input_csv)
    input_time = time.time() - start_time

    n_input_features = df.shape[1] - 1 if target_column in df.columns else df.shape[1]

    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # 2. Eliminazione colonne con percentuale di valori validi inferiore a minPercValid
    n_rows = len(X)
    dropped_features = []
    retained_cols = []

    for col in X.columns:
        valid_count = ((X[col].notna()) & (X[col] != 0)).sum()
        valid_perc = valid_count / n_rows if n_rows > 0 else 0
        if valid_perc < minPercValid:
            dropped_features.append(col)
        else:
            retained_cols.append(col)

    X_filtered = X[retained_cols]

    # 3. Normalizzazione z-score delle feature
    X_normalized = (X_filtered - X_filtered.mean()) / (X_filtered.std(ddof=0) + 1e-8)
    X_normalized = X_normalized.fillna(0)  # Gestione eventuali divisioni per zero

    # Ricostruzione dataframe finale con il target
    df_normalized = X_normalized.copy()
    df_normalized[target_column] = y.values

    # Salvataggio del dataset normalizzato completo
    Path(normalized_csv).parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_csv(normalized_csv, index=False)

    # 3.5 Divisione opzionale in training set e test set (se richiesto dai parametri)
    if train_csv and test_csv:
        n_samples = len(df_normalized)
        M = int(n_samples * (1 - test_split_perc))  # I primi M campioni sono il training set

        df_train = df_normalized.iloc[:M]
        df_test = df_normalized.iloc[M:]

        Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(train_csv, index=False)

        Path(test_csv).parent.mkdir(parents=True, exist_ok=True)
        df_test.to_csv(test_csv, index=False)

    processing_time = time.time() - start_time - input_time

    # 4. Salvataggio statistiche in JSON (rispecchia esattamente le chiavi d'esempio della specifica)
    stats = {
        "n_input_features": int(n_input_features),
        "n_kept_features": int(len(retained_cols)),
        "dataset_size": int(len(df)),
        "dataset_input_time": round(input_time, 4),
        "dataset_processing_time": round(processing_time, 4),
        "dropped_feature_names": dropped_features
    }

    Path(outInitalRes_json).parent.mkdir(parents=True, exist_ok=True)
    with open(outInitalRes_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Dataset")
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out-data", required=True, help="Output normalized CSV path")
    parser.add_argument("--out-json", required=True, help="Output JSON stats path")
    parser.add_argument("--min-perc-valid", type=float, default=0.05, help="Minimum percentage of valid non-zero data")
    parser.add_argument("--test-split-perc", type=float, default=0.2, help="Percentage of test set")
    parser.add_argument("--out-train", default=None, help="Output training CSV path")
    parser.add_argument("--out-test", default=None, help="Output test CSV path")

    args = parser.parse_args()

    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid,
        train_csv=args.out_train,
        test_csv=args.out_test,
        test_split_perc=args.test_split_perc
    )