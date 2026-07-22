import pandas as pd
import numpy as np
import json
import time
import argparse
from sklearn.preprocessing import StandardScaler


def fit_normalize(
        input_csv: str,
        target_column: str,
        normalized_csv: str,
        outInitalRes_json: str,
        minPercValid: float = 0.05
):
    t0 = time.time()
    df = pd.read_csv(input_csv)
    t_read = time.time() - t0

    t1 = time.time()
    n_input_features = df.shape[1] - 1

    # Separare la colonna target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Eliminare feature con percentuale di dati validi (non nulli e != 0) inferiore alla soglia
    valid_mask = X.notna() & (X != 0)
    valid_perc = valid_mask.mean()

    cols_to_keep = valid_perc[valid_perc >= minPercValid].index.tolist()
    cols_to_drop = valid_perc[valid_perc < minPercValid].index.tolist()

    X_kept = X[cols_to_keep]

    # Normalizzazione Z-score
    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X_kept), columns=cols_to_keep)

    # Ricombinare con il target e salvare
    df_norm = pd.concat([X_norm, y.reset_index(drop=True)], axis=1)
    df_norm.to_csv(normalized_csv, index=False)

    t_proc = time.time() - t1

    # Creazione JSON
    stats = {
        "n_input_features": n_input_features,
        "n_kept_features": len(cols_to_keep),
        "dataset_size": len(df),
        "dataset_input_time": round(t_read, 2),
        "dataset_processing_time": round(t_proc, 2),
        "dropped_feature_names": cols_to_drop
    }

    with open(outInitalRes_json, 'w') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-perc-valid", type=float, default=0.05)
    args = parser.parse_args()

    fit_normalize(args.input, args.target, args.out_data, args.out_json, args.min_perc_valid)