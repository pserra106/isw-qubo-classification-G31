# This module builds the QUBO matrix based on Spearman correlations
# and varies alpha iteratively to find the desired number of features (K).

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


def select_features(
        normalized_csv: str,
        reducedTrain_csv: str,
        reducedTest_csv: str,
        output_ottim_csv: str,
        output_json: str,
        target_column: str,
        percTest: float = 0.30,
        percSelected: float = 0.20,
        allowance: int = 1,
        seed: int = 42,
        alpha_computations: int = 100
):
    np.random.seed(seed)
    start_time = time.time()

    # 1. Lettura del dataset normalizzato
    df = pd.read_csv(normalized_csv)

    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel dataset.")

    y = df[target_column].values
    X = df.drop(columns=[target_column])

    feature_names = list(X.columns)
    m_features = len(feature_names)

    target_k = round(percSelected * m_features)
    min_k = target_k - allowance
    max_k = target_k + allowance

    # 2. Divisione iniziale in training e test set per evitare data leakage durante il QUBO
    total_samples = len(df)
    n_test = int(total_samples * percTest)
    n_train = total_samples - n_test

    X_train = X.iloc[:n_train]
    y_train = y[:n_train]
    X_test = X.iloc[n_train:]
    y_test = y[n_train:]

    # 3. Calcolo delle correlazioni di Spearman basato ESCLUSIVAMENTE sul training set
    t_q_start = time.time()
    n = m_features
    corr_v = np.zeros(n)
    for i, col in enumerate(feature_names):
        val, _ = spearmanr(X_train[col].values, y_train)
        corr_v[i] = abs(val) if not np.isnan(val) else 0.0

    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                val, _ = spearmanr(X_train.iloc[:, i].values, X_train.iloc[:, j].values)
                c = abs(val) if not np.isnan(val) else 0.0
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c
    q_matrix_creation_time = time.time() - t_q_start

    # 4. Ottimizzazione variando alpha per trovare il numero K di feature tramite QUBO
    alphas = np.linspace(0.01, 0.99, alpha_computations)
    optimization_records = []
    opt_times = []

    best_x = None
    best_alpha = 0.5
    best_diff = float('inf')

    for alpha in alphas:
        t_opt_start = time.time()

        # Costruzione della matrice QUBO Q: f(x) = - x^T Q x
        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = alpha * corr_v[i]
            for j in range(n):
                if i != j:
                    Q[i, j] = -(1.0 - alpha) * corr_matrix[i, j]

        # Contributo energetico marginale per ogni feature basato sulla matrice Q
        marginal_gains = np.diag(Q) + 2.0 * np.sum(Q, axis=1)
        sorted_indices = np.argsort(marginal_gains)[::-1]

        active_ratio = np.clip(percSelected * (0.5 + alpha), 0.05, 0.95)
        k_candidate = int(np.clip(round(n * active_ratio), 1, n))

        x = np.zeros(n, dtype=int)
        x[sorted_indices[:k_candidate]] = 1

        cost_val = - float(x.T @ Q @ x)
        opt_time = time.time() - t_opt_start
        opt_times.append(opt_time)

        n_ones = int(np.sum(x))
        optimization_records.append({
            "alpha": round(float(alpha), 4),
            "time": round(opt_time, 4),
            "n_features": n_ones,
            "cost": round(cost_val, 4)
        })

        diff = abs(n_ones - target_k)
        if diff < best_diff:
            best_diff = diff
            best_x = x
            best_alpha = alpha
            if min_k <= n_ones <= max_k:
                if diff == 0:
                    break

    if best_x is None:
        best_x = np.zeros(n, dtype=int)
        best_x[:target_k] = 1

    selected_indices = np.where(best_x == 1)[0]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    # Salvataggio del log delle ottimizzazioni in CSV
    Path(output_ottim_csv).parent.mkdir(parents=True, exist_ok=True)
    df_opt = pd.DataFrame(optimization_records)
    df_opt.to_csv(output_ottim_csv, index=False)

    # 5. Applicazione della riduzione delle feature sia al Training che al Test set
    X_train_reduced = X_train.iloc[:, selected_indices].copy()
    X_train_reduced[target_column] = y_train

    X_test_reduced = X_test.iloc[:, selected_indices].copy()
    X_test_reduced[target_column] = y_test

    Path(reducedTrain_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(reducedTest_csv).parent.mkdir(parents=True, exist_ok=True)
    X_train_reduced.to_csv(reducedTrain_csv, index=False)
    X_test_reduced.to_csv(reducedTest_csv, index=False)

    # 6. Salvataggio del report JSON finale
    result_data = {
        "n_features": int(m_features),
        "target_ratio": float(np.mean(y)),
        "target_k": int(target_k),
        "allowance": int(allowance),
        "n_selected": int(len(selected_feature_names)),
        "alpha": round(float(best_alpha), 4),
        "selected_vector": best_x.tolist(),
        "selected_feature_names": selected_feature_names,
        "algorithm": "greedy_qubo_spearman",
        "seed": int(seed),
        "alpha_computations": len(optimization_records),
        "percTest": float(percTest),
        "training_dataset_size": int(len(X_train)),
        "test_dataset_size": int(len(X_test)),
        "q_matrix_creation_time": round(q_matrix_creation_time, 4),
        "mean_optimization_time": round(float(np.mean(opt_times)), 4),
        "std_dev_optimization_time": round(float(np.std(opt_times)), 4)
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUBO Feature Selection")
    parser.add_argument("--in-normalized", required=True, help="Input normalized dataset CSV path")
    parser.add_argument("--out-train", required=True, help="Output reduced training CSV path")
    parser.add_argument("--out-test", required=True, help="Output reduced test CSV path")
    parser.add_argument("--out-optimizations", required=True, help="Output optimizations CSV log path")
    parser.add_argument("--out-json", required=True, help="Output summary JSON path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--perc-test", type=float, default=0.30, help="Percentage of test data")
    parser.add_argument("--perc-selected", type=float, default=0.20, help="Target percentage of features to select")
    parser.add_argument("--allowance", type=int, default=1, help="Allowed tolerance on K features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha-computations", type=int, default=100, help="Number of alpha iterations")

    args = parser.parse_args()
    select_features(
        normalized_csv=args.in_normalized,
        reducedTrain_csv=args.out_train,
        reducedTest_csv=args.out_test,
        output_ottim_csv=args.out_optimizations,
        output_json=args.out_json,
        target_column=args.target,
        percTest=args.perc_test,
        percSelected=args.perc_selected,
        allowance=args.allowance,
        seed=args.seed,
        alpha_computations=args.alpha_computations
    )