import pandas as pd
import numpy as np
import json
import time
import argparse
import csv
from scipy.stats import spearmanr
import neal  # Richiede: pip install neal


def select_features(
        normalized_csv: str,
        reducedTrain_csv: str,
        reducedTest_csv: str,
        output_ottim_csv: str,
        output_json: str,
        target_column: str,
        percTest: float = 0.30,
        allowance: int = 1,
        seed: int = 42,
        percSelected: float = 0.20,
        alpha_computations: int = 100
):
    df = pd.read_csv(normalized_csv)
    y = df[target_column].values
    X_df = df.drop(columns=[target_column])
    features = X_df.columns.tolist()
    m = len(features)

    # Calcolo target k e tolleranza
    target_k = int(round(percSelected * m))
    min_k = target_k - allowance
    max_k = target_k + allowance

    t0_q = time.time()
    # Calcolo delle correlazioni (in modulo come da specifiche)
    corr_matrix, _ = spearmanr(X_df.values)
    corr_matrix = np.abs(corr_matrix)

    rho_V = []
    for i in range(m):
        corr, _ = spearmanr(X_df.iloc[:, i].values, y)
        rho_V.append(abs(corr) if not np.isnan(corr) else 0.0)
    rho_V = np.array(rho_V)
    t_q_creation = time.time() - t0_q

    sampler = neal.SimulatedAnnealingSampler()

    optimizations_log = []
    alphas = np.linspace(0, 1, alpha_computations)

    best_vector = None
    best_alpha = None
    best_k = 0
    opt_times = []

    # Ricerca del parametro alpha
    for alpha in alphas:
        t0_opt = time.time()

        # Costruzione matrice QUBO (minimizzare: -alpha * Influenza + (1-alpha) * Indipendenza)
        Q = np.zeros((m, m))
        for i in range(m):
            Q[i, i] = -alpha * rho_V[i]
            for j in range(i + 1, m):
                val = (1 - alpha) * corr_matrix[i, j]
                Q[i, j] = val
                Q[j, i] = val

        # Risoluzione
        response = sampler.sample_qubo(Q, seed=seed)
        sample = response.first.sample
        energy = response.first.energy

        t_opt = time.time() - t0_opt
        opt_times.append(t_opt)

        selected_vars = [sample[i] for i in range(m)]
        current_k = sum(selected_vars)

        optimizations_log.append([alpha, t_opt, current_k, energy])

        # Controllo tolleranza
        if min_k <= current_k <= max_k:
            best_vector = selected_vars
            best_alpha = alpha
            best_k = current_k
            break

        # Se non lo troviamo, salviamo il più vicino per robustezza
        if best_vector is None or abs(current_k - target_k) < abs(best_k - target_k):
            best_vector = selected_vars
            best_alpha = alpha
            best_k = current_k

    # Salvataggio log ottimizzazioni
    with open(output_ottim_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'optimization_time', 'n_features', 'cost_value'])
        writer.writerows(optimizations_log)

    # Taglio del dataset per il training/test set
    cut_idx = int(len(df) * (1 - percTest))
    train_df = df.iloc[:cut_idx]
    test_df = df.iloc[cut_idx:]

    # Riduzione delle feature
    selected_feature_names = [features[i] for i in range(m) if best_vector[i] == 1]
    cols_to_keep_final = selected_feature_names + [target_column]

    train_reduced = train_df[cols_to_keep_final]
    test_reduced = test_df[cols_to_keep_final]

    train_reduced.to_csv(reducedTrain_csv, index=False)
    test_reduced.to_csv(reducedTest_csv, index=False)

    # Json output
    out_data = {
        "n_features": m,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": best_k,
        "alpha": float(best_alpha) if best_alpha else None,
        "selected_vector": best_vector,
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(optimizations_log),
        "percTest": percTest,
        "training_dataset_size": len(train_reduced),
        "test_dataset_size": len(test_reduced),
        "q_matrix_creation_time": round(t_q_creation, 2),
        "mean_optimization_time": round(np.mean(opt_times), 3),
        "std_dev_optimization_time": round(np.std(opt_times), 3)
    }

    with open(output_json, 'w') as f:
        json.dump(out_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-normalized", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-test", required=True)
    parser.add_argument("--out-optimizations", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--perc-selected", type=float, default=0.20)
    parser.add_argument("--allowance", type=int, default=1)
    parser.add_argument("--perc-test", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-computations", type=int, default=100)
    args = parser.parse_args()

    select_features(args.in_normalized, args.out_train, args.out_test, args.out_optimizations, args.out_json,
                    args.target, args.perc_test, args.allowance, args.seed, args.perc_selected, args.alpha_computations)