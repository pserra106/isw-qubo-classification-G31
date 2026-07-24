import os
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


def _solve_qubo_sa(Q: np.ndarray, seed: int = 42, sweeps: int = 2000, 
                   temp_start: float = 10.0, temp_end: float = 0.01) -> np.ndarray:
    """
    Solves QUBO problem min (x^T Q x) using Simulated Annealing.
    
    Parameters
    ----------
    Q : np.ndarray
        Symmetric QUBO matrix of size (m, m).
    seed : int
        Random seed for reproducibility.
    sweeps : int
        Number of Metropolis-Hastings iterations.
    temp_start : float
        Initial temperature for annealing schedule.
    temp_end : float
        Final temperature for annealing schedule.
        
    Returns
    -------
    best_x : np.ndarray
        Optimal binary solution vector.
    """
    rng = np.random.default_rng(seed)
    m = Q.shape[0]
    
    # Initialize random binary vector
    x = rng.integers(0, 2, size=m, dtype=np.int8)
    
    # Pre-calculate symmetric matrix interactions: S = Q + Q^T - diag(Q)
    # E(x) = x^T Q x
    S = Q + Q.T
    np.fill_diagonal(S, np.diagonal(Q))
    
    # Compute current energy
    current_energy = float(x @ Q @ x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Exponential cooling schedule
    temperatures = np.geomspace(temp_start, temp_end, num=sweeps)
    
    for T in temperatures:
        # Pick random variable to flip
        i = rng.integers(0, m)
        
        # Calculate energy delta for flipping x[i]
        # x_i_new = 1 - x[i] => delta_x = 1 - 2*x[i]
        delta_xi = 1 - 2 * x[i]
        
        # Fast incremental delta energy computation
        delta_E = delta_xi * (S[i] @ x - S[i, i] * x[i] + Q[i, i])
        
        # Accept or reject move
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            x[i] = 1 - x[i]
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                
    return best_x


def select_features(
    normalized_csv: str,          # Input dataset name
    reducedTrain_csv: str,        # Name of output training dataset with reduced feat.
    reducedTest_csv: str,         # Name of output test dataset with reduced features
    output_ottim_csv: str,        # Name of output optimization data varying alpha
    output_json: str,             # Name of output statistics and data file
    target_column: str,           # Column name of target
    percTest: float = 0.30,       # % of test data with respect to the dataset size
    percSelected: float = 0.20,   # Percentage of features to select
    allowance: int = 1,           # Allowance of features to select
    seed: int = 42,               # Seed for random repeatability
    alpha_computations: int = 100 # Max. n. of optimizations varying alpha
) -> None:
    """
    Performs feature selection via QUBO optimization by varying alpha to reach target K features.
    """
    np.random.seed(seed)
    
    # 1. Read input dataset
    df = pd.read_csv(normalized_csv)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    feature_cols = [col for col in df.columns if col != target_column]
    m = len(feature_cols)
    
    # Target K number of features
    target_k = int(round(percSelected * m))
    min_k = max(1, target_k - allowance)
    max_k = min(m, target_k + allowance)
    
    # 2. Build Spearman Correlation Matrices (Relevance & Redundancy)
    t_q_start = time.perf_counter()
    
    # Extract feature matrix X and target y
    X = df[feature_cols].values
    y = df[target_column].values
    
    # Feature-Target Relevance: s_i = |Spearman(f_i, y)|
    s = np.zeros(m)
    for i in range(m):
        corr, _ = spearmanr(X[:, i], y)
        s[i] = np.abs(corr) if not np.isnan(corr) else 0.0
        
    # Feature-Feature Redundancy: r_ij = |Spearman(f_i, f_j)|
    r_corr, _ = spearmanr(X)
    if m == 1:
        r_corr = np.array([[1.0]])
    r = np.nan_to_num(np.abs(r_corr), nan=0.0)
    
    q_matrix_creation_time = time.perf_counter() - t_q_start
    
    # 3. Optimization Loop Varying Alpha
    # Binary Search / Adaptive Search Strategy over alpha in [0, 1]
    alpha_low, alpha_high = 0.0, 1.0
    
    optimizations_log = []
    opt_times = []
    
    best_solution = None
    best_k_diff = float('inf')
    
    for iteration in range(alpha_computations):
        # Determine alpha for current iteration
        if iteration == 0:
            alpha = 0.5
        elif iteration == 1:
            alpha = 0.0
        elif iteration == 2:
            alpha = 1.0
        else:
            alpha = (alpha_low + alpha_high) / 2.0

        # Construct QUBO matrix Q(alpha)
        # Minimize: E(x) = - alpha * sum(s_i * x_i) + (1 - alpha) * sum_{i < j}(r_ij * x_i * x_j)
        Q = np.zeros((m, m))
        
        # Diagonal elements (Relevance)
        np.fill_diagonal(Q, -alpha * s)
        
        # Off-diagonal elements (Redundancy)
        for i in range(m):
            for j in range(i + 1, m):
                Q[i, j] = (1.0 - alpha) * r[i, j]
                
        # Make Q symmetric for solver stability
        Q_sym = 0.5 * (Q + Q.T)

        # Solve QUBO via Simulated Annealing
        t_opt_start = time.perf_counter()
        opt_seed = seed + iteration
        x_sol = _solve_qubo_sa(Q_sym, seed=opt_seed)
        t_opt_end = time.perf_counter()
        
        opt_time = t_opt_end - t_opt_start
        opt_times.append(opt_time)
        
        n_selected = int(np.sum(x_sol))
        cost_val = float(x_sol @ Q_sym @ x_sol)
        
        optimizations_log.append({
            "alpha": float(alpha),
            "optimization_time": float(opt_time),
            "n_features": n_selected,
            "cost_function_value": cost_val,
            "vector": x_sol.copy()
        })
        
        # Update best solution tracking
        k_diff = abs(n_selected - target_k)
        if k_diff < best_k_diff:
            best_k_diff = k_diff
            best_solution = optimizations_log[-1]

        # Check if tolerance criteria is met
        if min_k <= n_selected <= max_k:
            best_solution = optimizations_log[-1]
            break
            
        # Adjust search range for binary search
        if n_selected < min_k:
            alpha_low = alpha  # Needs higher relevance weight (alpha)
        else:
            alpha_high = alpha # Needs lower redundancy weight (1 - alpha)

    # 4. Save Optimization Log CSV
    optim_df = pd.DataFrame([
        {
            "alpha": row["alpha"],
            "optimization_time": row["optimization_time"],
            "n_features": row["n_features"],
            "cost_function_value": row["cost_function_value"]
        }
        for row in sorted(optimizations_log, key=lambda x: x["alpha"])
    ])
    
    # Ensure target directory exists
    if os.path.dirname(output_ottim_csv):
        os.makedirs(os.path.dirname(output_ottim_csv), exist_ok=True)
    optim_df.to_csv(output_ottim_csv, index=False)

    # 5. Extract Final Selected Features & Reduce Dataset
    final_vector = best_solution["vector"]
    selected_indices = np.where(final_vector == 1)[0]
    selected_feature_names = [feature_cols[i] for i in selected_indices]
    
    # Filter dataset: Selected features + Target column at the end
    reduced_cols = selected_feature_names + [target_column]
    df_reduced = df[reduced_cols]
    
    # Train / Test split
    df_train, df_test = train_test_split(
        df_reduced, 
        test_size=percTest, 
        random_state=seed
    )
    
    # Save reduced CSV files
    if os.path.dirname(reducedTrain_csv):
        os.makedirs(os.path.dirname(reducedTrain_csv), exist_ok=True)
    if os.path.dirname(reducedTest_csv):
        os.makedirs(os.path.dirname(reducedTest_csv), exist_ok=True)
        
    df_train.to_csv(reducedTrain_csv, index=False)
    df_test.to_csv(reducedTest_csv, index=False)

    # 6. Save JSON Statistics
    json_stats = {
        "n_features": m,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": int(best_solution["n_features"]),
        "alpha": float(best_solution["alpha"]),
        "selected_vector": final_vector.tolist(),
        "selected_feature_names": selected_feature_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(optimizations_log),
        "percTest": percTest,
        "training_dataset_size": len(df_train),
        "test_dataset_size": len(df_test),
        "q_matrix_creation_time": round(float(q_matrix_creation_time), 4),
        "mean_optimization_time": round(float(np.mean(opt_times)), 4),
        "std_dev_optimization_time": round(float(np.std(opt_times)), 4)
    }

    if os.path.dirname(output_json):
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_stats, f, indent=4)