import os
import time
import json
import argparse
import pandas as pd
import numpy as np


def fit_normalize(
        input_csv: str,
        target_column: str,
        normalized_csv: str,
        outInitalRes_json: str,
        minPercValid: float = 0.05,
):
    """
    Reads the dataset, drops largely empty/zero columns, standardizes the remaining
    features (z-score), handles missing values, and saves the output and metadata.
    """
    # 1. Measure input dataset loading time
    start_input_time = time.perf_counter()
    df = pd.read_csv(input_csv)
    dataset_input_time = time.perf_counter() - start_input_time

    # 2. Measure processing time
    start_processing_time = time.perf_counter()

    dataset_size = len(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Separate the target column from the features
    target_series = df[target_column]
    features_df = df.drop(columns=[target_column])
    n_input_features = len(features_df.columns)

    # 3. Eliminate empty or nearly empty columns
    # A value is valid if it is not NaN AND not equal to 0
    is_valid_mask = features_df.notna() & (features_df != 0)

    # Calculate the percentage of valid entries for each column
    valid_percentages = is_valid_mask.sum() / dataset_size

    # Identify which columns to keep and which to drop based on minPercValid
    cols_to_keep = valid_percentages[valid_percentages >= minPercValid].index.tolist()
    dropped_feature_names = valid_percentages[valid_percentages < minPercValid].index.tolist()

    # Filter the dataframe
    features_df = features_df[cols_to_keep]
    n_kept_features = len(cols_to_keep)

    # 4. Normalize the features (z-score standardization)
    # Calculate mean and std deviation, ignoring NaNs
    means = features_df.mean()
    stds = features_df.std(ddof=1)

    # Safeguard against division by zero for constant columns
    stds = stds.replace(0, 1.0)

    # Apply z-score
    normalized_features = (features_df - means) / stds

    # Manage missing values: replace NaNs with 0 (which is the mean after standardization)
    normalized_features = normalized_features.fillna(0)

    # 5. Re-attach the target column
    final_df = pd.concat([normalized_features, target_series], axis=1)

    dataset_processing_time = time.perf_counter() - start_processing_time

    # Ensure output directories exist before saving
    from pathlib import Path

    Path(normalized_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(outInitalRes_json).parent.mkdir(parents=True, exist_ok=True)

    # 6. Save outputs
    # Save the normalized dataframe as CSV
    final_df.to_csv(normalized_csv, index=False)

    # Save the JSON metadata
    json_data = {
        "n_input_features": n_input_features,
        "n_kept_features": n_kept_features,
        "dataset_size": dataset_size,
        "dataset_input_time": round(dataset_input_time, 2),
        "dataset_processing_time": round(dataset_processing_time, 2),
        "dropped_feature_names": dropped_feature_names
    }

    with open(outInitalRes_json, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    # Mandatory command-line interface
    parser = argparse.ArgumentParser(description="Phase 1: Dataset preprocessing")

    parser.add_argument("--input", type=str, required=True, help="Input dataset name (.csv)")
    parser.add_argument("--target", type=str, required=True, help="Column name of the target")
    parser.add_argument("--out-data", type=str, required=True, help="Name of output normalized dataset (.csv)")
    parser.add_argument("--out-json", type=str, required=True, help="Name of output statistics and data file (.json)")
    parser.add_argument("--min-perc-valid", type=float, default=0.05,
                        help="Minimum percentage of valid non-zero data for a column")

    args = parser.parse_args()

    # Execute the core function
    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid
    )