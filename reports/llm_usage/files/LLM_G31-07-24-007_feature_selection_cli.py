if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Feature selection via QUBO")
    
    parser.add_argument("--in-normalized", type=str, required=True, help="Input normalized dataset (.csv)")
    parser.add_argument("--out-train", type=str, required=True, help="Output training dataset (.csv)")
    parser.add_argument("--out-test", type=str, required=True, help="Output test dataset (.csv)")
    parser.add_argument("--out-optimizations", type=str, required=True, help="Output optimization data varying alpha (.csv)")
    parser.add_argument("--out-json", type=str, required=True, help="Output statistics and data file (.json)")
    parser.add_argument("--target", type=str, required=True, help="Column name of the target")
    parser.add_argument("--perc-selected", type=float, default=0.20, help="Percentage of features to select")
    parser.add_argument("--allowance", type=int, default=1, help="Allowance of features to select")
    parser.add_argument("--perc-test", type=float, default=0.30, help="Percentage of test data")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random repeatability")
    parser.add_argument("--alpha-computations", type=int, default=100, help="Max n. of optimizations varying alpha")

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