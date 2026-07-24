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
        allowance=3,          # Increased allowance slightly for heuristic stability
        seed=42,
        alpha_computations=10 # Increased from 5 to find a better alpha weight
    )
    # ... rest of the test