# 5. Extract Final Selected Features & Reduce Dataset
    final_vector = best_solution["vector"]
    selected_indices = np.where(final_vector == 1)[0]
    selected_feature_names = [feature_cols[i] for i in selected_indices]

    # Filter dataset: Selected features + Target column at the end
    reduced_cols = selected_feature_names + [target_column]
    df_reduced = df[reduced_cols]

    # Calculate M for strict sequential split ("Taglio Netto" - Section 7.3)
    n_total = len(df_reduced)
    n_test = int(round(n_total * percTest))
    m_train_idx = n_total - n_test

    # Sequential split (First M samples for train, remainder for test)
    df_train = df_reduced.iloc[:m_train_idx].copy()
    df_test = df_reduced.iloc[m_train_idx:].copy()

    # Save reduced CSV files
    if os.path.dirname(reducedTrain_csv):
        os.makedirs(os.path.dirname(reducedTrain_csv), exist_ok=True)
    if os.path.dirname(reducedTest_csv):
        os.makedirs(os.path.dirname(reducedTest_csv), exist_ok=True)

    df_train.to_csv(reducedTrain_csv, index=False)
    df_test.to_csv(reducedTest_csv, index=False)