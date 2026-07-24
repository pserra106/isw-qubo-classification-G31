import os
import time
import json
import argparse
import joblib
import pandas as pd
import numpy as np

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


def train(
        classifier: str,  # classifier to use
        reducedTrain_csv: str,  # training dataset
        target_column: str,  # target column name
        model_path: str,  # saved trained classifier
        metrics_json: str,  # file with training statistics
        seed: int = 42,
):
    """
    Trains a selected binary classifier on the reduced training dataset.
    """
    valid_classifiers = ["random_forest", "logistic_regression", "gradient_boosting"]
    if classifier not in valid_classifiers:
        raise ValueError(f"Invalid classifier '{classifier}'. Must be one of: {valid_classifiers}")

    # 1. Read input dataset and measure input time
    t0_input = time.perf_counter()
    df_train = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.perf_counter() - t0_input

    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # 2. Extract features and target
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    n_samples = len(df_train)
    n_features = len(X_train.columns)

    target_1_count = int(y_train.sum())
    target_1_percentage = (target_1_count / n_samples) * 100.0

    # 3. Instantiate the selected model
    if classifier == "random_forest":
        clf = RandomForestClassifier(random_state=seed)
    elif classifier == "logistic_regression":
        clf = LogisticRegression(random_state=seed, max_iter=1000)
    elif classifier == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=seed)

    # 4. Train the classifier and measure training time
    t0_train = time.perf_counter()
    clf.fit(X_train, y_train)
    training_time = time.perf_counter() - t0_train

    # Attach the classifier name to the model object so predict() can retrieve it
    clf.custom_classifier_name_ = classifier

    # 5. Save the trained model
    if os.path.dirname(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    # 6. Save the metrics JSON
    metrics_data = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": os.path.basename(reducedTrain_csv),
        "target_column": target_column,
        "model_path": os.path.basename(model_path),
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": round(target_1_percentage, 2),
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }

    if os.path.dirname(metrics_json):
        os.makedirs(os.path.dirname(metrics_json), exist_ok=True)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)


def predict(
        reduced_Test_csv: str,  # Input test set
        target_column: str,  # Target column name
        model_path: str,  # saved trained classifier to use
        predictions_csv: str,  # Output predictions
        classif_stats_json: str,  # File with classification stats
):
    """
    Loads a trained classifier, makes predictions on the test dataset,
    and outputs predictions and evaluation statistics.
    """
    # 1. Load the dataset
    df_test = pd.read_csv(reduced_Test_csv)

    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X_test = df_test.drop(columns=[target_column])
    y_true = df_test[target_column]

    n_samples = len(df_test)
    target_1_count = int(y_true.sum())
    target_1_percentage = (target_1_count / n_samples) * 100.0

    # 2. Load the trained model
    clf = joblib.load(model_path)
    classifier_name = getattr(clf, "custom_classifier_name_", "unknown_classifier")

    # 3. Generate predictions and probability scores
    y_pred = clf.predict(X_test)

    # Attempt to get probabilities for the positive class (class 1)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        # Fallback if a model does not support predict_proba
        y_score = y_pred.astype(float)

    # 4. Save the predictions CSV
    predictions_df = pd.DataFrame({
        "row_n": range(n_samples),
        "target": y_true.values,
        "prediction": y_pred,
        "score": np.round(y_score, 4)
    })

    if os.path.dirname(predictions_csv):
        os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False)

    # 5. Calculate Classification Quality Statistics
    acc = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, f1, and support for classes 0 and 1
    # labels=[0, 1] ensures that arrays consistently map index 0 to class 0, index 1 to class 1
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.0  # Occurs if only one class is present in y_true

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 6. Save the Stats JSON
    stats_data = {
        "classifier": classifier_name,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(target_1_percentage, 2),
        "accuracy": float(acc),
        "class_0": {
            "precision": float(precisions[0]),
            "recall": float(recalls[0]),
            "f1": float(f1_scores[0]),
            "support": int(supports[0])
        },
        "class_1": {
            "precision": float(precisions[1]),
            "recall": float(recalls[1]),
            "f1": float(f1_scores[1]),
            "support": int(supports[1])
        },
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }

    if os.path.dirname(classif_stats_json):
        os.makedirs(os.path.dirname(classif_stats_json), exist_ok=True)

    with open(classif_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 & 4: Classifier Training and Prediction")

    # Create subparsers for "train" and "predict" modes
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # ----- TRAIN SUBPARSER -----
    parser_train = subparsers.add_parser("train", help="Train the classifier")
    parser_train.add_argument("--classifier", type=str, required=True, help="Classifier to use (e.g., random_forest)")
    parser_train.add_argument("--in-reduced", type=str, required=True, help="Training dataset (.csv)")
    parser_train.add_argument("--target", type=str, required=True, help="Target column name")
    parser_train.add_argument("--out-model", type=str, required=True, help="Path to save trained classifier (.joblib)")
    parser_train.add_argument("--out-metrics", type=str, required=True, help="Path to save training statistics (.json)")
    parser_train.add_argument("--seed", type=int, default=42, help="Seed for random reproducibility")

    # ----- PREDICT SUBPARSER -----
    parser_predict = subparsers.add_parser("predict", help="Generate predictions using a trained classifier")
    parser_predict.add_argument("--input-testset", type=str, required=True, help="Input test set (.csv)")
    parser_predict.add_argument("--target", type=str, required=True, help="Target column name")
    parser_predict.add_argument("--model", type=str, required=True, help="Saved trained classifier to use (.joblib)")
    parser_predict.add_argument("--out-predictions", type=str, required=True, help="Output predictions (.csv)")
    parser_predict.add_argument("--out-stats", type=str, required=True, help="File with classification stats (.json)")

    args = parser.parse_args()

    # Dispatch to the appropriate function based on the CLI sub-command
    if args.command == "train":
        train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=args.out_model,
            metrics_json=args.out_metrics,
            seed=args.seed
        )
    elif args.command == "predict":
        predict(
            reduced_Test_csv=args.input_testset,
            target_column=args.target,
            model_path=args.model,
            predictions_csv=args.out_predictions,
            classif_stats_json=args.out_stats
        )

