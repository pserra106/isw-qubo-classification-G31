# Implementa i tre classificatori richiesti (incluso Random Forest) e
# le funzioni train e predict con i relativi output JSON e CSV.

import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

def get_classifier(name: str, seed: int):
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed)
    elif name == "logistic_regression":
        return LogisticRegression(random_state=seed, max_iter=1000)
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Classificatore non supportato: {name}.")

def train(
        classifier: str,
        reducedTrain_csv: str,
        target_column: str,
        model_path: str,
        metrics_json: str,
        seed: int = 42
):
    t_start = time.time()
    df = pd.read_csv(reducedTrain_csv)
    input_time = time.time() - t_start

    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel training set.")

    y = df[target_column].values
    X = df.drop(columns=[target_column])

    clf = get_classifier(classifier, seed)

    t_train_start = time.time()
    clf.fit(X, y)
    training_time = time.time() - t_train_start

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    target_1_count = int(np.sum(y == 1))
    target_1_percentage = float((target_1_count / len(y)) * 100) if len(y) > 0 else 0.0

    metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": str(reducedTrain_csv),
        "target_column": target_column,
        "model_path": str(model_path),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "target_1_percentage": round(target_1_percentage, 4),
        "dataset_input_time": round(input_time, 4),
        "training_time": round(training_time, 4)
    }

    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


def predict(
        reduced_Test_csv: str,
        target_column: str,
        model_path: str,
        predictions_csv: str,
        classif_stats_json: str
):
    df = pd.read_csv(reduced_Test_csv)
    if target_column not in df.columns:
        raise ValueError(f"Colonna target '{target_column}' non trovata nel test set.")

    y_true = df[target_column].values
    X_test = df.drop(columns=[target_column])

    clf = joblib.load(model_path)

    # Predizioni standard di probabilità e classi con soglia standard 0.5
    y_scores = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_test).astype(float)
    y_pred = clf.predict(X_test) if hasattr(clf, "predict") else (y_scores >= 0.5).astype(int)

    # Salvataggio predizioni CSV
    df_preds = pd.DataFrame({
        "row_n": range(len(df)),
        "target": y_true,
        "prediction": y_pred,
        "score": y_scores
    })
    Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(predictions_csv, index=False)

    # Calcolo metriche standard con etichette [0, 1]
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # Allineato all'ordine standard [0, 1]

    classifier_name = getattr(clf, "__class__", type(clf)).__name__.lower()
    if "randomforest" in classifier_name:
        classifier_str = "random_forest"
    elif "logistic" in classifier_name:
        classifier_str = "logistic_regression"
    elif "gradient" in classifier_name:
        classifier_str = "gradient_boosting"
    else:
        classifier_str = classifier_name

    stats = {
        "classifier": classifier_str,
        "n_samples": int(len(df)),
        "target_1_count": int(np.sum(y_true == 1)),
        "target_1_percentage": float((np.sum(y_true == 1) / len(y_true)) * 100) if len(y_true) > 0 else 0.0,
        "accuracy": acc,
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }

    Path(classif_stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(classif_stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training and Prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train parser
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--classifier", required=True)
    p_train.add_argument("--in-reduced", required=True)
    p_train.add_argument("--target", required=True)
    p_train.add_argument("--out-model", required=True)
    p_train.add_argument("--out-metrics", required=True)
    p_train.add_argument("--seed", type=int, default=42)

    # Predict parser
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--input-testset", required=True)
    p_pred.add_argument("--target", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--out-predictions", required=True)
    p_pred.add_argument("--out-stats", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args.classifier, args.in_reduced, args.target, args.out_model, args.out_metrics, args.seed)
    elif args.command == "predict":
        predict(args.input_testset, args.target, args.model, args.out_predictions, args.out_stats)