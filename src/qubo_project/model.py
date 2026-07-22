import pandas as pd
import numpy as np
import json
import time
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def train(
        classifier: str,
        reducedTrain_csv: str,
        target_column: str,
        model_path: str,
        metrics_json: str,
        seed: int = 42
):
    t0 = time.time()
    df = pd.read_csv(reducedTrain_csv)
    t_read = time.time() - t0

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Selezione del classificatore
    if classifier == "random_forest":
        clf = RandomForestClassifier(random_state=seed)
    elif classifier == "logistic_regression":
        clf = LogisticRegression(random_state=seed, max_iter=1000)
    elif classifier == "decision_tree":
        clf = DecisionTreeClassifier(random_state=seed)
    else:
        raise ValueError("Classificatore non supportato")

    t1 = time.time()
    clf.fit(X, y)
    t_train = time.time() - t1

    joblib.dump(clf, model_path)

    stats = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": reducedTrain_csv,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": len(df),
        "n_features": X.shape[1],
        "target_1_percentage": round((y == 1).mean() * 100, 2),
        "dataset_input_time": round(t_read, 2),
        "training_time": round(t_train, 2)
    }

    with open(metrics_json, 'w') as f:
        json.dump(stats, f, indent=4)


def predict(
        reduced_Test_csv: str,
        target_column: str,
        model_path: str,
        predictions_csv: str,
        classif_stats_json: str
):
    df = pd.read_csv(reduced_Test_csv)
    y_true = df[target_column]
    X = df.drop(columns=[target_column])

    clf = joblib.load(model_path)

    # Recupera il nome del classificatore dalla classe dell'oggetto
    clf_name = "random_forest" if isinstance(clf, RandomForestClassifier) else "other"

    y_pred = clf.predict(X)
    y_scores = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    # Salvataggio CSV predizioni
    pred_df = pd.DataFrame({
        'row_n': df.index,
        'target': y_true,
        'prediction': y_pred,
        'score': np.round(y_scores, 2)
    })
    pred_df.to_csv(predictions_csv, index=False)

    # Calcolo metriche
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    roc_auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # Ordine per la matrice come da PDF

    stats = {
        "classifier": clf_name,
        "n_samples": len(df),
        "target_1_count": int(np.sum(y_true == 1)),
        "target_1_percentage": round((np.sum(y_true == 1) / len(df)) * 100, 2),
        "accuracy": float(acc),
        "class_0": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(supp[0])
        },
        "class_1": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(supp[1])
        },
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "labels": [1, 0],
            "matrix": cm.tolist()
        }
    }

    with open(classif_stats_json, 'w') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sotto-parser per train
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--classifier", required=True)
    parser_train.add_argument("--in-reduced", required=True)
    parser_train.add_argument("--target", required=True)
    parser_train.add_argument("--out-model", required=True)
    parser_train.add_argument("--out-metrics", required=True)
    parser_train.add_argument("--seed", type=int, default=42)

    # Sotto-parser per predict
    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("--input-testset", required=True)
    parser_predict.add_argument("--target", required=True)
    parser_predict.add_argument("--model", required=True)
    parser_predict.add_argument("--out-predictions", required=True)
    parser_predict.add_argument("--out-stats", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args.classifier, args.in_reduced, args.target, args.out_model, args.out_metrics, args.seed)
    elif args.command == "predict":
        predict(args.input_testset, args.target, args.model, args.out_predictions, args.out_stats)