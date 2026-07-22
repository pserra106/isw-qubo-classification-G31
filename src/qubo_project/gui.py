#Un'interfaccia semplice basata su Streamlit (avviabile con streamlit run src/qubo_project/gui.py)
# che permette di controllare tutte le fasi del progetto:

import streamlit as st
import pandas as pd
from pathlib import Path
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train, predict

st.title("G31 - QUBO Classification Dashboard")

st.sidebar.header("Parametri Pipeline")
uploaded_file = st.sidebar.file_uploader("Carica Dataset CSV", type=["csv"])
target_col = st.sidebar.text_input("Colonna Target", value="target")

if uploaded_file is not None:
    data_path = Path("data/input_dataset.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Dataset caricato con successo!")

    if st.button("1. Esegui Preprocessing"):
        with st.spinner("Elaborazione preprocessing in corso..."):
            fit_normalize(str(data_path), target_col, "outputs/normalized.csv", "outputs/preprocessing_result.json")
        st.success("Preprocessing completato!")

    if st.button("2. Esegui Feature Selection QUBO"):
        with st.spinner("Ottimizzazione QUBO in corso..."):
            select_features(
                "outputs/normalized.csv",
                "outputs/training_reduced.csv",
                "outputs/test_reduced.csv",
                "outputs/optimizations.csv",
                "outputs/feature_selection_result.json",
                target_col
            )
        st.success("Feature Selection completata!")

    classifier_choice = st.selectbox("Seleziona Classificatore", ["random_forest", "logistic_regression", "gradient_boosting"])

    if st.button("3. Addestra Modello"):
        with st.spinner("Training del modello in corso..."):
            train(
                classifier_choice,
                "outputs/training_reduced.csv",
                target_col,
                "outputs/model.joblib",
                "outputs/training_metrics.json"
            )
        st.success("Modello addestrato e salvato!")

    if st.button("4. Esegui Predizioni e Valutazione"):
        with st.spinner("Valutazione sul test set in corso..."):
            predict(
                "outputs/test_reduced.csv",
                target_col,
                "outputs/model.joblib",
                "outputs/predictions.csv",
                "outputs/classification_stats.json"
            )
        st.success("Predizioni completate!")
        if Path("outputs/classification_stats.json").exists():
            import json
            with open("outputs/classification_stats.json") as f:
                stats = json.load(f)
            st.json(stats)