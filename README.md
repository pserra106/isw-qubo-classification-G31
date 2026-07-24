# Progetto ISW 2025-26: QUBO Classification - Group G31

This repository contains a Python application for binary classification using a QUBO-based feature selection pipeline.

### 🚀 Setup and Installation

To ensure full reproducibility, please configure your environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 🖥️ How to Launch the GUI

This project uses Streamlit for its user interface. To launch the GUI locally, ensure you have installed the dependencies from requirements.txt, then run the following command from the root of the repository:

```bash
streamlit run src/qubo_project/gui.py
```

### 💻 Command Line Interface (CLI) Usage

The automated evaluation system and users can run the entire pipeline directly from the terminal without using the GUI. Below are the mandatory commands to execute each phase:
Phase 1: Preprocessing

```bash
python src/qubo_project/preprocessing.py \
  --input data/sample_test_dataset.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_result.json \
  --min-perc-valid 0.05
```

## Phase 2: Feature Selection (QUBO)

```bash
python src/qubo_project/feature_selection.py \
  --in-normalized outputs/normalized.csv \
  --out-train outputs/training_reduced.csv \
  --out-test outputs/test_reduced.csv \
  --out-optimizations outputs/optimizations.csv \
  --out-json outputs/feature_selection_result.json \
  --target target \
  --perc-selected 0.20 \
  --allowance 1 \
  --perc-test 0.30 \
  --seed 42 \
  --alpha-computations 100
```

## Phase 3: Model Training

```bash
python src/qubo_project/model.py train \
  --classifier random_forest \
  --in-reduced outputs/training_reduced.csv \
  --target target \
  --out-model outputs/model.joblib \
  --out-metrics outputs/training_metrics.json \
  --seed 42
```

## Phase 4: Prediction

```bash
python src/qubo_project/model.py predict \
  --input-testset outputs/test_reduced.csv \
  --target target \
  --model outputs/model.joblib \
  --out-predictions outputs/predictions.csv \
  --out-stats outputs/classification_stats.json
```

### 🧪 Automated Testing

This project includes automated tests to verify core functionalities (data preprocessing, missing value handling, QUBO feature selection, and model generation).

To run the test suite against the sample test dataset, execute:

```bash
pytest
```