# Quantum IDS Project

A project for an Intrusion Detection System (IDS) using Quantum Machine Learning models alongside classical models, evaluated on datasets like NSL-KDD.

## Project Structure

```
quantum_ids_project/
├── data/
│   ├── raw/                 # Raw datasets (e.g., nsl_kdd.csv)
│   ├── processed/           # Processed and cleaned data
│   └── external/            # External data sources
├── config/                  # Configuration settings and paths
├── preprocessing/           # Data cleaning, encoding, scaling, and PCA scripts
├── models/                  # Classical (SVM, RF) and Quantum (QSVM, VQC) models
├── utils/                   # Helper functions
├── notebooks/               # Jupyter notebooks for experimentation
├── main.py                  # Main execution script
└── requirements.txt         # Project dependencies
```

## Setup & Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script with available flags:
```bash
python main.py --help
python main.py --preprocess
python main.py --train
```
