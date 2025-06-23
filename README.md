# DREAM Challenge 2025 - WDR91 Activity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning pipeline for molecular activity prediction in the DREAM Challenge 2025, specifically targeting **WDR91 protein activity prediction** using multiple molecular fingerprint representations.

## Key Features

- **Multi-branch MLP**: PyTorch-based neural network with separate branches for each fingerprint type
- **LightGBM Ensemble**: Gradient boosting ensemble with stacking and Platt scaling calibration
- **Multi-Fingerprint Support**: Nine fingerprint types (ECFP4/6, FCFP4/6, MACCS, ATOMPAIR, TOPTOR, RDK, AVALON)
- **Advanced Cross-Validation**: Stratified group k-fold with cluster-aware splitting
- **Molecular Diversity**: Tanimoto similarity-based clustering for diverse compound selection

## Repository Structure

```text
├── configs/                 # Configuration and argument parsing
│   └── config.py           # Centralized configuration management
├── models/                 # Core model implementations
│   ├── lgbm_ensemble.py    # LightGBM ensemble with stacking
│   └── multibranch_mlp.py  # Multi-branch neural network
├── train/                  # Training pipelines
│   ├── train_lgbm.py       # LightGBM training script
│   └── train_mlp.py        # MLP training script
├── predict/                # Prediction and submission generation
│   ├── predict_lgbm_submission.py  # LightGBM prediction pipeline
│   └── predict_mlp_submission.py   # MLP prediction pipeline
├── utils/                  # Shared utilities and helper functions
│   └── utils.py           # Fingerprint processing, data handling, etc.
└── requirements.txt        # Python dependencies
```

**Note:** Output directories (`output/`, `models/`, `logs/`, etc.) are created automatically when you run the scripts with `--outdir` parameter.

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **CUDA support** (optional, for GPU acceleration)

### Quick Installation

1. **Clone the repository:**

```bash
git clone https://github.com/LucaRuvo/DREAM_Challenge_2025.git
cd DREAM_Challenge_2025
```

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### RDKit Installation

```bash
# Via conda (recommended)
conda install -c conda-forge rdkit

# Via pip (alternative)
pip install rdkit
```

## Data Requirements & Format

### Input Data Format

Your datasets should be **Parquet files** with the following structure:

#### Training Data (Challenge training set: `WDR91_HitGen.parquet`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `RandomID` | str | Unique compound identifier | `"CMPD_001234"` |
| `LABEL` | int | Activity label (0=inactive, 1=active) | `1` |
| `ECFP4` | str | ECFP4 fingerprint (comma-separated) | `"0,1,0,1,1,0,..."` |
| `ECFP6` | str | ECFP6 fingerprint | `"1,0,1,0,0,1,..."` |
| `FCFP4` | str | FCFP4 fingerprint | `"1,1,0,0,1,0,..."` |
| `FCFP6` | str | FCFP6 fingerprint | `"0,1,1,0,0,1,..."` |
| `MACCS` | str | MACCS keys fingerprint | `"0,0,1,1,0,..."` |
| `ATOMPAIR` | str | Atom pair fingerprint | `"1,0,0,1,1,..."` |
| `TOPTOR` | str | Topological torsion fingerprint | `"0,1,0,0,1,..."` |
| `RDK` | str | RDKit fingerprint | `"1,0,1,1,0,..."` |
| `AVALON` | str | Avalon fingerprint | `"0,0,1,0,1,..."` |
| `BB1_ID`, `BB2_ID`, `BB3_ID` | str | Building block IDs (optional) | `"BB_001"` |

**Note:** All 9 fingerprint types (ECFP4, ECFP6, FCFP4, FCFP6, MACCS, ATOMPAIR, TOPTOR, RDK, AVALON) must be included in the pipeline.

#### Test Data

Same format as training data, but **without the `LABEL` column**.

#### Submission Format

The final submission for STEP 1 was a **4-column CSV file** with the following format:

| Column Name | Data Type | Description | Accepted Values |
|-------------|-----------|-------------|-----------------|
| `RandomID` | str | Anonymized compound IDs (must match test set) | IDs from test data |
| `Sel_200` | int | Binary selection flag for top 200 diverse compounds | `0` or `1` (exactly 200 ones) |
| `Sel_500` | int | Binary selection flag for top 500 diverse compounds | `0` or `1` (exactly 500 ones) |
| `Score` | float | Binding probability/confidence score for all compounds | `0.0` to `1.0` |

**Requirements:**

- File name: `"TeamYOURTEAMNAME.csv"`
- Exactly **200** compounds with `Sel_200=1`
- Exactly **500** compounds with `Sel_500=1`
- All `RandomID` values must match those in the test set
- `Score` values for all ~339K compounds in test set

## Usage Guide

### Quick Start Example

Here's a complete example to get started:

```bash
# 1. Train a LightGBM model
python train/train_lgbm.py \
    --train-data "data/WDR91_HitGen.parquet" \
    --test-data "data/WDR91_all_data.parquet" \
    --outdir "./output/lgbm_run1/" \
    --seed 42 \
    --n-splits 5

# 2. Generate LightGBM predictions and submission
python predict/predict_lgbm_submission.py \
    --test-data "data/WDR91_all_data.parquet" \
    --output-file "./output/submission_lgbm.csv" \
    --model-dir "./output/lgbm_run1/" \
    --seed 42

# 3. Train an MLP model  
python train/train_mlp.py \
    --train-data "data/WDR91_HitGen.parquet" \
    --outdir "./output/mlp_run1/" \
    --epochs 100 \
    --batch-size 256 \
    --seed 42

# 4. Generate MLP predictions and submission
python predict/predict_mlp_submission.py \
    --test-data "data/WDR91_all_data.parquet" \
    --output-file "./output/submission_mlp.csv" \
    --model-dir "./output/mlp_run1/" \
    --batch-size 256 \
    --seed 42
```

### Configuration Options

#### Essential Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--train-data` | Path to training Parquet file | `"data/train.parquet"` |
| `--test-data` | Path to test Parquet file (training only) | `"data/test.parquet"` |
| `--output-file` | Path to output submission CSV (prediction only) | `"./output/submission.csv"` |
| `--model-dir` | Directory containing trained models (prediction only) | `"./output/run1/"` |
| `--outdir` | Output directory for training results | `"./output/run1/"` |
| `--seed` | Random seed for reproducibility | `42` |

#### Model-Specific Arguments

**LightGBM Options:**

- `--n-splits`: Number of CV folds (default: 5)
- `--grouping-strategy`: Clustering method (`auto`, `bb`, `kmeans`)
- `--device`: Compute device (`cpu`, `gpu`)

**MLP Options:**

- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Batch size (default: 256)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--branch-hidden-dims`: Architecture preset (`deeper`, `wider`, `shallower`)
- `--dropout-rate`: Dropout probability (default: 0.4)
- `--positive-class-weight`: Class balance weight (default: 1.0)

#### Device Configuration

The scripts automatically detect and use GPU when available. You can also explicitly specify the device:

- `--device cuda`: Use GPU (PyTorch format) - **Auto-detected by default**
- `--device cpu`: Force CPU usage
- No `--device` argument: **Auto-detects CUDA/GPU**

## Model Architectures

### LightGBM Ensemble Details

**Architecture:**

```text
Input: 9 Fingerprint Types
    ↓
[FP1_Model] [FP2_Model] ... [FP9_Model]  ← Individual LightGBM models
    ↓           ↓              ↓
[  OOF Predictions Matrix  ]             ← Out-of-fold predictions
    ↓
[Logistic Regression Stacker]            ← Meta-learner
    ↓
Final Prediction Probability
```

**Key Features:**

- **Multi-fingerprint ensemble**: Separate LightGBM model for each fingerprint type
- **Stacking**: Logistic regression meta-learner combines base model predictions
- **Cross-validation**: Stratified group k-fold with cluster-aware splitting
- **Probability calibration**: Platt scaling using out-of-fold predictions
- **Class balancing**: Handles imbalanced datasets (typical in drug discovery)

### Multi-branch MLP Details

**Architecture:**

```text
Input: 9 Fingerprint Types (as tensors)
    ↓
[FP1_Branch] [FP2_Branch] ... [FP9_Branch]  ← Separate neural networks    ↓           ↓              ↓
[  Concatenated Embeddings  ]               ← Combined representations
    ↓
[Common MLP Layers]                         ← Shared final layers
    ↓
Single Output (Binary Classification)
```

**Branch Architecture per Fingerprint:**

```text
Input Fingerprint (2048 or 167 dims)
    ↓
Linear(input_dim → 2048) + GroupNorm + ReLU + Dropout
    ↓
Linear(2048 → 1024) + GroupNorm + ReLU + Dropout
    ↓
Linear(1024 → 256) + GroupNorm + ReLU
    ↓
Embedding Output (256 dims)
```

**Common Layers:**

```text
Concatenated Embeddings (9 × 256 = 2304 dims)
    ↓
Linear(2304 → 1024) + GroupNorm + ReLU + Dropout
    ↓
Linear(1024 → 512) + GroupNorm + ReLU + Dropout
    ↓
Linear(512 → 1) + Sigmoid
    ↓
Binary Prediction
```

**Key Features:**

- **Separate fingerprint processing**: Each fingerprint type has its own neural network branch
- **GroupNorm**: Robust to small batch sizes
- **Configurable architecture**: Preset options (deeper/wider/shallower)
- **Dropout regularization**: Prevents overfitting
- **Early stopping**: Monitors validation loss
- **Class weighting**: Handles imbalanced data

## Advanced Features

### Cross-Validation Strategies

1. **Stratified Group K-Fold**: Maintains class balance while respecting molecular clusters
2. **Building Block Grouping**: Uses chemical building blocks for grouped CV (computationally efficient)
3. **ECFP4-based Clustering**: MiniBatch K-means clustering on fingerprints

### Molecular Diversity & Selection

The pipeline includes compound selection for diverse submissions:

1. **Tanimoto Similarity Clustering**: Groups chemically similar compounds
2. **Iterative Diversity Filtering**: Ensures maximum 2 compounds per cluster
3. **Top-K Selection**: Selects top 200 and top 500 compounds
4. **Ranking Preservation**: Maintains prediction-based ranking within diversity constraints

### Calibration & Uncertainty

- **Platt Scaling**: Calibrates prediction probabilities using out-of-fold data
- **Ensemble Uncertainty**: Multiple model predictions provide confidence estimates
- **Cross-validation Metrics**: Comprehensive evaluation (AUC, AUPRC, F1, Precision, Recall)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Citation

If you use this code in your research, please cite:

```bibtex
@software{dream_challenge_2025,
  title={DREAM Challenge 2025 - WDR91 Activity Prediction Pipeline},
  author={LucaRuvo},
  year={2025},
  url={https://github.com/yourusername/DREAM_Challenge_2025},
  version={1.0.0}
}
```

## Acknowledgments

- **DREAM Challenge Organizers**: For providing the WDR91 challenge dataset
- **Scientific Community**: For open-source molecular machine learning tools
