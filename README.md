# MCPNet: Multiscale Convolutional Prototype Network for EEG-Based Parkinson's Disease Detection

Replication of **Qiu et al., 2024** — *A Novel EEG-Based Parkinson's Disease Detection Model Using Multiscale Convolutional Prototype Networks*

## Overview

This project implements a **few-shot learning** approach for detecting Parkinson's Disease (PD) from EEG signals. Unlike traditional deep learning classifiers that need large labeled datasets, MCPNet uses **prototype networks** to classify new subjects using only a handful of labeled EEG samples.

### Key Features

- **Multiscale CNN Encoder**: Parallel convolutional branches (kernel sizes 3, 5, 7) capture EEG patterns at multiple resolutions
- **Prototype-based Classification**: Computes class centroids in embedding space — no retraining needed for new subjects
- **Prototype Calibration**: Adapts prototypes to each test subject using a few labeled samples
- **Dual Feature Extraction**: Combines PSD (spectral power) and PLV (phase connectivity)
- **LOSO Evaluation**: Leave-One-Subject-Out cross-validation for clinically realistic performance estimates
- **3 EEG Datasets**: UC San Diego, UNM, Iowa (87 total subjects)

## Project Structure

```
MCPNet-Parkinsons-EEG/
├── src/
│   ├── config.py           # Hyperparameters, paths, frequency bands
│   ├── dataset.py          # Dataset loading + synthetic data generator
│   ├── preprocessing.py    # EEG preprocessing pipeline (5 steps)
│   ├── features.py         # PSD and PLV feature extraction
│   ├── model.py            # MCPNet architecture (PyTorch)
│   ├── train.py            # Episodic training + LOSO evaluation
│   └── main.py             # Full pipeline runner (CLI)
├── docs/
│   └── Phase1_MCPNet_Study.md  # Paper analysis and study notes
├── data/
│   ├── raw/                # Place downloaded EEG datasets here
│   └── processed/          # Pipeline outputs saved here
└── README.md
```

## Pipeline

```
Raw EEG → Band-pass (0.5-50 Hz) → Notch (50/60 Hz) → ICA → Channel Harmonization (32ch)
    → Epoch Segmentation (1s) → PSD Extraction → PLV Extraction
    → MCPNet Encoder → Prototypes → Calibration → Classification (PD vs HC)
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run with synthetic data (no downloads needed)

```bash
cd src
python main.py --n_subjects 10 --k_shot 5 --n_episodes 10 --n_epochs 5
```

### Run with real datasets

```bash
# Download datasets from OpenNeuro first (see below)
python main.py --real --k_shot 5
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--real` | Use real EEG datasets | synthetic |
| `--n_subjects` | Synthetic subjects count | 10 |
| `--k_shot` | K-shot value (1, 5, 10, 20) | all |
| `--no-plv` | Disable PLV features | enabled |
| `--no-calibration` | Disable prototype calibration | enabled |
| `--n_episodes` | Training episodes per epoch | 100 |
| `--n_epochs` | Training epochs | 50 |

## Datasets

| Dataset | PD | HC | Sampling Rate | Duration | Source |
|---------|----|----|---------------|----------|--------|
| UC San Diego | 15 | 16 | 512 Hz | 3 min | [OpenNeuro ds003490](https://openneuro.org/datasets/ds003490) |
| UNM | 14 | 14 | 500 Hz | 2 min | [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778) |
| Iowa | 14 | 14 | 500 Hz | 2 min | OpenNeuro |

Place downloaded data in `data/raw/UC/`, `data/raw/UNM/`, `data/raw/Iowa/`.

## Model Architecture

```
Input: PSD (32×5) + PLV (32×32×5)
         │
    Multiscale CNN Encoder
    ├── Branch 1 (3×3 kernel) ── fine-grained patterns
    ├── Branch 2 (5×5 kernel) ── medium-scale patterns
    └── Branch 3 (7×7 kernel) ── global patterns
         │
    Concatenate → FC → 128-dim embedding
         │
    ┌────┴────┐
  Support   Query
    │         │
  Prototypes  │
    │         │
  Calibrate   │
    │         │
  Distance ───┘
    │
  PD or HC
```

## Results (Synthetic Data Test)

| K-shot | Accuracy | Sensitivity | Specificity | F1 |
|--------|----------|-------------|-------------|-----|
| 5 | 0.9797 | 0.9594 | 1.0000 | 0.979 |

*Note: Synthetic data has exaggerated PD/HC differences. Real data results will be lower and more clinically meaningful.*

## Reference

Qiu et al. (2024). *A Novel EEG-Based Parkinson's Disease Detection Model Using Multiscale Convolutional Prototype Networks.* IEEE Transactions on Neural Systems and Rehabilitation Engineering.

## Authors

Spruha Kar, Aarsh, Aaryan — DTU Research Group (ML_RP@DTU)
