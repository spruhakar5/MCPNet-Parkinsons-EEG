# MCPNet: Multiscale Convolutional Prototype Network for EEG-Based Parkinson's Disease Detection

Replication of **Qiu et al., 2024** — *A Novel EEG-Based Parkinson's Disease Detection Model Using Multiscale Convolutional Prototype Networks*

---

## Overview

This project implements a **few-shot learning** approach for detecting Parkinson's Disease (PD) from resting-state EEG signals. Unlike traditional deep learning classifiers that require large labeled datasets, MCPNet uses **prototype networks** — it can classify a new, unseen patient using only a handful of labeled EEG samples, making it practical for real-world clinical settings where labeled data is scarce.

The implementation covers the **entire pipeline end-to-end**: raw EEG loading, signal preprocessing, feature extraction, model training with episodic learning, and rigorous Leave-One-Subject-Out (LOSO) evaluation across three independent EEG datasets.

### Why This Matters

- Parkinson's affects ~10 million people worldwide, and early detection is critical
- EEG is non-invasive and cheap compared to fMRI or PET scans
- Most prior ML approaches fail on new patients because they overfit to subject-specific patterns
- MCPNet addresses this with few-shot learning + LOSO validation = realistic generalization

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multiscale CNN Encoder** | 3 parallel conv branches (kernel 3, 5, 7) capture fine, medium, and global EEG patterns simultaneously |
| **Prototype Classification** | Computes class centroids in 128-dim embedding space — classify by nearest prototype, no retraining needed |
| **Prototype Calibration** | Adapts prototypes to each test subject using a few labeled calibration samples |
| **Dual Features (PSD + PLV)** | Combines local spectral power (PSD) with inter-regional connectivity (PLV) for richer representation |
| **LOSO Evaluation** | Leave-One-Subject-Out ensures zero data leakage — every test subject is completely unseen |
| **3 EEG Datasets** | UC San Diego (31 subjects), UNM (28), Iowa (28) = 87 total subjects |
| **Synthetic Data Mode** | Built-in synthetic EEG generator for testing the pipeline without downloading real data |

---

## Project Structure

```
MCPNet-Parkinsons-EEG/
├── src/
│   ├── config.py           # All hyperparameters, frequency bands, channel config
│   ├── download_data.py    # Auto-download datasets from OpenNeuro
│   ├── dataset.py          # Dataset loading (3 datasets) + synthetic generator
│   ├── preprocessing.py    # 5-step EEG preprocessing pipeline
│   ├── features.py         # PSD (Welch's method) + PLV (Hilbert transform)
│   ├── model.py            # MCPNet: multiscale encoder + prototypes + calibration
│   ├── train.py            # Episodic N-way K-shot training + LOSO evaluation
│   └── main.py             # Full pipeline runner with CLI arguments
├── docs/
│   └── Phase1_MCPNet_Study.md  # Detailed paper analysis (8 sections)
├── data/
│   ├── raw/                # Place downloaded EEG datasets here
│   └── processed/          # Pipeline outputs (results JSON)
├── setup.sh                # One-command setup (deps + data download)
├── requirements.txt
└── README.md
```

---

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LOADING                                │
│  Load raw EEG (.set/.edf/.bdf) from UC, UNM, Iowa datasets     │
│  OR generate synthetic EEG for testing                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                                 │
│  1. Band-pass filter (0.5–50 Hz) — remove drift & HF noise     │
│  2. Notch filter (50/60 Hz) — remove power line interference    │
│  3. ICA artifact removal — eye blinks, muscle, cardiac          │
│  4. Channel harmonization — standardize to 32 common channels   │
│  5. Epoch segmentation — 1-second non-overlapping windows       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION                              │
│  PSD: Power Spectral Density per channel per band               │
│       → shape: (n_epochs, 32 channels, 5 bands)                │
│       → bands: delta, theta, alpha, beta, gamma                 │
│                                                                 │
│  PLV: Phase Locking Value between all channel pairs             │
│       → shape: (n_epochs, 32, 32, 5 bands)                     │
│       → measures functional brain connectivity                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCPNet MODEL                                │
│                                                                 │
│  Multiscale CNN Encoder:                                        │
│    Branch 1 (3×3) ─┐                                            │
│    Branch 2 (5×5) ─┼─ concat → FC → 128-dim embedding          │
│    Branch 3 (7×7) ─┘                                            │
│                                                                 │
│  Prototype Computation:                                         │
│    PD_prototype = mean(PD support embeddings)                   │
│    HC_prototype = mean(HC support embeddings)                   │
│                                                                 │
│  Prototype Calibration:                                         │
│    calibrated = α·original + (1-α)·test_subject_mean            │
│                                                                 │
│  Classification:                                                │
│    predict = argmin(euclidean_distance(query, prototypes))       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LOSO EVALUATION                                │
│  For each of 87 subjects:                                       │
│    1. Hold out subject S as test                                │
│    2. Train on remaining 86 subjects (episodic training)        │
│    3. Compute prototypes from training support set              │
│    4. Calibrate prototypes using K samples from subject S       │
│    5. Classify remaining epochs of subject S                    │
│    6. Record accuracy, sensitivity, specificity, F1             │
│  Average across all 87 folds                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

> **Clone, setup, run — 3 commands.** The raw EEG data (87 subjects, 3 datasets) is automatically downloaded from OpenNeuro when you run the setup script. No manual downloads needed.

```bash
git clone https://github.com/spruhakar5/MCPNet-Parkinsons-EEG.git
cd MCPNet-Parkinsons-EEG
bash setup.sh              # installs all deps + downloads all 3 EEG datasets from OpenNeuro
cd src && python main.py --real --k_shot 5   # run full pipeline on real data
```

### Step-by-step (if you prefer manual control)

**1. Clone the repository**
```bash
git clone https://github.com/spruhakar5/MCPNet-Parkinsons-EEG.git
cd MCPNet-Parkinsons-EEG
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the real EEG datasets**
```bash
cd src
python download_data.py --dataset all    # fetches UC, UNM, Iowa from OpenNeuro
python download_data.py --verify         # confirm downloads
```

**4. Run the pipeline**
```bash
# With real data:
python main.py --real --k_shot 5

# Or test with synthetic data first (no download needed):
python main.py --n_subjects 10 --k_shot 5 --n_episodes 10 --n_epochs 5
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--real` | Use real EEG datasets from `data/raw/` | synthetic |
| `--n_subjects N` | Number of synthetic subjects to generate | 10 |
| `--skip_ica` | Skip ICA step (faster for testing) | True |
| `--k_shot K` | Specific K-shot value (1, 5, 10, or 20) | run all |
| `--no-plv` | Disable PLV features (PSD only) | PLV enabled |
| `--no-calibration` | Disable prototype calibration | calibration enabled |
| `--n_episodes N` | Training episodes per epoch | 100 |
| `--n_epochs N` | Number of training epochs per fold | 50 |

---

## Datasets

| Dataset | PD Subjects | HC Subjects | Total | Sampling Rate | Duration | Epochs/Subject | Source |
|---------|-------------|-------------|-------|---------------|----------|----------------|--------|
| UC San Diego | 15 | 16 | 31 | 512 Hz | 3 min | ~180 | [OpenNeuro ds003490](https://openneuro.org/datasets/ds003490) |
| UNM | 14 | 14 | 28 | 500 Hz | 2 min | ~120 | [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778) |
| Iowa | 14 | 14 | 28 | 500 Hz | 2 min | ~120 | OpenNeuro |
| **Combined** | **43** | **44** | **87** | — | — | **~12,300** | — |

### Automated Download (Recommended)

```bash
# One-command setup: installs deps + downloads all 3 datasets
bash setup.sh

# Or download datasets individually:
cd src
python download_data.py --dataset UC     # UC San Diego only
python download_data.py --dataset UNM    # UNM only
python download_data.py --dataset Iowa   # Iowa only
python download_data.py --dataset all    # All three

# Verify downloads:
python download_data.py --verify
```

The download script auto-fetches from OpenNeuro using `openneuro-py`. Falls back to AWS S3 direct download if needed. If both fail, it prints manual download URLs.

### Manual Download (Fallback)

```bash
pip install openneuro-py
openneuro download --dataset ds003490 data/raw/UC
openneuro download --dataset ds002778 data/raw/UNM
openneuro download --dataset ds004584 data/raw/Iowa
```

Or download directly from https://openneuro.org and place files in `data/raw/<dataset_name>/`.

Each dataset folder should contain a `participants.tsv` with columns `participant_id` and `group` (PD or HC).

> **Note**: Raw EEG data is several GB and is not stored in this repo. The `download_data.py` script fetches it from OpenNeuro on demand.

---

## Code Walkthrough

### `config.py` — Central Configuration

All hyperparameters in one place. Key settings:

- **32 common EEG channels** (10-20 system) used across all datasets
- **5 frequency bands**: delta (0.5-4 Hz), theta (4-8), alpha (8-13), beta (13-30), gamma (30-50)
- **Few-shot settings**: 2-way classification, K-shots of 1/5/10/20
- **Model**: 128-dim embeddings, kernel sizes [3, 5, 7], Adam optimizer with StepLR

### `dataset.py` — Data Loading

- **`load_all_datasets()`**: Loads real EEG data from `data/raw/` using MNE (supports .set, .edf, .bdf, .fif)
- **`generate_synthetic_data()`**: Creates fake EEG signals for pipeline testing — PD subjects get boosted theta and reduced beta power (mimicking real PD spectral changes)
- **`Subject` dataclass**: Carries a subject through the entire pipeline (raw → epochs → PSD/PLV features)
- Automatically handles `participants.tsv` label parsing and Iowa channel remapping (Pz → Fz)

### `preprocessing.py` — EEG Signal Cleaning

Five sequential steps:

1. **Band-pass filter (0.5–50 Hz)**: FIR filter removes DC drift and high-frequency noise while retaining all relevant brain rhythms
2. **Notch filter (50/60 Hz)**: Removes power line interference (applies both frequencies — harmless if one isn't present)
3. **ICA artifact removal**: FastICA with automatic EOG component detection using frontal channels (Fp1, Fp2). Falls back to kurtosis-based heuristic if auto-detection fails. Removes at most 5 components
4. **Channel harmonization**: Case-insensitive matching to select and reorder 32 standard channels across all datasets
5. **Epoch segmentation**: 1-second non-overlapping windows using MNE's fixed-length events

### `features.py` — PSD and PLV Extraction

**PSD (Power Spectral Density)**:
- Computed per channel using **Welch's method** (scipy.signal.welch)
- Averages power within each of the 5 frequency bands
- Output: `(n_epochs, 32, 5)` — captures *local* neural activity

**PLV (Phase Locking Value)**:
- For each frequency band: band-pass filter → Hilbert transform → extract instantaneous phase
- PLV = |mean(exp(j * phase_difference))| for each channel pair
- Output: `(n_epochs, 32, 32, 5)` — captures *network-level* synchronization
- Symmetric matrix with diagonal = 1

### `model.py` — MCPNet Architecture

**ConvBranch**: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → AdaptiveAvgPool

**MultiscaleEncoder**: 3 parallel ConvBranches (kernels 3, 5, 7) → concatenate → FC → BatchNorm → ReLU → 128-dim embedding

**MCPNet**:
- Separate encoders for PSD (1-channel input) and PLV (5-channel input)
- Fusion layer combines both embeddings
- `compute_prototypes()`: Mean embedding per class from support set
- `calibrate_prototypes()`: Weighted average of training prototype and test-subject mean
- `classify()`: Euclidean distance → softmax → nearest prototype

### `train.py` — Episodic Training and LOSO

**Episode creation**: Randomly samples K support + N query epochs per class from available subjects

**Training loop**: For each episode → encode support/query → compute prototypes → classify queries → NLL loss → backprop

**LOSO evaluation**: Fresh model per fold → train on 86 subjects → test on held-out subject → calibrate with K test-subject epochs → classify remaining epochs → aggregate metrics (accuracy, sensitivity, specificity, F1)

### `main.py` — Pipeline Runner

Orchestrates the full pipeline: load data → preprocess → extract features → run LOSO for each K-shot setting → save results to JSON → print summary table.

---

## Model Architecture Diagram

```
Input Features
├── PSD: (batch, 32, 5)          ← spectral power per channel per band
└── PLV: (batch, 32, 32, 5)     ← phase connectivity between all channel pairs
         │                │
         ▼                ▼
┌─────────────┐  ┌──────────────┐
│ PSD Encoder │  │  PLV Encoder │
│ (1-ch input)│  │ (5-ch input) │
│             │  │              │
│ ┌─────────┐ │  │ ┌──────────┐│
│ │ 3×3 conv│ │  │ │ 3×3 conv ││
│ │ 5×5 conv│ │  │ │ 5×5 conv ││
│ │ 7×7 conv│ │  │ │ 7×7 conv ││
│ └────┬────┘ │  │ └─────┬────┘│
│   concat    │  │    concat   │
│   FC→128    │  │    FC→128   │
└──────┬──────┘  └──────┬──────┘
       │                │
       └───── concat ───┘
              │
         FC → 128-dim
         (fused embedding)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
 Support              Query
 Embeddings           Embeddings
    │
    ▼
 Prototypes
 (mean per class)
    │
    ▼
 Calibrate with
 test subject samples
    │
    ▼
 Euclidean distance
 to each prototype
    │
    ▼
 Prediction: PD or HC
```

---

## Evaluation Strategy

### Why LOSO over K-Fold?

| Method | Data Leakage? | Clinical Realism | Typical Reported Accuracy |
|--------|---------------|------------------|--------------------------|
| **K-Fold CV** | Yes — epochs from same subject in train+test | Low — never encounters truly unseen patient | >99% (inflated) |
| **LOSO** | No — entire subject held out | High — simulates new patient encounter | 60-85% (realistic) |

### Few-Shot Episode Structure (2-way K-shot)

```
Support Set:           Query Set:
┌─────────────────┐    ┌─────────────────┐
│ K epochs (HC)   │    │ N epochs (HC)   │ ← classify these
│ K epochs (PD)   │    │ N epochs (PD)   │
└─────────────────┘    └─────────────────┘
        │                      │
        ▼                      ▼
  Compute prototypes    Compare to prototypes
  (mean embedding       (nearest = prediction)
   per class)
```

---

## Results

### Synthetic Data Benchmark (6 subjects, K=5)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 97.97% |
| Sensitivity (PD recall) | 95.94% |
| Specificity (HC recall) | 100.00% |
| F1 Score | 0.979 |
| Confusion Matrix | TN=345, FP=0, FN=14, TP=331 |

*Synthetic data uses exaggerated PD/HC spectral differences for pipeline validation. Real dataset results will be lower and more clinically meaningful, consistent with LOSO evaluation rigor.*

---

## Frequency Bands and Their Relevance to PD

| Band | Range | Brain Activity | PD Signature |
|------|-------|----------------|--------------|
| Delta (δ) | 0.5–4 Hz | Deep sleep, unconscious | Increased in advanced PD |
| Theta (θ) | 4–8 Hz | Drowsiness, memory | **Increased** — cortical slowing |
| Alpha (α) | 8–13 Hz | Relaxed wakefulness | Altered resting-state patterns |
| Beta (β) | 13–30 Hz | Motor planning, focus | **Decreased** — motor dysfunction |
| Gamma (γ) | 30–50 Hz | Higher cognition | Variable changes |

---

## Reference

Qiu et al. (2024). *A Novel EEG-Based Parkinson's Disease Detection Model Using Multiscale Convolutional Prototype Networks.* IEEE Transactions on Neural Systems and Rehabilitation Engineering.

## Authors

Spruha Kar, Aarsh, Aaryan — DTU Research Group (ML_RP@DTU)
