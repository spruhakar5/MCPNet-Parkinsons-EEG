"""
Configuration for the MCPNet EEG-based Parkinson's Detection pipeline.
"""

import os
from pathlib import Path

# ── Project paths ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ── EEG parameters ──
# Common channels across all 3 datasets (32 channels, 10-20 system)
COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2',
    'F9', 'F10',
]

# Frequency bands for PSD and PLV
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

# Preprocessing
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 50.0
NOTCH_FREQS = [50.0, 60.0]  # Apply both; one will be relevant per dataset
EPOCH_DURATION = 1.0  # seconds

# Dataset-specific info
DATASETS = {
    'UC': {
        'sfreq': 512,
        'duration_sec': 180,  # ~3 minutes
        'n_epochs': 180,
        'pd_subjects': 15,
        'hc_subjects': 16,
    },
    'UNM': {
        'sfreq': 500,
        'duration_sec': 120,  # ~2 minutes
        'n_epochs': 120,
        'pd_subjects': 14,
        'hc_subjects': 14,
    },
    'Iowa': {
        'sfreq': 500,
        'duration_sec': 120,  # ~2 minutes
        'n_epochs': 120,
        'pd_subjects': 14,
        'hc_subjects': 14,
        'channel_remap': {'Pz': 'Fz'},  # Iowa uses Pz instead of Fz
    },
}

# ── Few-shot settings ──
N_WAY = 2       # PD vs HC
K_SHOTS = [1, 5, 10, 20]
N_QUERY = 15    # query samples per class per episode
N_EPISODES_TRAIN = 100  # episodes per LOSO fold for training
N_EPISODES_TEST = 50

# ── Model hyperparameters ──
EMBEDDING_DIM = 128
KERNEL_SIZES = [3, 5, 7]  # multiscale kernels
LEARNING_RATE = 1e-3
N_TRAIN_EPOCHS = 50  # training epochs (not EEG epochs)
BATCH_SIZE = 1  # episodic training uses 1 episode per step
CALIBRATION_ALPHA = 0.5  # weight for prototype calibration

# ── Device ──
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
