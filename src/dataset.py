"""
Dataset preparation and loading utilities.

The three EEG datasets used in the MCPNet paper:
1. UC San Diego  - available via OpenNeuro (ds003490)
2. UNM           - available via OpenNeuro (ds002778)
3. Iowa          - available via OpenNeuro (ds002778 or supplementary)

This module provides:
- Download instructions
- Loading raw EEG files (.set/.edf/.bdf)
- Organizing subjects with labels (PD / HC)
- A unified Subject dataclass
"""

import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mne

from config import DATA_RAW, DATA_PROCESSED, DATASETS, COMMON_CHANNELS


@dataclass
class Subject:
    """Represents a single EEG subject."""
    subject_id: str
    dataset: str          # 'UC', 'UNM', or 'Iowa'
    label: int            # 0 = HC, 1 = PD
    raw_path: str = ""    # path to raw EEG file
    raw: Optional[mne.io.BaseRaw] = field(default=None, repr=False)
    epochs: Optional[mne.Epochs] = field(default=None, repr=False)
    psd_features: Optional[np.ndarray] = field(default=None, repr=False)
    plv_features: Optional[np.ndarray] = field(default=None, repr=False)


def print_download_instructions():
    """Print instructions for downloading the datasets."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              EEG DATASET DOWNLOAD INSTRUCTIONS                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  Place downloaded files in: data/raw/<dataset_name>/           ║
    ║                                                                ║
    ║  1. UC San Diego Dataset (ds003490)                            ║
    ║     URL: https://openneuro.org/datasets/ds003490               ║
    ║     Place in: data/raw/UC/                                     ║
    ║     Structure: data/raw/UC/sub-XXX/eeg/sub-XXX_eeg.set        ║
    ║                                                                ║
    ║  2. UNM Dataset (ds002778)                                     ║
    ║     URL: https://openneuro.org/datasets/ds002778               ║
    ║     Place in: data/raw/UNM/                                    ║
    ║     Structure: data/raw/UNM/sub-XXX/eeg/sub-XXX_eeg.set       ║
    ║                                                                ║
    ║  3. Iowa Dataset                                               ║
    ║     URL: https://openneuro.org/datasets/ds002778               ║
    ║     Place in: data/raw/Iowa/                                   ║
    ║     Structure: data/raw/Iowa/sub-XXX/eeg/sub-XXX_eeg.set      ║
    ║                                                                ║
    ║  Alternative: Use the OpenNeuro CLI:                           ║
    ║    pip install openneuro-py                                    ║
    ║    openneuro download --dataset ds003490 data/raw/UC           ║
    ║                                                                ║
    ║  Each dataset folder should also contain a                     ║
    ║  participants.tsv with columns: participant_id, group          ║
    ║  where group is 'PD' or 'HC'                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


def load_participants_tsv(dataset_dir: Path) -> dict:
    """
    Load participants.tsv to get subject ID → label mapping.
    Expected columns: participant_id, group (PD or HC)
    Returns dict: {'sub-001': 1, 'sub-002': 0, ...}
    """
    tsv_path = dataset_dir / "participants.tsv"
    labels = {}

    if not tsv_path.exists():
        print(f"  [WARNING] No participants.tsv found at {tsv_path}")
        print(f"  You will need to provide labels manually.")
        return labels

    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')
        id_col = header.index('participant_id')
        group_col = header.index('group')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(id_col, group_col):
                sid = parts[id_col]
                group = parts[group_col].upper().strip()
                labels[sid] = 1 if group == 'PD' else 0

    print(f"  Loaded {len(labels)} subject labels from {tsv_path}")
    return labels


def discover_eeg_files(dataset_dir: Path) -> list:
    """Find all EEG files (.set, .edf, .bdf, .fif) in a dataset directory."""
    patterns = ['**/*.set', '**/*.edf', '**/*.bdf', '**/*.fif']
    files = []
    for pat in patterns:
        files.extend(dataset_dir.glob(pat))
    return sorted(files)


def load_raw_eeg(filepath: Path, dataset_name: str) -> mne.io.BaseRaw:
    """
    Load a single raw EEG file using MNE.
    Handles .set (EEGLAB), .edf, .bdf, and .fif formats.
    """
    ext = filepath.suffix.lower()

    if ext == '.set':
        raw = mne.io.read_raw_eeglab(str(filepath), preload=True, verbose=False)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    elif ext == '.bdf':
        raw = mne.io.read_raw_bdf(str(filepath), preload=True, verbose=False)
    elif ext == '.fif':
        raw = mne.io.read_raw_fif(str(filepath), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG format: {ext}")

    # Apply channel remapping if needed (e.g., Iowa: Pz → Fz)
    ds_config = DATASETS.get(dataset_name, {})
    remap = ds_config.get('channel_remap', {})
    if remap:
        existing_ch = raw.ch_names
        rename_dict = {old: new for old, new in remap.items() if old in existing_ch}
        if rename_dict:
            raw.rename_channels(rename_dict)
            print(f"    Remapped channels: {rename_dict}")

    return raw


def load_dataset(dataset_name: str) -> list:
    """
    Load all subjects from a dataset.
    Returns list of Subject objects with raw EEG loaded.
    """
    dataset_dir = DATA_RAW / dataset_name
    if not dataset_dir.exists():
        print(f"[SKIP] Dataset directory not found: {dataset_dir}")
        print_download_instructions()
        return []

    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")

    # Get labels
    labels = load_participants_tsv(dataset_dir)

    # Find EEG files
    eeg_files = discover_eeg_files(dataset_dir)
    if not eeg_files:
        print(f"  [WARNING] No EEG files found in {dataset_dir}")
        return []

    print(f"  Found {len(eeg_files)} EEG files")

    subjects = []
    for fpath in eeg_files:
        # Extract subject ID from path (e.g., sub-001)
        parts = fpath.parts
        sub_id = None
        for p in parts:
            if p.startswith('sub-'):
                sub_id = p
                break

        if sub_id is None:
            sub_id = fpath.stem

        # Get label
        label = labels.get(sub_id, -1)  # -1 if unknown
        if label == -1:
            print(f"  [WARNING] No label for {sub_id}, skipping")
            continue

        print(f"  Loading {sub_id} ({'PD' if label == 1 else 'HC'})... ", end="")
        try:
            raw = load_raw_eeg(fpath, dataset_name)
            subj = Subject(
                subject_id=sub_id,
                dataset=dataset_name,
                label=label,
                raw_path=str(fpath),
                raw=raw,
            )
            subjects.append(subj)
            print(f"OK ({len(raw.ch_names)} ch, {raw.n_times/raw.info['sfreq']:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"  Total loaded: {len(subjects)} subjects "
          f"({sum(1 for s in subjects if s.label==1)} PD, "
          f"{sum(1 for s in subjects if s.label==0)} HC)")

    return subjects


def load_all_datasets() -> list:
    """Load all three datasets and return combined list of subjects."""
    all_subjects = []
    for ds_name in ['UC', 'UNM', 'Iowa']:
        subjects = load_dataset(ds_name)
        all_subjects.extend(subjects)

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_subjects)} subjects loaded across all datasets")
    print(f"  PD: {sum(1 for s in all_subjects if s.label == 1)}")
    print(f"  HC: {sum(1 for s in all_subjects if s.label == 0)}")
    print(f"{'='*60}")

    return all_subjects


def generate_synthetic_data(n_subjects=20, n_channels=32, sfreq=500,
                            duration_sec=120) -> list:
    """
    Generate synthetic EEG-like data for testing the pipeline
    when real datasets are not yet available.

    PD subjects have slightly more theta power and less beta power.
    """
    print(f"\n{'='*60}")
    print(f"Generating synthetic EEG data for pipeline testing")
    print(f"  {n_subjects} subjects, {n_channels} channels, {sfreq} Hz, {duration_sec}s")
    print(f"{'='*60}")

    subjects = []
    n_pd = n_subjects // 2
    n_hc = n_subjects - n_pd

    for i in range(n_subjects):
        label = 1 if i < n_pd else 0
        sub_id = f"syn-{i+1:03d}"

        np.random.seed(42 + i)
        n_samples = sfreq * duration_sec
        t = np.arange(n_samples) / sfreq

        # Generate multi-channel EEG-like signal
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Base signal: mix of frequency components
            delta = 20 * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
            theta = 10 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            alpha = 8 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            beta = 5 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            gamma = 2 * np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi)

            if label == 1:  # PD: more theta, less beta
                theta *= 1.5
                beta *= 0.6
            else:  # HC: normal
                theta *= 0.8
                beta *= 1.2

            noise = 3 * np.random.randn(n_samples)
            data[ch] = delta + theta + alpha + beta + gamma + noise

        # Scale to microvolts
        data *= 1e-6

        # Create MNE Raw object
        ch_names = COMMON_CHANNELS[:n_channels]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)

        subj = Subject(
            subject_id=sub_id,
            dataset='synthetic',
            label=label,
            raw=raw,
        )
        subjects.append(subj)
        print(f"  Created {sub_id} ({'PD' if label==1 else 'HC'})")

    print(f"  Total: {n_pd} PD, {n_hc} HC")
    return subjects


if __name__ == "__main__":
    # Try loading real data first; fall back to synthetic
    subjects = load_all_datasets()
    if not subjects:
        print("\nNo real data found. Generating synthetic data for testing...")
        subjects = generate_synthetic_data()
