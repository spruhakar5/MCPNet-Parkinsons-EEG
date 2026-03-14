"""
EEG Preprocessing Pipeline for MCPNet.

Steps:
1. Band-pass filtering (0.5–50 Hz)
2. Notch filtering (50/60 Hz)
3. ICA artifact removal
4. Channel harmonization (select 32 common channels)
5. Epoch segmentation (1-second non-overlapping windows)
"""

import numpy as np
import mne
from mne.preprocessing import ICA

from config import (
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQS,
    COMMON_CHANNELS, EPOCH_DURATION,
)


def bandpass_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Apply band-pass filter (0.5–50 Hz).
    Removes DC drift (low-freq) and high-frequency noise.
    """
    raw_filtered = raw.copy().filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method='fir',
        fir_design='firwin',
        verbose=False,
    )
    return raw_filtered


def notch_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Apply notch filter to remove power line interference.
    Applies both 50 Hz and 60 Hz notch (harmless if one isn't present).
    """
    raw_notched = raw.copy().notch_filter(
        freqs=NOTCH_FREQS,
        method='fir',
        fir_design='firwin',
        verbose=False,
    )
    return raw_notched


def run_ica(raw: mne.io.BaseRaw, n_components=20, random_state=42) -> mne.io.BaseRaw:
    """
    Run ICA for artifact removal.
    Automatically detects and removes EOG (eye) and muscle artifact components.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Filtered EEG data.
    n_components : int
        Number of ICA components to compute.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    raw_clean : mne.io.BaseRaw
        EEG data with artifact components removed.
    """
    # Fit ICA
    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter=500,
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    # Auto-detect EOG-like components using frontal channels
    eog_indices = []
    frontal_chs = [ch for ch in ['Fp1', 'Fp2', 'F7', 'F8'] if ch in raw.ch_names]
    if frontal_chs:
        for ch in frontal_chs:
            try:
                indices, scores = ica.find_bads_eog(
                    raw, ch_name=ch, verbose=False
                )
                eog_indices.extend(indices)
            except Exception:
                pass

    # If no automatic detection worked, use correlation-based heuristic
    if not eog_indices:
        # Exclude components with very high variance (likely artifacts)
        sources = ica.get_sources(raw).get_data()
        kurtosis = np.array([
            float(np.mean(s**4) / (np.mean(s**2)**2) - 3)
            for s in sources
        ])
        # Components with high kurtosis are likely eye blink artifacts
        threshold = np.mean(kurtosis) + 2 * np.std(kurtosis)
        eog_indices = list(np.where(kurtosis > threshold)[0])

    # Remove duplicates and limit
    eog_indices = list(set(eog_indices))[:5]  # remove at most 5 components

    if eog_indices:
        ica.exclude = eog_indices

    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean


def harmonize_channels(raw: mne.io.BaseRaw,
                       target_channels: list = None) -> mne.io.BaseRaw:
    """
    Select and reorder channels to match the common 32-channel set.
    Drops extra channels, raises warning for missing ones.
    """
    if target_channels is None:
        target_channels = COMMON_CHANNELS

    available = raw.ch_names
    # Find channels that exist (case-insensitive matching)
    ch_map = {}
    for target in target_channels:
        for avail in available:
            if avail.lower() == target.lower():
                ch_map[target] = avail
                break

    found = list(ch_map.values())
    missing = [ch for ch in target_channels if ch not in ch_map]

    if missing:
        print(f"    [WARN] Missing channels: {missing}")
        # For missing channels, we'll interpolate or use zeros
        # Pick only available target channels
        target_channels = [ch for ch in target_channels if ch in ch_map]
        found = [ch_map[ch] for ch in target_channels]

    # Pick only EEG channels first, then select our targets
    raw_picked = raw.copy().pick(found)

    # Rename to standardized names if needed
    rename = {ch_map[t]: t for t in target_channels if ch_map[t] != t}
    if rename:
        raw_picked.rename_channels(rename)

    return raw_picked


def segment_epochs(raw: mne.io.BaseRaw,
                   duration: float = EPOCH_DURATION) -> mne.Epochs:
    """
    Segment continuous EEG into fixed-length non-overlapping epochs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed continuous EEG.
    duration : float
        Epoch duration in seconds (default: 1.0).

    Returns
    -------
    epochs : mne.Epochs
        Segmented EEG epochs.
    """
    # Create events at regular intervals
    events = mne.make_fixed_length_events(raw, duration=duration)

    # Create epochs (no baseline correction — already filtered)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=duration - (1.0 / raw.info['sfreq']),
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs


def preprocess_subject(subject, skip_ica=False):
    """
    Run the full preprocessing pipeline on a single subject.

    Parameters
    ----------
    subject : Subject
        Subject object with .raw attribute populated.
    skip_ica : bool
        If True, skip ICA (useful for synthetic data or speed).

    Returns
    -------
    subject : Subject
        Same subject with .epochs attribute populated.
    """
    raw = subject.raw
    sid = subject.subject_id
    print(f"  Preprocessing {sid}... ", end="", flush=True)

    # Step 1: Band-pass filter
    raw = bandpass_filter(raw)

    # Step 2: Notch filter
    raw = notch_filter(raw)

    # Step 3: ICA artifact removal
    if not skip_ica:
        try:
            raw = run_ica(raw)
        except Exception as e:
            print(f"[ICA skipped: {e}] ", end="")

    # Step 4: Channel harmonization
    raw = harmonize_channels(raw)

    # Step 5: Epoch segmentation
    epochs = segment_epochs(raw)

    subject.epochs = epochs
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    n_times = epochs.get_data().shape[2]
    print(f"OK → {n_epochs} epochs × {n_channels} ch × {n_times} samples")

    return subject


def preprocess_all(subjects, skip_ica=False):
    """
    Run preprocessing on all subjects.

    Parameters
    ----------
    subjects : list of Subject
        List of subjects with .raw loaded.
    skip_ica : bool
        If True, skip ICA step.

    Returns
    -------
    subjects : list of Subject
        Same subjects with .epochs populated.
    """
    print(f"\n{'='*60}")
    print(f"PREPROCESSING PIPELINE")
    print(f"  Band-pass: {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"  Notch: {NOTCH_FREQS} Hz")
    print(f"  ICA: {'Enabled' if not skip_ica else 'Disabled'}")
    print(f"  Channels: {len(COMMON_CHANNELS)} common channels")
    print(f"  Epoch duration: {EPOCH_DURATION}s")
    print(f"{'='*60}")

    processed = []
    for subj in subjects:
        try:
            subj = preprocess_subject(subj, skip_ica=skip_ica)
            processed.append(subj)
        except Exception as e:
            print(f"  [ERROR] {subj.subject_id}: {e}")

    print(f"\nPreprocessing complete: {len(processed)}/{len(subjects)} subjects")
    return processed


if __name__ == "__main__":
    from dataset import generate_synthetic_data

    # Test with synthetic data
    subjects = generate_synthetic_data(n_subjects=4)
    subjects = preprocess_all(subjects, skip_ica=True)

    # Print summary
    for s in subjects:
        data = s.epochs.get_data()
        print(f"  {s.subject_id}: epochs shape = {data.shape}")
