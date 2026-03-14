"""
Feature Extraction: PSD and PLV.

PSD (Power Spectral Density):
  - Measures power in each frequency band per channel.
  - Output shape per epoch: (n_channels, n_bands) = (32, 5)

PLV (Phase Locking Value):
  - Measures phase synchronization between channel pairs.
  - Output shape per epoch: (n_channels, n_channels, n_bands) = (32, 32, 5)
"""

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt

from config import FREQ_BANDS


# ─────────────────────────────────────────────────────────────
# PSD Feature Extraction
# ─────────────────────────────────────────────────────────────

def compute_psd_epoch(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute PSD features for a single epoch.

    Parameters
    ----------
    epoch_data : np.ndarray
        Shape (n_channels, n_times). Single epoch EEG data.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    psd_features : np.ndarray
        Shape (n_channels, n_bands). Average power in each frequency band
        for each channel.
    """
    n_channels = epoch_data.shape[0]
    n_bands = len(FREQ_BANDS)
    psd_features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        # Welch's method: compute PSD
        freqs, pxx = welch(
            epoch_data[ch],
            fs=sfreq,
            nperseg=min(256, epoch_data.shape[1]),
            noverlap=128,
        )

        for b_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
            # Find frequency indices in the band
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            if band_mask.any():
                # Average power in this band
                psd_features[ch, b_idx] = np.mean(pxx[band_mask])

    return psd_features


def compute_psd_all_epochs(epochs_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute PSD for all epochs.

    Parameters
    ----------
    epochs_data : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    sfreq : float
        Sampling frequency.

    Returns
    -------
    psd_all : np.ndarray
        Shape (n_epochs, n_channels, n_bands).
    """
    n_epochs = epochs_data.shape[0]
    n_channels = epochs_data.shape[1]
    n_bands = len(FREQ_BANDS)

    psd_all = np.zeros((n_epochs, n_channels, n_bands))
    for i in range(n_epochs):
        psd_all[i] = compute_psd_epoch(epochs_data[i], sfreq)

    return psd_all


# ─────────────────────────────────────────────────────────────
# PLV Feature Extraction
# ─────────────────────────────────────────────────────────────

def bandpass_filter_signal(data: np.ndarray, sfreq: float,
                           low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Apply a band-pass Butterworth filter to extract a specific frequency band.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_times).
    sfreq : float
        Sampling frequency.
    low, high : float
        Band edges in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered : np.ndarray
        Band-pass filtered data, same shape as input.
    """
    nyq = sfreq / 2.0
    low_norm = max(low / nyq, 0.001)
    high_norm = min(high / nyq, 0.999)

    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def compute_plv_epoch(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute PLV (Phase Locking Value) for a single epoch across all
    frequency bands.

    PLV between channels i and j:
        PLV_ij = |1/T * sum_t( exp(j * (phi_i(t) - phi_j(t))) )|

    where phi is the instantaneous phase from Hilbert transform.

    Parameters
    ----------
    epoch_data : np.ndarray
        Shape (n_channels, n_times).
    sfreq : float
        Sampling frequency.

    Returns
    -------
    plv_features : np.ndarray
        Shape (n_channels, n_channels, n_bands).
    """
    n_channels = epoch_data.shape[0]
    n_bands = len(FREQ_BANDS)
    plv_features = np.zeros((n_channels, n_channels, n_bands))

    for b_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        # Band-pass filter for this frequency band
        filtered = bandpass_filter_signal(epoch_data, sfreq, fmin, fmax)

        # Hilbert transform to get analytic signal
        analytic = hilbert(filtered, axis=-1)

        # Extract instantaneous phase
        phases = np.angle(analytic)  # shape: (n_channels, n_times)

        # Compute PLV for all channel pairs
        n_times = phases.shape[1]
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Phase difference
                phase_diff = phases[i] - phases[j]

                # PLV = magnitude of mean complex exponential
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))

                plv_features[i, j, b_idx] = plv
                plv_features[j, i, b_idx] = plv  # symmetric

        # Diagonal is 1 (perfect self-synchrony)
        np.fill_diagonal(plv_features[:, :, b_idx], 1.0)

    return plv_features


def compute_plv_all_epochs(epochs_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute PLV for all epochs.

    Parameters
    ----------
    epochs_data : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    sfreq : float
        Sampling frequency.

    Returns
    -------
    plv_all : np.ndarray
        Shape (n_epochs, n_channels, n_channels, n_bands).
    """
    n_epochs = epochs_data.shape[0]
    n_channels = epochs_data.shape[1]
    n_bands = len(FREQ_BANDS)

    plv_all = np.zeros((n_epochs, n_channels, n_channels, n_bands))
    for i in range(n_epochs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    PLV epoch {i+1}/{n_epochs}...", flush=True)
        plv_all[i] = compute_plv_epoch(epochs_data[i], sfreq)

    return plv_all


# ─────────────────────────────────────────────────────────────
# Combined Feature Extraction
# ─────────────────────────────────────────────────────────────

def extract_features(subject):
    """
    Extract PSD and PLV features for a subject.

    Parameters
    ----------
    subject : Subject
        Must have .epochs populated.

    Returns
    -------
    subject : Subject
        With .psd_features and .plv_features populated.
    """
    epochs_data = subject.epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = subject.epochs.info['sfreq']
    sid = subject.subject_id

    print(f"  Extracting features for {sid} "
          f"({epochs_data.shape[0]} epochs)...")

    # PSD: (n_epochs, n_channels, n_bands)
    print(f"    Computing PSD...", flush=True)
    subject.psd_features = compute_psd_all_epochs(epochs_data, sfreq)
    print(f"    PSD shape: {subject.psd_features.shape}")

    # PLV: (n_epochs, n_channels, n_channels, n_bands)
    print(f"    Computing PLV...", flush=True)
    subject.plv_features = compute_plv_all_epochs(epochs_data, sfreq)
    print(f"    PLV shape: {subject.plv_features.shape}")

    return subject


def extract_features_all(subjects):
    """Extract features for all subjects."""
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION")
    print(f"  PSD: {len(FREQ_BANDS)} frequency bands × channels")
    print(f"  PLV: pairwise phase locking × {len(FREQ_BANDS)} bands")
    print(f"{'='*60}")

    for subj in subjects:
        try:
            extract_features(subj)
        except Exception as e:
            print(f"  [ERROR] {subj.subject_id}: {e}")

    return subjects


if __name__ == "__main__":
    from dataset import generate_synthetic_data
    from preprocessing import preprocess_all

    # Quick test with synthetic data
    subjects = generate_synthetic_data(n_subjects=2)
    subjects = preprocess_all(subjects, skip_ica=True)
    subjects = extract_features_all(subjects)

    for s in subjects:
        print(f"\n{s.subject_id} ({'PD' if s.label==1 else 'HC'}):")
        print(f"  PSD: {s.psd_features.shape}")
        print(f"  PLV: {s.plv_features.shape}")
        print(f"  PSD sample (ch0, all bands): {s.psd_features[0, 0, :]}")
        print(f"  PLV sample (ch0-ch1, all bands): {s.plv_features[0, 0, 1, :]}")
