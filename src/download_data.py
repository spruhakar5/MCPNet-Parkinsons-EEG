"""
Auto-download EEG datasets from OpenNeuro.

Downloads the three datasets used in the MCPNet paper:
1. UC San Diego (ds003490) - 15 PD + 16 HC subjects
2. UNM (ds002778) - 14 PD + 14 HC subjects
3. Iowa (ds004584) - 14 PD + 14 HC subjects

Usage:
    python download_data.py                # Download all datasets
    python download_data.py --dataset UC   # Download only UC dataset
    python download_data.py --dataset UNM  # Download only UNM dataset
    python download_data.py --dataset Iowa # Download only Iowa dataset

Requirements:
    pip install openneuro-py
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# OpenNeuro dataset IDs and target directories
DATASET_INFO = {
    'UC': {
        'openneuro_id': 'ds003490',
        'description': 'UC San Diego EEG Parkinson\'s Dataset',
        'subjects': '15 PD + 16 HC = 31 subjects',
        'sfreq': '512 Hz',
        'duration': '~3 min per subject',
        'format': 'BrainVision (.vhdr/.eeg/.vmrk)',
    },
    'UNM': {
        'openneuro_id': 'ds002778',
        'description': 'University of New Mexico EEG Dataset',
        'subjects': '14 PD + 14 HC = 28 subjects',
        'sfreq': '500 Hz',
        'duration': '~2 min per subject',
        'format': 'EDF / BrainVision',
    },
    'Iowa': {
        'openneuro_id': 'ds004584',
        'description': 'University of Iowa EEG Dataset',
        'subjects': '14 PD + 14 HC = 28 subjects',
        'sfreq': '500 Hz',
        'duration': '~2 min per subject',
        'format': 'EDF / BrainVision',
    },
}

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def check_openneuro_installed():
    """Check if openneuro-py is installed."""
    try:
        import openneuro
        return True
    except ImportError:
        return False


def install_openneuro():
    """Install openneuro-py."""
    print("Installing openneuro-py...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
    print("openneuro-py installed successfully.\n")


def download_dataset(dataset_name):
    """Download a single dataset from OpenNeuro."""
    if dataset_name not in DATASET_INFO:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASET_INFO.keys())}")
        return False

    info = DATASET_INFO[dataset_name]
    ds_id = info['openneuro_id']
    target_dir = DATA_DIR / dataset_name

    print(f"\n{'='*60}")
    print(f"Downloading: {info['description']}")
    print(f"  OpenNeuro ID: {ds_id}")
    print(f"  Subjects: {info['subjects']}")
    print(f"  Sampling rate: {info['sfreq']}")
    print(f"  Duration: {info['duration']}")
    print(f"  Format: {info['format']}")
    print(f"  Target: {target_dir}")
    print(f"{'='*60}\n")

    # Check if already downloaded
    if target_dir.exists() and any(target_dir.rglob("*.vhdr")) or any(target_dir.rglob("*.edf")):
        print(f"  Dataset already exists at {target_dir}")
        print(f"  Skipping download. Delete the folder to re-download.\n")
        return True

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openneuro
        print(f"  Downloading {ds_id} via openneuro-py...")
        openneuro.download(dataset=ds_id, target_dir=str(target_dir))
        print(f"\n  Download complete: {target_dir}")
        return True
    except Exception as e:
        print(f"\n  openneuro-py download failed: {e}")
        print(f"  Trying fallback with DataLad...")
        return download_with_datalad(ds_id, target_dir)


def download_with_datalad(ds_id, target_dir):
    """Fallback: try downloading with datalad or direct AWS."""
    # Try using the OpenNeuro S3 bucket directly
    s3_url = f"s3://openneuro.org/{ds_id}"

    print(f"\n  Attempting download via AWS CLI from {s3_url}...")
    print(f"  (Requires: pip install awscli)")

    try:
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request", s3_url, str(target_dir)],
            check=True,
        )
        print(f"  Download complete: {target_dir}")
        return True
    except FileNotFoundError:
        print("  AWS CLI not found. Install with: pip install awscli")
    except subprocess.CalledProcessError as e:
        print(f"  AWS download failed: {e}")

    # Final fallback: print manual instructions
    print(f"\n  === MANUAL DOWNLOAD REQUIRED ===")
    print(f"  1. Go to: https://openneuro.org/datasets/{ds_id}")
    print(f"  2. Click 'Download' and select the version")
    print(f"  3. Extract to: {target_dir}")
    print(f"  ================================\n")
    return False


def verify_dataset(dataset_name):
    """Verify a downloaded dataset has expected files."""
    target_dir = DATA_DIR / dataset_name
    if not target_dir.exists():
        print(f"  [{dataset_name}] Not found at {target_dir}")
        return False

    # Count EEG files
    eeg_patterns = ["**/*.vhdr", "**/*.edf", "**/*.bdf", "**/*.set"]
    eeg_files = []
    for pat in eeg_patterns:
        eeg_files.extend(target_dir.rglob(pat.replace("**/", "")))

    # Count subjects
    sub_dirs = [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]

    # Check for participants.tsv
    has_tsv = (target_dir / "participants.tsv").exists()

    print(f"  [{dataset_name}]")
    print(f"    Path: {target_dir}")
    print(f"    Subject folders: {len(sub_dirs)}")
    print(f"    EEG files: {len(eeg_files)}")
    print(f"    participants.tsv: {'Found' if has_tsv else 'MISSING'}")

    if not has_tsv:
        print(f"    WARNING: participants.tsv is needed for labels (PD/HC)")

    return len(eeg_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Download EEG datasets from OpenNeuro")
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['UC', 'UNM', 'Iowa', 'all'],
                        help='Which dataset to download (default: all)')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify existing downloads, do not download')
    parser.add_argument('--info', action='store_true',
                        help='Print dataset information and exit')
    args = parser.parse_args()

    # Print info
    if args.info:
        print("\nAvailable EEG Datasets:\n")
        for name, info in DATASET_INFO.items():
            print(f"  {name}:")
            for k, v in info.items():
                print(f"    {k}: {v}")
            print()
        return

    # Verify only
    if args.verify:
        print("\nVerifying downloaded datasets:\n")
        for name in DATASET_INFO:
            verify_dataset(name)
        return

    # Install openneuro if needed
    if not check_openneuro_installed():
        install_openneuro()

    # Download
    datasets_to_download = [args.dataset] if args.dataset and args.dataset != 'all' else list(DATASET_INFO.keys())

    print(f"\nData directory: {DATA_DIR}")
    print(f"Datasets to download: {datasets_to_download}\n")

    results = {}
    for ds_name in datasets_to_download:
        success = download_dataset(ds_name)
        results[ds_name] = success

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "OK" if success else "FAILED/MANUAL"
        print(f"  {name}: {status}")

    # Verify all
    print(f"\nVerifying downloads:")
    for name in datasets_to_download:
        verify_dataset(name)

    print(f"\nNext step: cd src && python main.py --real --k_shot 5")


if __name__ == "__main__":
    main()
