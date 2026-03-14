"""
MCPNet Main Pipeline Runner.

Usage:
    python main.py                    # Run with synthetic data (for testing)
    python main.py --real             # Run with real datasets
    python main.py --k_shot 5        # Set K-shot value
    python main.py --no-plv          # Disable PLV features
    python main.py --no-calibration  # Disable prototype calibration
"""

import argparse
import json
import time
from pathlib import Path

from config import DATA_PROCESSED, K_SHOTS, N_EPISODES_TRAIN, N_TRAIN_EPOCHS
from dataset import load_all_datasets, generate_synthetic_data
from preprocessing import preprocess_all
from features import extract_features_all
from train import loso_evaluation


def save_results(results, filename):
    """Save results to JSON."""
    out_dir = DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {path}")


def run_pipeline(args):
    """Run the full MCPNet pipeline."""

    print("╔══════════════════════════════════════════════════════╗")
    print("║   MCPNet: Multiscale Convolutional Prototype Net    ║")
    print("║   EEG-Based Parkinson's Disease Detection           ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Step 1: Load data ──
    if args.real:
        subjects = load_all_datasets()
        if not subjects:
            print("\nNo real data found. Use --synthetic or download datasets.")
            return
    else:
        subjects = generate_synthetic_data(n_subjects=args.n_subjects)

    # ── Step 2: Preprocess ──
    subjects = preprocess_all(subjects, skip_ica=args.skip_ica)

    # ── Step 3: Extract features ──
    subjects = extract_features_all(subjects)

    # ── Step 4: LOSO evaluation ──
    k_shot_list = [args.k_shot] if args.k_shot else K_SHOTS
    all_results = {}

    for k in k_shot_list:
        print(f"\n{'#'*60}")
        print(f"# Running LOSO with K={k}")
        print(f"{'#'*60}")

        start_time = time.time()

        results = loso_evaluation(
            subjects,
            k_shot=k,
            use_plv=args.use_plv,
            calibrate=args.calibrate,
            n_episodes=args.n_episodes,
            n_epochs=args.n_epochs,
        )

        elapsed = time.time() - start_time
        results['elapsed_seconds'] = elapsed
        all_results[f'k{k}'] = results

        print(f"\nCompleted K={k} in {elapsed:.1f}s")

    # ── Step 5: Save results ──
    save_results(all_results, 'loso_results.json')

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL K-SHOT SETTINGS")
    print(f"{'='*60}")
    print(f"{'K':>5} {'Accuracy':>10} {'Sensitivity':>12} "
          f"{'Specificity':>12} {'F1':>8}")
    print(f"{'-'*50}")
    for k_key, res in all_results.items():
        print(f"{k_key:>5} {res['overall_accuracy']:>10.4f} "
              f"{res['sensitivity']:>12.4f} "
              f"{res['specificity']:>12.4f} "
              f"{res['f1_score']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCPNet Pipeline")
    parser.add_argument('--real', action='store_true',
                        help='Use real EEG datasets')
    parser.add_argument('--n_subjects', type=int, default=10,
                        help='Number of synthetic subjects (if not --real)')
    parser.add_argument('--skip_ica', action='store_true', default=True,
                        help='Skip ICA (faster, default for synthetic)')
    parser.add_argument('--k_shot', type=int, default=None,
                        help='Specific K-shot value (default: run all)')
    parser.add_argument('--use_plv', action='store_true', default=True,
                        help='Use PLV features')
    parser.add_argument('--no-plv', dest='use_plv', action='store_false',
                        help='Disable PLV features')
    parser.add_argument('--calibrate', action='store_true', default=True,
                        help='Use prototype calibration')
    parser.add_argument('--no-calibration', dest='calibrate',
                        action='store_false',
                        help='Disable prototype calibration')
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES_TRAIN,
                        help='Episodes per training epoch')
    parser.add_argument('--n_epochs', type=int, default=N_TRAIN_EPOCHS,
                        help='Training epochs')

    args = parser.parse_args()
    run_pipeline(args)
