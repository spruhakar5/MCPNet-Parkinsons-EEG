"""
Training and LOSO Evaluation Pipeline for MCPNet.

- Episodic training with N-way K-shot sampling
- Leave-One-Subject-Out (LOSO) cross-validation
- Prototype calibration at test time
- Metrics: accuracy, sensitivity, specificity, F1-score
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import (
    N_WAY, K_SHOTS, N_QUERY, N_EPISODES_TRAIN, N_EPISODES_TEST,
    LEARNING_RATE, N_TRAIN_EPOCHS, DEVICE, EMBEDDING_DIM, FREQ_BANDS,
)
from model import MCPNet


def create_episode(subjects, k_shot, n_query, use_plv=True):
    """
    Create a single N-way K-shot episode from a list of subjects.

    Samples k_shot support + n_query query per class from random subjects.

    Returns
    -------
    support_psd, support_plv, support_labels,
    query_psd, query_plv, query_labels
    """
    # Separate subjects by class
    pd_subjects = [s for s in subjects if s.label == 1]
    hc_subjects = [s for s in subjects if s.label == 0]

    # Randomly sample epochs for each class
    def sample_epochs(subj_list, n_samples):
        psd_list, plv_list = [], []
        remaining = n_samples
        # Shuffle subjects and sample from each
        shuffled = random.sample(subj_list, min(len(subj_list), remaining))
        for subj in shuffled:
            if remaining <= 0:
                break
            n_available = subj.psd_features.shape[0]
            n_take = min(remaining, n_available)
            indices = np.random.choice(n_available, n_take, replace=False)
            psd_list.append(subj.psd_features[indices])
            if use_plv:
                plv_list.append(subj.plv_features[indices])
            remaining -= n_take

        psd = np.concatenate(psd_list, axis=0)[:n_samples]
        plv = np.concatenate(plv_list, axis=0)[:n_samples] if use_plv else None
        return psd, plv

    total_per_class = k_shot + n_query

    pd_psd, pd_plv = sample_epochs(pd_subjects, total_per_class)
    hc_psd, hc_plv = sample_epochs(hc_subjects, total_per_class)

    # Split into support and query
    support_psd = np.concatenate([hc_psd[:k_shot], pd_psd[:k_shot]], axis=0)
    query_psd = np.concatenate([hc_psd[k_shot:k_shot+n_query],
                                pd_psd[k_shot:k_shot+n_query]], axis=0)

    support_labels = np.array([0]*k_shot + [1]*k_shot)
    query_labels = np.array([0]*n_query + [1]*n_query)

    support_plv, query_plv = None, None
    if use_plv:
        support_plv = np.concatenate([hc_plv[:k_shot], pd_plv[:k_shot]], axis=0)
        query_plv = np.concatenate([hc_plv[k_shot:k_shot+n_query],
                                    pd_plv[k_shot:k_shot+n_query]], axis=0)

    return (support_psd, support_plv, support_labels,
            query_psd, query_plv, query_labels)


def to_tensor(arr, device=DEVICE):
    """Convert numpy array to torch tensor on device."""
    if arr is None:
        return None
    return torch.tensor(arr, dtype=torch.float32).to(device)


def train_one_fold(model, train_subjects, k_shot=5, n_query=N_QUERY,
                   n_episodes=N_EPISODES_TRAIN, n_epochs=N_TRAIN_EPOCHS,
                   use_plv=True, lr=LEARNING_RATE):
    """
    Train the model for one LOSO fold using episodic training.

    Parameters
    ----------
    model : MCPNet
    train_subjects : list of Subject
    k_shot : int
    n_query : int
    n_episodes : int
        Episodes per epoch.
    n_epochs : int
        Training epochs (outer loop).
    use_plv : bool
    lr : float

    Returns
    -------
    model : MCPNet (trained)
    losses : list of float
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    epoch_losses = []

    for epoch in range(n_epochs):
        running_loss = 0.0

        for ep in range(n_episodes):
            try:
                (s_psd, s_plv, s_labels,
                 q_psd, q_plv, q_labels) = create_episode(
                    train_subjects, k_shot, n_query, use_plv
                )
            except (ValueError, IndexError):
                continue

            # To tensors
            s_psd_t = to_tensor(s_psd)
            s_plv_t = to_tensor(s_plv)
            s_labels_t = to_tensor(s_labels).long()
            q_psd_t = to_tensor(q_psd)
            q_plv_t = to_tensor(q_plv)
            q_labels_t = to_tensor(q_labels).long()

            # Forward
            log_probs, _ = model(s_psd_t, s_plv_t, s_labels_t,
                                 q_psd_t, q_plv_t)

            # Loss: negative log-likelihood
            loss = F.nll_loss(log_probs, q_labels_t)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(n_episodes, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} — Loss: {avg_loss:.4f}")

    return model, epoch_losses


def evaluate_subject(model, test_subject, train_subjects, k_shot=5,
                     use_plv=True, calibrate=True):
    """
    Evaluate the model on a single held-out subject.

    Parameters
    ----------
    model : MCPNet (trained)
    test_subject : Subject
    train_subjects : list of Subject
    k_shot : int
        Number of calibration/support samples from test subject.
    use_plv : bool
    calibrate : bool
        Whether to apply prototype calibration.

    Returns
    -------
    accuracy : float
    y_true, y_pred : lists
    """
    model.eval()

    psd = test_subject.psd_features   # (n_epochs, 32, 5)
    plv = test_subject.plv_features   # (n_epochs, 32, 32, 5)
    n_epochs = psd.shape[0]

    # Split test subject's epochs: k_shot for calibration, rest for query
    indices = np.random.permutation(n_epochs)
    cal_indices = indices[:k_shot]
    query_indices = indices[k_shot:]

    if len(query_indices) == 0:
        return 0.0, [], []

    # Build support set from training subjects
    pd_train = [s for s in train_subjects if s.label == 1]
    hc_train = [s for s in train_subjects if s.label == 0]

    def get_support(subj_list, k):
        all_psd, all_plv = [], []
        for s in subj_list:
            all_psd.append(s.psd_features)
            if use_plv:
                all_plv.append(s.plv_features)
        all_psd = np.concatenate(all_psd, axis=0)
        idx = np.random.choice(len(all_psd), k, replace=False)
        psd_out = all_psd[idx]
        plv_out = None
        if use_plv:
            all_plv = np.concatenate(all_plv, axis=0)
            plv_out = all_plv[idx]
        return psd_out, plv_out

    s_psd_hc, s_plv_hc = get_support(hc_train, k_shot)
    s_psd_pd, s_plv_pd = get_support(pd_train, k_shot)

    support_psd = np.concatenate([s_psd_hc, s_psd_pd], axis=0)
    support_labels = np.array([0]*k_shot + [1]*k_shot)
    support_plv = None
    if use_plv:
        support_plv = np.concatenate([s_plv_hc, s_plv_pd], axis=0)

    # Calibration data from test subject
    cal_psd = psd[cal_indices]
    cal_plv = plv[cal_indices] if use_plv else None
    cal_labels = np.array([test_subject.label] * k_shot)

    # Query data from test subject
    query_psd = psd[query_indices]
    query_plv = plv[query_indices] if use_plv else None
    query_labels_true = np.array([test_subject.label] * len(query_indices))

    with torch.no_grad():
        s_psd_t = to_tensor(support_psd)
        s_plv_t = to_tensor(support_plv)
        s_labels_t = to_tensor(support_labels).long()
        q_psd_t = to_tensor(query_psd)
        q_plv_t = to_tensor(query_plv)

        if calibrate:
            cal_psd_t = to_tensor(cal_psd)
            cal_plv_t = to_tensor(cal_plv)
            cal_labels_t = to_tensor(cal_labels).long()

            log_probs, predictions = model(
                s_psd_t, s_plv_t, s_labels_t,
                q_psd_t, q_plv_t,
                cal_psd_t, cal_plv_t, cal_labels_t
            )
        else:
            log_probs, predictions = model(
                s_psd_t, s_plv_t, s_labels_t,
                q_psd_t, q_plv_t
            )

    y_pred = predictions.cpu().numpy().tolist()
    y_true = query_labels_true.tolist()
    acc = accuracy_score(y_true, y_pred)

    return acc, y_true, y_pred


def loso_evaluation(subjects, k_shot=5, use_plv=True, calibrate=True,
                    n_episodes=N_EPISODES_TRAIN, n_epochs=N_TRAIN_EPOCHS):
    """
    Full Leave-One-Subject-Out evaluation.

    For each subject:
    1. Hold it out as the test subject
    2. Train model on remaining subjects
    3. Evaluate on the held-out subject
    4. Record metrics

    Parameters
    ----------
    subjects : list of Subject
    k_shot : int
    use_plv : bool
    calibrate : bool
    n_episodes : int
    n_epochs : int

    Returns
    -------
    results : dict with per-subject and aggregate metrics
    """
    print(f"\n{'='*60}")
    print(f"LOSO EVALUATION")
    print(f"  Subjects: {len(subjects)}")
    print(f"  K-shot: {k_shot}")
    print(f"  PLV: {'Yes' if use_plv else 'No'}")
    print(f"  Calibration: {'Yes' if calibrate else 'No'}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    all_y_true = []
    all_y_pred = []
    per_subject_acc = []
    subject_results = []

    for fold_idx, test_subj in enumerate(subjects):
        print(f"\n--- Fold {fold_idx+1}/{len(subjects)}: "
              f"Test={test_subj.subject_id} "
              f"({'PD' if test_subj.label==1 else 'HC'}) ---")

        # Training set: all except test subject
        train_subjs = [s for s in subjects if s.subject_id != test_subj.subject_id]

        # Check we have both classes in training
        train_labels = set(s.label for s in train_subjs)
        if len(train_labels) < 2:
            print(f"  [SKIP] Only one class in training set")
            continue

        # Initialize fresh model for each fold
        model = MCPNet(
            n_channels=32,
            n_bands=len(FREQ_BANDS),
            use_plv=use_plv,
        ).to(DEVICE)

        # Train
        model, losses = train_one_fold(
            model, train_subjs, k_shot=k_shot,
            n_episodes=n_episodes, n_epochs=n_epochs,
            use_plv=use_plv,
        )

        # Evaluate
        acc, y_true, y_pred = evaluate_subject(
            model, test_subj, train_subjs,
            k_shot=k_shot, use_plv=use_plv, calibrate=calibrate,
        )

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        per_subject_acc.append(acc)

        label_str = 'PD' if test_subj.label == 1 else 'HC'
        print(f"  {test_subj.subject_id} ({label_str}) → "
              f"Accuracy: {acc:.4f} ({sum(1 for t,p in zip(y_true,y_pred) if t==p)}"
              f"/{len(y_true)} correct)")

        subject_results.append({
            'subject_id': test_subj.subject_id,
            'dataset': test_subj.dataset,
            'true_label': test_subj.label,
            'accuracy': acc,
            'n_queries': len(y_true),
        })

    # Aggregate metrics
    if all_y_true:
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        overall_f1 = f1_score(all_y_true, all_y_pred, average='binary')
        cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        overall_acc = overall_f1 = sensitivity = specificity = 0

    mean_subject_acc = np.mean(per_subject_acc) if per_subject_acc else 0

    results = {
        'overall_accuracy': overall_acc,
        'mean_subject_accuracy': mean_subject_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': overall_f1,
        'per_subject': subject_results,
        'confusion_matrix': cm.tolist() if all_y_true else [],
    }

    print(f"\n{'='*60}")
    print(f"LOSO RESULTS (K={k_shot})")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:      {overall_acc:.4f}")
    print(f"  Mean Subject Accuracy: {mean_subject_acc:.4f}")
    print(f"  Sensitivity (Recall):  {sensitivity:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  F1 Score:              {overall_f1:.4f}")
    if all_y_true:
        print(f"  Confusion Matrix:")
        print(f"    TN={tn}  FP={fp}")
        print(f"    FN={fn}  TP={tp}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    from dataset import generate_synthetic_data
    from preprocessing import preprocess_all
    from features import extract_features_all

    # Run a quick test with synthetic data
    print("Running quick LOSO test with synthetic data...\n")

    subjects = generate_synthetic_data(n_subjects=6)
    subjects = preprocess_all(subjects, skip_ica=True)
    subjects = extract_features_all(subjects)

    # Run LOSO with small settings for testing
    results = loso_evaluation(
        subjects,
        k_shot=5,
        use_plv=True,
        calibrate=True,
        n_episodes=10,   # reduced for testing
        n_epochs=5,      # reduced for testing
    )
