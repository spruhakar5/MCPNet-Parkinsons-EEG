# MCPNet — Complete Code Explanation

A line-by-line walkthrough of the entire codebase for the MCPNet EEG-based Parkinson's Disease detection pipeline.

**Authors**: Spruha Kar, Aarsh, Aaryan — ML_RP@DTU
**Paper**: Qiu et al. (2024) — *A Novel EEG-Based Parkinson's Disease Detection Model Using Multiscale Convolutional Prototype Networks*

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [config.py — Configuration](#2-configpy--configuration)
3. [download_data.py — Dataset Download](#3-download_datapy--dataset-download)
4. [dataset.py — Data Loading](#4-datasetpy--data-loading)
5. [preprocessing.py — EEG Signal Cleaning](#5-preprocessingpy--eeg-signal-cleaning)
6. [features.py — PSD and PLV Extraction](#6-featurespy--psd-and-plv-extraction)
7. [model.py — MCPNet Architecture](#7-modelpy--mcpnet-architecture)
8. [train.py — Training and Evaluation](#8-trainpy--training-and-evaluation)
9. [main.py — Pipeline Runner](#9-mainpy--pipeline-runner)
10. [How Everything Connects](#10-how-everything-connects)
11. [Key Concepts Glossary](#11-key-concepts-glossary)

---

## 1. High-Level Architecture

The pipeline has 5 stages. Data flows through them sequentially:

```
Stage 1: LOAD DATA
    ↓
Stage 2: PREPROCESS (filter + clean + segment)
    ↓
Stage 3: EXTRACT FEATURES (PSD + PLV)
    ↓
Stage 4: TRAIN MODEL (episodic few-shot learning)
    ↓
Stage 5: EVALUATE (LOSO cross-validation)
```

Each stage is handled by a separate Python file. The `main.py` orchestrates them all.

---

## 2. `config.py` — Configuration

**Purpose**: Single source of truth for every setting in the pipeline. Change a value here, and it propagates everywhere.

### Project Paths

```python
PROJECT_ROOT = Path(__file__).parent.parent   # MCPNet/ directory
DATA_RAW = PROJECT_ROOT / "data" / "raw"      # where raw EEG files go
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # where results are saved
```

`Path(__file__)` gives the path of `config.py` itself. `.parent.parent` goes up two levels: from `src/config.py` → `src/` → `MCPNet/`.

### Channel Configuration

```python
COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    ...
]
```

These are the **32 EEG electrode positions** from the international 10-20 system. The naming convention tells you where on the scalp:
- **F** = Frontal, **C** = Central, **P** = Parietal, **O** = Occipital, **T** = Temporal
- **Odd numbers** = left hemisphere, **Even numbers** = right hemisphere
- **z** = midline (zero)

All three datasets must be standardized to these same 32 channels so the model sees consistent input.

### Frequency Bands

```python
FREQ_BANDS = {
    'delta': (0.5, 4),    # deep sleep, unconscious processes
    'theta': (4, 8),      # drowsiness, memory — INCREASED in PD
    'alpha': (8, 13),     # relaxed wakefulness
    'beta':  (13, 30),    # motor planning — DECREASED in PD
    'gamma': (30, 50),    # higher cognition
}
```

These are the standard EEG frequency bands used in neuroscience. PD primarily affects **theta** (increases due to cortical slowing) and **beta** (decreases due to motor dysfunction). This is why PSD features are useful for PD detection.

### Dataset-Specific Settings

```python
DATASETS = {
    'UC': {
        'sfreq': 512,              # samples per second
        'duration_sec': 180,       # 3 minutes of recording
        'n_epochs': 180,           # 180 one-second windows
        'pd_subjects': 15,
        'hc_subjects': 16,
    },
    'Iowa': {
        ...
        'channel_remap': {'Pz': 'Fz'},  # Iowa has Pz instead of Fz
    },
}
```

Each dataset has different recording parameters. The Iowa dataset needs a special channel remapping because it recorded Pz instead of Fz.

### Few-Shot Learning Settings

```python
N_WAY = 2          # binary classification: PD vs HC
K_SHOTS = [1, 5, 10, 20]  # how many support samples per class
N_QUERY = 15       # query samples per class per episode
N_EPISODES_TRAIN = 100     # episodes per training epoch
```

**N-way K-shot** means: given K labeled examples from each of N classes, classify new samples. We use 2-way (PD vs HC) with varying K.

### Model Hyperparameters

```python
EMBEDDING_DIM = 128        # dimensionality of the feature embedding
KERNEL_SIZES = [3, 5, 7]   # the three parallel conv branch sizes
LEARNING_RATE = 1e-3        # Adam optimizer learning rate
CALIBRATION_ALPHA = 0.5     # 50% original prototype + 50% test subject
```

### Device Selection

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Uses GPU if available (NVIDIA CUDA), otherwise falls back to CPU.

---

## 3. `download_data.py` — Dataset Download

**Purpose**: Auto-fetches the three EEG datasets from OpenNeuro so users don't need to manually download anything.

### Key Components

```python
DATASET_INFO = {
    'UC': {'openneuro_id': 'ds003490', ...},
    'UNM': {'openneuro_id': 'ds002778', ...},
    'Iowa': {'openneuro_id': 'ds004584', ...},
}
```

Maps our dataset names to their OpenNeuro identifiers.

### Download Strategy (3 fallbacks)

1. **`openneuro-py`** library — preferred method
2. **AWS S3 direct download** — OpenNeuro stores data on AWS
3. **Manual instructions** — prints the URL for the user

```python
def download_dataset(dataset_name):
    try:
        openneuro.download(dataset=ds_id, target_dir=str(target_dir))
    except:
        download_with_datalad(ds_id, target_dir)  # fallback
```

### Verification

```python
def verify_dataset(dataset_name):
    # Counts EEG files, subject folders, checks for participants.tsv
```

After downloading, this confirms the data is complete and properly structured.

---

## 4. `dataset.py` — Data Loading

**Purpose**: Load EEG recordings into memory and organize them as `Subject` objects that flow through the pipeline.

### The Subject Dataclass

```python
@dataclass
class Subject:
    subject_id: str          # e.g., "sub-001"
    dataset: str             # 'UC', 'UNM', or 'Iowa'
    label: int               # 0 = HC (healthy), 1 = PD (Parkinson's)
    raw: mne.io.BaseRaw      # the raw EEG signal (filled by loading)
    epochs: mne.Epochs       # segmented EEG (filled by preprocessing)
    psd_features: np.ndarray # PSD matrix (filled by feature extraction)
    plv_features: np.ndarray # PLV matrix (filled by feature extraction)
```

A `Subject` is the main data container. It starts with just an ID and label, and gets progressively enriched as it passes through each pipeline stage.

### Loading Real Data

```python
def load_raw_eeg(filepath, dataset_name):
    ext = filepath.suffix.lower()
    if ext == '.set':
        raw = mne.io.read_raw_eeglab(str(filepath), preload=True)
    elif ext == '.edf':
        raw = mne.io.read_raw_edf(str(filepath), preload=True)
    ...
```

**MNE** is the standard Python library for EEG/MEG analysis. It can read many formats:
- `.set` = EEGLAB format (MATLAB-based)
- `.edf` = European Data Format (standard medical)
- `.bdf` = BioSemi Data Format
- `.fif` = MNE's native format

The function auto-detects the format from the file extension and loads the data.

### Label Loading

```python
def load_participants_tsv(dataset_dir):
    # Reads participants.tsv (standard BIDS format)
    # Returns: {'sub-001': 1, 'sub-002': 0, ...}
```

BIDS (Brain Imaging Data Structure) is the standard way to organize neuroimaging data. The `participants.tsv` file contains one row per subject with their diagnosis.

### Synthetic Data Generator

```python
def generate_synthetic_data(n_subjects=20, ...):
    for i in range(n_subjects):
        # Mix sine waves at different frequencies
        delta = 20 * np.sin(2 * np.pi * 2 * t + ...)   # 2 Hz
        theta = 10 * np.sin(2 * np.pi * 6 * t + ...)   # 6 Hz
        alpha = 8 * np.sin(2 * np.pi * 10 * t + ...)   # 10 Hz
        beta = 5 * np.sin(2 * np.pi * 20 * t + ...)    # 20 Hz
        gamma = 2 * np.sin(2 * np.pi * 40 * t + ...)   # 40 Hz

        if label == 1:  # PD subjects
            theta *= 1.5   # MORE theta (cortical slowing)
            beta *= 0.6    # LESS beta (motor dysfunction)
```

This creates fake EEG signals by combining sine waves at different frequencies. PD subjects get amplified theta and reduced beta — mimicking the real spectral changes observed in Parkinson's. This allows testing the pipeline without downloading real data.

**Why `data *= 1e-6`?** EEG signals are measured in **microvolts** (µV). MNE expects data in Volts, so we multiply by 10^-6 to convert.

---

## 5. `preprocessing.py` — EEG Signal Cleaning

**Purpose**: Take noisy raw EEG and clean it through 5 steps to get analysis-ready epochs.

### Step 1: Band-pass Filter (0.5–50 Hz)

```python
def bandpass_filter(raw):
    raw_filtered = raw.copy().filter(
        l_freq=0.5,      # low cutoff
        h_freq=50.0,     # high cutoff
        method='fir',     # Finite Impulse Response filter
        fir_design='firwin',
    )
```

**What it does**: Keeps only frequencies between 0.5 and 50 Hz.
- **Below 0.5 Hz**: Slow drift from electrode movement, breathing, sweat — not brain activity
- **Above 50 Hz**: Mostly muscle artifacts and electrical noise — not useful for PD analysis
- **FIR filter**: A type of digital filter that has a finite response. `firwin` is a specific design method that creates optimal filters.

### Step 2: Notch Filter (50/60 Hz)

```python
def notch_filter(raw):
    raw_notched = raw.copy().notch_filter(
        freqs=[50.0, 60.0],  # removes both
    )
```

**What it does**: Removes the exact frequencies of power line interference.
- Power lines oscillate at 50 Hz (India, Europe) or 60 Hz (US)
- This creates a sharp spike in the EEG spectrum that would contaminate gamma band analysis
- A notch filter removes a very narrow frequency range (e.g., 49-51 Hz) while keeping everything else
- We apply both 50 and 60 Hz — if one isn't present, the filter has no effect

### Step 3: ICA Artifact Removal

```python
def run_ica(raw, n_components=20, random_state=42):
    ica = ICA(n_components=20, method='fastica')
    ica.fit(raw)
```

**ICA (Independent Component Analysis)** is a mathematical technique that separates mixed signals into independent sources. Think of it like this:

Imagine you're in a room with 3 people talking. Your ears hear a mixture of all 3 voices. ICA can separate the mixture back into the 3 individual voices.

In EEG:
- The recorded signals are **mixtures** of brain activity + eye blinks + muscle activity + heartbeat
- ICA separates these into independent **components**
- We identify artifact components and remove them
- The remaining components (brain sources) are reconstructed back into clean EEG

**Automatic artifact detection:**

```python
# Method 1: Use frontal channels to find eye blink components
indices, scores = ica.find_bads_eog(raw, ch_name='Fp1')

# Method 2 (fallback): Use kurtosis
# Eye blinks produce high kurtosis (sharp, peaky distribution)
kurtosis = np.mean(s**4) / (np.mean(s**2)**2) - 3
```

**Kurtosis** measures how "peaked" a distribution is. Eye blinks create brief, large-amplitude spikes, which produce high kurtosis in the corresponding ICA component. We flag components with kurtosis > mean + 2*std as artifacts.

### Step 4: Channel Harmonization

```python
def harmonize_channels(raw, target_channels=COMMON_CHANNELS):
    # Case-insensitive matching
    for target in target_channels:
        for avail in available:
            if avail.lower() == target.lower():
                ch_map[target] = avail
    raw_picked = raw.copy().pick(found)
```

**What it does**: Different datasets may have different channel names or extra channels. This function:
1. Matches available channels to our standard 32-channel list (case-insensitive)
2. Picks only those channels and drops extras
3. Renames them to standardized names

**Why needed**: The model expects exactly 32 channels in a specific order. Without harmonization, channels from different datasets wouldn't align and the model couldn't process them uniformly.

### Step 5: Epoch Segmentation

```python
def segment_epochs(raw, duration=1.0):
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration - 1/sfreq,
                        baseline=None, preload=True)
```

**What it does**: Cuts the continuous EEG recording into 1-second non-overlapping chunks.

**Why 1 second?**
- EEG is non-stationary over long periods — spectral properties change. Within 1 second, we can assume approximate stationarity
- Gives enough frequency resolution: 1 second at 500 Hz = 500 data points, which is sufficient for PSD estimation
- Generates many samples per subject: a 3-minute recording → 180 one-second epochs

**`tmax=duration - 1/sfreq`**: This ensures each epoch has exactly `sfreq` samples (e.g., 500 samples at 500 Hz). Without the `-1/sfreq`, we'd get 501 samples.

### Putting It Together

```python
def preprocess_subject(subject, skip_ica=False):
    raw = bandpass_filter(raw)      # Step 1
    raw = notch_filter(raw)         # Step 2
    if not skip_ica:
        raw = run_ica(raw)          # Step 3
    raw = harmonize_channels(raw)   # Step 4
    epochs = segment_epochs(raw)    # Step 5
    subject.epochs = epochs
```

Each step transforms the data and passes it to the next. The `Subject` object's `.epochs` field is populated with the clean, segmented data.

---

## 6. `features.py` — PSD and PLV Extraction

**Purpose**: Transform raw EEG epochs into two types of features that capture different aspects of brain activity.

### PSD: Power Spectral Density

**Concept**: PSD tells you "how much power (energy) does the brain produce in each frequency band at each electrode?"

```python
def compute_psd_epoch(epoch_data, sfreq):
    for ch in range(n_channels):
        freqs, pxx = welch(epoch_data[ch], fs=sfreq, nperseg=256, noverlap=128)
```

**Welch's method** works like this:
1. Divide the 1-second signal into overlapping segments (256 samples each, 128 overlap)
2. Apply a window function to each segment (reduces spectral leakage)
3. Compute the FFT (Fast Fourier Transform) of each segment
4. Average the squared magnitudes across segments

The result `pxx` is the power at each frequency. We then average within each band:

```python
for b_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    psd_features[ch, b_idx] = np.mean(pxx[band_mask])
```

**Output shape**: `(32 channels, 5 bands)` per epoch. So for each channel, we get 5 numbers representing how much delta, theta, alpha, beta, and gamma power is present.

### PLV: Phase Locking Value

**Concept**: PLV tells you "how synchronized are two brain regions?" It measures whether the oscillations at two electrodes maintain a consistent phase relationship.

**Step 1: Band-pass filter for each frequency band**

```python
def bandpass_filter_signal(data, sfreq, low, high, order=4):
    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, data, axis=-1)
```

We need to compute PLV separately for each frequency band, so we first isolate each band with a Butterworth filter. `filtfilt` applies the filter forward and backward, which eliminates phase distortion.

**Step 2: Hilbert transform to get instantaneous phase**

```python
analytic = hilbert(filtered, axis=-1)
phases = np.angle(analytic)
```

The **Hilbert transform** converts a real signal into a complex-valued **analytic signal**. The angle (argument) of this complex signal gives the **instantaneous phase** — at every time point, we know the "position" of the oscillation cycle (0° to 360°).

**Step 3: Compute phase locking**

```python
for i in range(n_channels):
    for j in range(i + 1, n_channels):
        phase_diff = phases[i] - phases[j]
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
```

**Breaking this down:**
- `phases[i] - phases[j]`: The phase difference between channels i and j at every time point
- `np.exp(1j * phase_diff)`: Converts each phase difference to a unit vector on the complex plane
- `np.mean(...)`: Averages all these unit vectors
- `np.abs(...)`: Takes the magnitude of the average

**Intuition**: If two channels always have the same phase difference (e.g., always 45° apart), all the unit vectors point in the same direction → their average has magnitude close to 1 → PLV ≈ 1 (highly synchronized).

If the phase difference is random → unit vectors point in random directions → they cancel out → magnitude ≈ 0 → PLV ≈ 0 (not synchronized).

**Output shape**: `(32, 32, 5)` per epoch. A 32×32 connectivity matrix for each of the 5 frequency bands. This is symmetric (PLV(i,j) = PLV(j,i)) with diagonal = 1.

### Why Both Features?

| Feature | Captures | Level | PD Signature |
|---------|----------|-------|-------------|
| PSD | How active is each brain region? | Local (per channel) | Theta↑, Beta↓ |
| PLV | How synchronized are brain regions? | Network (pairs) | Disrupted connectivity |

PD affects both local activity AND network communication. Using both gives the model a complete picture.

---

## 7. `model.py` — MCPNet Architecture

**Purpose**: The neural network that learns to classify PD vs HC from the extracted features.

### ConvBranch — Single Scale Feature Extractor

```python
class ConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
```

**Layer by layer:**
1. **Conv2d**: A 2D convolution that slides a kernel (small matrix) across the input, computing dot products. Detects spatial patterns.
   - `padding = kernel_size // 2` keeps the output the same size as input ("same" padding)
2. **BatchNorm2d**: Normalizes each feature map to have zero mean and unit variance. Stabilizes training and allows higher learning rates.
3. **ReLU** (via `F.relu`): The activation function. Replaces negative values with 0. Introduces non-linearity.
4. **AdaptiveAvgPool2d(1)**: Reduces any spatial size to 1×1 by averaging. This makes the output size independent of input size.

**Forward pass:**
```python
def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))    # conv → normalize → activate
    x = F.relu(self.bn2(self.conv2(x)))    # second conv block
    x = self.pool(x)                        # global average pool → (batch, channels, 1, 1)
    x = x.reshape(x.size(0), -1)           # flatten → (batch, channels)
    return x
```

### MultiscaleEncoder — Parallel Multi-Resolution Processing

```python
class MultiscaleEncoder(nn.Module):
    def __init__(self, in_channels, branch_dim=64, kernel_sizes=[3, 5, 7]):
        self.branches = nn.ModuleList([
            ConvBranch(in_channels, branch_dim, ks) for ks in kernel_sizes
        ])
        self.fc = nn.Linear(branch_dim * 3, EMBEDDING_DIM)  # 64*3=192 → 128
```

**Why multiple kernel sizes?**
- **3×3 kernel**: Small receptive field → detects fine-grained, local patterns (e.g., a specific channel's power spike in one band)
- **5×5 kernel**: Medium receptive field → detects regional patterns (e.g., frontal vs parietal differences)
- **7×7 kernel**: Large receptive field → detects global patterns (e.g., overall spectral tilt)

PD-related changes exist at all these scales, so capturing all of them gives a richer representation.

**Forward pass:**
```python
def forward(self, x):
    branch_outputs = [branch(x) for branch in self.branches]  # 3 parallel paths
    concatenated = torch.cat(branch_outputs, dim=1)            # merge: 3 × 64 = 192
    embedding = F.relu(self.bn(self.fc(concatenated)))         # project to 128-dim
    return embedding
```

### MCPNet — The Full Model

```python
class MCPNet(nn.Module):
    def __init__(self, n_channels=32, n_bands=5, use_plv=True):
        self.psd_encoder = MultiscaleEncoder(in_channels=1)       # PSD path
        self.plv_encoder = MultiscaleEncoder(in_channels=n_bands) # PLV path
        self.fusion = nn.Sequential(                               # combine both
            nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(),
        )
```

The model has **two parallel encoders**:
- **PSD encoder**: Takes input shape `(batch, 1, 32, 5)` — treat the 32×5 PSD matrix as a 1-channel "image"
- **PLV encoder**: Takes input shape `(batch, 5, 32, 32)` — treat each frequency band's 32×32 connectivity matrix as a separate channel (like RGB in images, but 5 channels)

The fusion layer combines both 128-dim embeddings into a single 128-dim vector.

### Encoding

```python
def encode(self, psd, plv=None):
    psd_input = psd.unsqueeze(1)                    # (batch, 32, 5) → (batch, 1, 32, 5)
    psd_emb = self.psd_encoder(psd_input)           # → (batch, 128)

    plv_input = plv.permute(0, 3, 1, 2).contiguous()  # (batch, 32, 32, 5) → (batch, 5, 32, 32)
    plv_emb = self.plv_encoder(plv_input)           # → (batch, 128)

    combined = torch.cat([psd_emb, plv_emb], dim=1) # → (batch, 256)
    embedding = self.fusion(combined)                # → (batch, 128)
```

**`.unsqueeze(1)`**: Adds a dimension at position 1. Conv2d expects `(batch, channels, height, width)`, but PSD is `(batch, 32, 5)`. Adding a channel dimension makes it `(batch, 1, 32, 5)`.

**`.permute(0, 3, 1, 2)`**: Rearranges dimensions. PLV comes as `(batch, 32, 32, 5)` but Conv2d needs channels as dim 1, so we move the bands dimension from position 3 to position 1.

**`.contiguous()`**: After permute, memory layout may not be sequential. This ensures it is, preventing errors during convolution.

### Prototype Computation

```python
def compute_prototypes(self, support_embeddings, support_labels):
    for c in sorted(classes.tolist()):
        mask = support_labels == c
        prototype = support_embeddings[mask].mean(dim=0)
        prototypes.append(prototype)
```

**What this does**: For each class (0=HC, 1=PD), takes all support samples of that class, embeds them, and averages the embeddings. The average is the **prototype** — a representative point in 128-dimensional space for that class.

**Example with K=5**:
- 5 HC support samples → encode each → get 5 vectors of size 128 → average → 1 HC prototype
- 5 PD support samples → encode each → get 5 vectors of size 128 → average → 1 PD prototype

### Prototype Calibration

```python
def calibrate_prototypes(self, prototypes, calibration_embeddings, calibration_labels):
    calibrated[i] = (self.alpha * prototypes[i] + (1 - self.alpha) * cal_mean)
```

**What this does**: Adjusts the prototypes using a few labeled samples from the **test subject**.

With `alpha = 0.5`:
```
calibrated_prototype = 0.5 × original_prototype + 0.5 × test_subject_mean
```

**Why?** Every person's brain is different. The prototype computed from training subjects may not perfectly represent what PD "looks like" for a new person. By blending the generic prototype with the new person's data, we adapt to their specific brain characteristics.

### Classification

```python
def classify(self, query_embeddings, prototypes):
    dists = torch.cdist(query_embeddings, prototypes)     # Euclidean distances
    log_probs = F.log_softmax(-dists, dim=1)              # convert to probabilities
    predictions = torch.argmin(dists, dim=1)               # nearest prototype wins
```

**`torch.cdist`**: Computes pairwise Euclidean distances. For each query sample, calculates its distance to the PD prototype and HC prototype.

**`F.log_softmax(-dists)`**: Converts negative distances to log-probabilities. Closer distance → higher probability. The negative sign flips it so smaller distance = bigger value.

**`argmin`**: The class with the smallest distance wins — nearest prototype classification.

---

## 8. `train.py` — Training and Evaluation

**Purpose**: Train MCPNet using episodic learning and evaluate with LOSO.

### Episode Creation

```python
def create_episode(subjects, k_shot, n_query, use_plv=True):
    pd_subjects = [s for s in subjects if s.label == 1]
    hc_subjects = [s for s in subjects if s.label == 0]
    ...
    # Sample K support + N query for each class
    support_psd = concat([hc_psd[:k_shot], pd_psd[:k_shot]])
    query_psd = concat([hc_psd[k_shot:], pd_psd[k_shot:]])
```

Each **episode** is one "mini classification task":
1. Randomly pick epochs from PD subjects and HC subjects
2. Split into **support set** (K per class — used to compute prototypes) and **query set** (N per class — used to compute loss)

This is called **episodic training** — instead of traditional batch training, the model sees many small classification problems, each with a different support set. This teaches it to generalize across different sets of examples.

### Training One Fold

```python
def train_one_fold(model, train_subjects, k_shot=5, ...):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(n_epochs):
        for ep in range(n_episodes):
            # Create episode
            s_psd, s_plv, s_labels, q_psd, q_plv, q_labels = create_episode(...)

            # Forward pass
            log_probs, _ = model(s_psd, s_plv, s_labels, q_psd, q_plv)

            # Compute loss
            loss = F.nll_loss(log_probs, q_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**Line by line:**
- **`Adam`**: Adaptive optimizer that adjusts learning rate per-parameter. Standard choice for deep learning.
- **`StepLR`**: Reduces learning rate by half every 20 epochs. Helps convergence.
- **`F.nll_loss`**: Negative Log Likelihood loss. Combined with `log_softmax`, this is equivalent to Cross-Entropy loss — the standard classification loss.
- **`optimizer.zero_grad()`**: Clears gradients from previous step (PyTorch accumulates them by default).
- **`loss.backward()`**: Computes gradients of loss w.r.t. all model parameters via backpropagation.
- **`optimizer.step()`**: Updates parameters using the computed gradients.

### LOSO Evaluation

```python
def loso_evaluation(subjects, k_shot=5, ...):
    for fold_idx, test_subj in enumerate(subjects):
        # Training set: everyone except test subject
        train_subjs = [s for s in subjects if s.subject_id != test_subj.subject_id]

        # Fresh model for each fold
        model = MCPNet(...).to(DEVICE)

        # Train on 86 subjects
        model, losses = train_one_fold(model, train_subjs, ...)

        # Test on held-out subject
        acc, y_true, y_pred = evaluate_subject(model, test_subj, train_subjs, ...)
```

**Critical point**: A **fresh model** is created for each fold. This means:
- Fold 1: Train new model on subjects 2-87, test on subject 1
- Fold 2: Train new model on subjects 1, 3-87, test on subject 2
- ...
- Fold 87: Train new model on subjects 1-86, test on subject 87

This ensures the test subject has **never been seen** during training — not even indirectly.

### Subject Evaluation

```python
def evaluate_subject(model, test_subject, train_subjects, k_shot=5, ...):
    # Split test subject's epochs
    cal_indices = indices[:k_shot]        # K epochs for calibration
    query_indices = indices[k_shot:]       # rest for testing

    # Build support set from training subjects
    s_psd_hc, s_plv_hc = get_support(hc_train, k_shot)
    s_psd_pd, s_plv_pd = get_support(pd_train, k_shot)

    # Calibrate and classify
    log_probs, predictions = model(
        support_psd, support_plv, support_labels,
        query_psd, query_plv,
        calibration_psd, calibration_plv, calibration_labels
    )
```

For the held-out test subject:
1. Take K of their epochs for **calibration** (prototype adjustment)
2. Use remaining epochs as **queries** (actual test set)
3. Build support set from training subjects (K from PD, K from HC)
4. Model encodes everything → computes prototypes → calibrates → classifies queries
5. Compare predictions to true labels → compute accuracy

### Metrics

```python
overall_acc = accuracy_score(all_y_true, all_y_pred)
overall_f1 = f1_score(all_y_true, all_y_pred, average='binary')
cm = confusion_matrix(all_y_true, all_y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)    # how many PD patients correctly identified
specificity = tn / (tn + fp)    # how many HC correctly identified
```

- **Accuracy**: Overall correct predictions / total predictions
- **Sensitivity (Recall)**: Of all actual PD patients, how many did we catch?
- **Specificity**: Of all actual healthy controls, how many did we correctly clear?
- **F1 Score**: Harmonic mean of precision and recall — balanced metric
- **Confusion Matrix**: TN (true HC), FP (healthy called PD), FN (PD missed), TP (PD caught)

---

## 9. `main.py` — Pipeline Runner

**Purpose**: Glue everything together with a command-line interface.

```python
def run_pipeline(args):
    # Step 1: Load data
    if args.real:
        subjects = load_all_datasets()
    else:
        subjects = generate_synthetic_data(n_subjects=args.n_subjects)

    # Step 2: Preprocess
    subjects = preprocess_all(subjects, skip_ica=args.skip_ica)

    # Step 3: Extract features
    subjects = extract_features_all(subjects)

    # Step 4: LOSO evaluation for each K-shot setting
    for k in k_shot_list:
        results = loso_evaluation(subjects, k_shot=k, ...)

    # Step 5: Save results
    save_results(all_results, 'loso_results.json')
```

The `argparse` setup provides CLI flags to control every aspect of the pipeline without editing code.

---

## 10. How Everything Connects

```
main.py calls:
│
├── dataset.py
│   ├── load_all_datasets() → reads raw EEG files → returns [Subject, Subject, ...]
│   └── generate_synthetic_data() → creates fake EEG → returns [Subject, Subject, ...]
│
├── preprocessing.py
│   └── preprocess_all(subjects) → for each subject:
│       ├── bandpass_filter()
│       ├── notch_filter()
│       ├── run_ica()
│       ├── harmonize_channels()
│       └── segment_epochs() → populates subject.epochs
│
├── features.py
│   └── extract_features_all(subjects) → for each subject:
│       ├── compute_psd_all_epochs() → populates subject.psd_features
│       └── compute_plv_all_epochs() → populates subject.plv_features
│
└── train.py
    └── loso_evaluation(subjects) → for each fold:
        ├── MCPNet() → creates fresh model
        ├── train_one_fold() → episodic training on 86 subjects
        │   └── create_episode() → sample support/query sets
        └── evaluate_subject() → test on held-out subject
            ├── model.encode() → embed features
            ├── model.compute_prototypes() → class centroids
            ├── model.calibrate_prototypes() → adapt to test subject
            └── model.classify() → nearest prototype → PD or HC
```

---

## 11. Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **EEG** | Electroencephalography — recording brain electrical activity from scalp electrodes |
| **Epoch** | A fixed-duration segment of EEG (1 second in our case) |
| **PSD** | Power Spectral Density — how much power at each frequency |
| **PLV** | Phase Locking Value — synchronization between two brain regions |
| **Few-shot learning** | Learning from very few labeled examples (K-shot = K examples per class) |
| **Prototype network** | Classifies by computing class centroids and measuring distance |
| **Support set** | The K labeled examples used to compute prototypes |
| **Query set** | The unlabeled examples to be classified |
| **Episode** | One mini classification task (support + query) used during training |
| **LOSO** | Leave-One-Subject-Out — hold out one entire subject for testing |
| **Prototype calibration** | Adjusting prototypes using test subject's data for personalization |
| **ICA** | Independent Component Analysis — separates mixed signals into sources |
| **Hilbert transform** | Converts real signal to complex analytic signal (gives instantaneous phase) |
| **Welch's method** | Estimates PSD by averaging FFTs of overlapping signal segments |
| **Embedding** | A learned vector representation of input data in a lower-dimensional space |
| **Euclidean distance** | Straight-line distance between two points: sqrt(sum((a-b)^2)) |
| **BatchNorm** | Normalizes layer inputs to stabilize and accelerate training |
| **Adam optimizer** | Adaptive learning rate optimizer — adjusts step size per parameter |
| **NLL loss** | Negative Log Likelihood — standard classification loss function |
| **Conv2d** | 2D convolution — slides a kernel across input to detect spatial patterns |
| **AdaptiveAvgPool** | Pools any spatial size to a fixed size by averaging |
