# Phase 1 – Paper Understanding and Technical Breakdown

## MCPNet: A Novel EEG-Based Parkinson's Disease Detection Model Using Multiscale Convolutional Prototype Networks (Qiu et al., 2024)

---

# SECTION 1 – Problem Understanding

### 1. What is the main research problem addressed in the paper?

The paper addresses the challenge of detecting Parkinson's Disease (PD) from EEG signals in a way that generalizes well to **unseen subjects**. Traditional deep learning models for EEG-based PD detection require large amounts of labeled training data and tend to overfit to subject-specific patterns. The authors propose a **few-shot learning** approach using prototype networks so that the model can classify new subjects using only a small number of labeled EEG samples (support set), making it more practical for real-world clinical deployment where collecting large labeled datasets per patient is expensive and time-consuming.

### 2. Why is EEG-based Parkinson's disease detection challenging?

- **High inter-subject variability**: EEG signals vary significantly across individuals due to differences in brain anatomy, electrode placement, and neural activity patterns. A model trained on one group of subjects may fail on new ones.
- **Small dataset sizes**: Most publicly available PD-EEG datasets contain only 20–30 subjects, which is far too small for robust deep learning.
- **Non-stationarity of EEG**: EEG signals are noisy, non-stationary, and contain artifacts (eye blinks, muscle movements, electrical interference), making feature extraction difficult.
- **Heterogeneity across datasets**: Different research groups record EEG with different equipment, sampling rates, channel configurations, and protocols, making cross-dataset generalization hard.
- **Class imbalance**: Some datasets have unequal numbers of PD and healthy control (HC) subjects.

### 3. What are the major gaps in previous research that the authors identify?

- Most prior studies use **k-fold cross-validation**, which can mix segments from the same subject into both training and test sets. This causes **data leakage** — the model memorizes subject-specific patterns rather than learning disease-relevant features, leading to inflated accuracy.
- Prior methods often evaluate on a **single dataset**, making it unclear whether results generalize across different recording conditions.
- Many existing models require **large labeled training sets** and cannot adapt to new subjects with limited data.
- Few studies have applied **few-shot learning** or **prototype-based methods** to EEG-based PD detection.
- Limited work on **cross-dataset evaluation**, where models are trained on one dataset and tested on a completely different one.

### 4. Why do the authors argue that k-fold cross-validation may produce inflated results in EEG studies?

In k-fold CV, data is randomly split into folds. Since each EEG recording is divided into many 1-second epochs (e.g., 180 epochs per subject), random splitting will place some epochs from the **same subject** in the training set and others in the test set. Because epochs from the same subject share subject-specific characteristics (skull thickness, baseline neural activity, electrode impedance), the model can effectively "recognize" the subject rather than learn disease-related patterns. This is a form of **data leakage** that leads to artificially high accuracy — often above 99% — that does not reflect real-world performance on truly unseen patients.

### 5. Why is LOSO (Leave-One-Subject-Out) validation considered more reliable?

In LOSO, **all epochs from one subject** are held out as the test set, and the model is trained on all remaining subjects. This is repeated for every subject. This ensures there is **zero overlap** between training and test subjects — the model must generalize to a person it has never seen. LOSO simulates the real clinical scenario where a trained model encounters a brand-new patient. While LOSO typically produces lower accuracy numbers than k-fold CV, those numbers are far more trustworthy and clinically meaningful.

---

# SECTION 2 – Dataset Understanding

### Dataset Summary Table

| Dataset | PD Subjects | HC Subjects | Sampling Rate | Recording Duration | Number of Channels |
|---------|-------------|-------------|---------------|--------------------|--------------------|
| UC San Diego | 15 | 16 | 512 Hz | 3 minutes | 32 (after harmonization) |
| UNM (University of New Mexico) | 14 | 14 | 500 Hz | 2 minutes | 32 (after harmonization) |
| Iowa | 14 | 14 | 500 Hz | 2 minutes | 32 (after harmonization) |

### Questions

### 1. What are the three datasets used in the paper?

- **UC San Diego dataset**: Collected at the University of California, San Diego.
- **UNM dataset**: Collected at the University of New Mexico.
- **Iowa dataset**: Collected at the University of Iowa.

All three are resting-state EEG recordings from PD patients and age-matched healthy controls (HC).

### 2. What are the number of PD and healthy control subjects in each dataset?

- UC San Diego: 15 PD, 16 HC (total 31)
- UNM: 14 PD, 14 HC (total 28)
- Iowa: 14 PD, 14 HC (total 28)

Combined total: 43 PD and 44 HC = 87 subjects.

### 3. What are the sampling rates and recording durations?

- UC San Diego: 512 Hz sampling rate, approximately 3 minutes of recording per subject.
- UNM and Iowa: 500 Hz sampling rate, approximately 2 minutes of recording per subject.

### 4. What differences exist between the datasets?

- **Sampling rate**: UC uses 512 Hz while UNM and Iowa use 500 Hz.
- **Recording duration**: UC has 3-minute recordings (yielding ~180 one-second epochs) while UNM and Iowa have 2-minute recordings (~120 epochs each).
- **Channel configuration**: The three datasets were recorded using slightly different electrode montages. For example, the Iowa dataset includes a Pz channel instead of Fz.
- **Recording equipment and conditions**: Different labs, different EEG systems, potentially different impedance levels and environmental noise.
- **Subject demographics**: Different subject pools from different geographic locations.

### 5. Why is channel harmonization necessary when combining datasets?

Since the three datasets were recorded using different EEG systems and electrode montages, the channel names and positions may not align perfectly. To combine or compare data across datasets, a common set of channels must be established. Channel harmonization selects 32 channels that are common (or can be mapped) across all three datasets. Without this step, the feature dimensions would be inconsistent and the model couldn't process data from different datasets uniformly. This is critical for cross-dataset evaluation.

---

# SECTION 3 – EEG Preprocessing Pipeline

### Questions

### 1. What preprocessing operations are applied to the raw EEG signals?

The following preprocessing steps are applied sequentially:
1. **Band-pass filtering** (0.5–50 Hz) to retain relevant neural frequencies
2. **Notch filtering** (50 Hz or 60 Hz) to remove power line interference
3. **ICA-based artifact removal** to eliminate eye blinks, muscle artifacts, and other non-neural signals
4. **Channel harmonization** to standardize electrode montage across datasets
5. **Epoch segmentation** to divide continuous recordings into 1-second non-overlapping segments

### 2. Why is band-pass filtering (0.5–50 Hz) applied?

- The **low cutoff (0.5 Hz)** removes slow DC drifts and baseline wander caused by electrode movement, skin conductance changes, and breathing.
- The **high cutoff (50 Hz)** removes high-frequency noise (muscle artifacts, electrical interference) that is not related to the brain rhythms of interest.
- The relevant EEG frequency bands for PD analysis fall within this range: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–50 Hz). Frequencies above 50 Hz are mostly noise for this application.

### 3. Why is notch filtering necessary?

Electrical power lines operate at 50 Hz (in most of Asia, Europe, and Oceania) or 60 Hz (in North America). This creates a persistent narrowband interference in EEG recordings that shows up as a sharp peak at the power line frequency. A notch filter removes this specific frequency (and its harmonics) without disturbing neighboring frequencies. Without it, this artifact would contaminate the gamma band features and distort PSD estimates.

### 4. Why is ICA (Independent Component Analysis) used in EEG preprocessing?

ICA is a blind source separation technique that decomposes the multi-channel EEG signal into statistically independent components. Each component typically corresponds to a distinct source — either a brain source or an artifact source. By identifying and removing artifact components, we can clean the EEG while preserving the underlying neural activity. ICA is preferred over simple threshold-based rejection because it can separate overlapping sources without discarding entire data segments.

### 5. What artifacts are removed using ICA?

- **Ocular artifacts**: Eye blinks and saccades (eye movements) — these produce large, distinctive patterns in frontal channels.
- **Muscle artifacts (EMG)**: High-frequency contamination from facial and scalp muscle activity.
- **Cardiac artifacts (ECG)**: Heartbeat-related electrical activity that can propagate to EEG electrodes.
- **Line noise residuals**: Any remaining power line artifacts not fully removed by notch filtering.
- **Movement artifacts**: Electrode shifts or cable movements during recording.

### 6. What is channel harmonization and why is it required?

Channel harmonization is the process of selecting and standardizing a common set of EEG channels across all datasets. Since different datasets may use different EEG caps with slightly different electrode positions or naming conventions, harmonization ensures that the same 32 channels (corresponding to the same approximate scalp locations) are used for all subjects. This allows the model to receive consistent input dimensions and ensures that spatial features (like PLV connectivity between specific channel pairs) are comparable across datasets.

### 7. Why was the Pz channel replaced by Fz in the Iowa dataset?

The Iowa dataset's EEG montage included the Pz electrode but was missing the Fz electrode, which was present in the other two datasets. To maintain a consistent set of 32 channels across all datasets, the Pz channel in the Iowa data was mapped/replaced by Fz. This substitution was made based on the need for uniform channel configuration. While Pz (parietal midline) and Fz (frontal midline) are at different scalp locations, this trade-off was necessary to enable cross-dataset compatibility.

### Preprocessing Pipeline Flowchart

```
┌─────────────────────┐
│     Raw EEG Data     │
│  (Multi-channel,     │
│   continuous signal) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Band-pass Filtering │
│    (0.5 – 50 Hz)    │
│  Removes DC drift &  │
│  high-freq noise     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Notch Filtering    │
│   (50 Hz / 60 Hz)   │
│  Removes power line  │
│  interference        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ICA Artifact        │
│  Removal             │
│  Removes: eye blinks,│
│  muscle, cardiac     │
│  artifacts           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Channel             │
│  Harmonization       │
│  Standardize to 32   │
│  common channels     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Epoch Segmentation  │
│  1-second non-       │
│  overlapping windows │
└─────────────────────┘
```

---

# SECTION 4 – Epoch Segmentation and Data Organization

### Questions

### 1. Why do the authors divide EEG recordings into 1-second segments?

- **Increases sample size**: A single 3-minute EEG recording yields ~180 one-second epochs, turning a small dataset of ~30 subjects into thousands of samples.
- **Stationarity assumption**: EEG signals are non-stationary over long durations, but can be considered approximately stationary within short (1-second) windows. This makes spectral analysis (PSD) more valid.
- **Compatibility with few-shot learning**: The episodic training setup requires many samples per class to form support and query sets. Short segments provide enough samples.
- **Standard practice**: 1-second windows are commonly used in EEG analysis as they balance temporal resolution with frequency resolution.

### 2. For the UC dataset, explain the tensor structure: 32 × 512 × 180

| Dimension | Value | Meaning |
|-----------|-------|---------|
| Channels  | 32    | Number of EEG electrodes (after harmonization) |
| Time points | 512 | Number of samples per second per channel (sampling rate = 512 Hz, window = 1 second) |
| Epochs    | 180   | Number of 1-second segments from a ~3-minute recording (180 seconds ÷ 1 second = 180 epochs) |

So for each UC subject, the data tensor has shape **32 channels × 512 time points × 180 epochs**.

### 3. For the UNM and Iowa datasets, explain the tensor structure: 32 × 500 × 120

| Dimension | Value | Meaning |
|-----------|-------|---------|
| Channels  | 32    | Number of EEG electrodes (after harmonization) |
| Time points | 500 | Number of samples per second per channel (sampling rate = 500 Hz, window = 1 second) |
| Epochs    | 120   | Number of 1-second segments from a ~2-minute recording (120 seconds ÷ 1 second = 120 epochs) |

So for each UNM/Iowa subject, the data tensor has shape **32 channels × 500 time points × 120 epochs**.

### 4. How many samples per subject are generated after segmentation?

- **UC dataset**: ~180 epochs per subject (3 min × 60 sec/min = 180 one-second epochs)
- **UNM dataset**: ~120 epochs per subject (2 min × 60 sec/min = 120 one-second epochs)
- **Iowa dataset**: ~120 epochs per subject (2 min × 60 sec/min = 120 one-second epochs)

Total samples across all subjects:
- UC: 31 subjects × 180 = 5,580 epochs
- UNM: 28 subjects × 120 = 3,360 epochs
- Iowa: 28 subjects × 120 = 3,360 epochs
- **Grand total: ~12,300 epochs**

### 5. Are these samples independent samples? Explain.

**No, they are not truly independent.** Epochs from the same subject are temporally adjacent segments from the same continuous EEG recording. They share:

- **Subject-specific characteristics**: The same brain anatomy, skull thickness, medication state, and electrode impedances.
- **Temporal correlation**: Adjacent 1-second windows are likely to have similar neural states (the brain doesn't drastically change between one second and the next).
- **Recording conditions**: Same session, same equipment calibration, same environmental noise.

This is precisely why **k-fold cross-validation is problematic** — randomly splitting epochs would place correlated epochs from the same subject in both train and test sets, creating data leakage. **LOSO validation addresses this** by ensuring all epochs from a given subject are either entirely in the training set or entirely in the test set.

### Tensor Structure Diagram

```
UC San Diego Dataset (per subject):
┌─────────────────────────────────────────────────┐
│                                                 │
│   Epoch 1        Epoch 2        ...  Epoch 180  │
│  ┌─────────┐   ┌─────────┐       ┌─────────┐   │
│  │ Ch1: ~~~│   │ Ch1: ~~~│       │ Ch1: ~~~│   │
│  │ Ch2: ~~~│   │ Ch2: ~~~│       │ Ch2: ~~~│   │
│  │ ...     │   │ ...     │       │ ...     │   │
│  │Ch32: ~~~│   │Ch32: ~~~│       │Ch32: ~~~│   │
│  └─────────┘   └─────────┘       └─────────┘   │
│   512 points    512 points        512 points    │
│                                                 │
│         Tensor shape: 32 × 512 × 180            │
└─────────────────────────────────────────────────┘

UNM / Iowa Dataset (per subject):
┌─────────────────────────────────────────────────┐
│                                                 │
│   Epoch 1        Epoch 2        ...  Epoch 120  │
│  ┌─────────┐   ┌─────────┐       ┌─────────┐   │
│  │ Ch1: ~~~│   │ Ch1: ~~~│       │ Ch1: ~~~│   │
│  │ Ch2: ~~~│   │ Ch2: ~~~│       │ Ch2: ~~~│   │
│  │ ...     │   │ ...     │       │ ...     │   │
│  │Ch32: ~~~│   │Ch32: ~~~│       │Ch32: ~~~│   │
│  └─────────┘   └─────────┘       └─────────┘   │
│   500 points    500 points        500 points    │
│                                                 │
│         Tensor shape: 32 × 500 × 120            │
└─────────────────────────────────────────────────┘

~~~ = time-series voltage values for 1-second window
```

---

# SECTION 5 – Feature Extraction (PSD and PLV)

### Questions

### 1. What is Power Spectral Density (PSD)?

PSD describes how the **power (energy) of a signal is distributed across different frequencies**. For EEG, it quantifies how much neural activity is present in each frequency band (delta, theta, alpha, beta, gamma). Mathematically, PSD is typically computed using Welch's method, which divides the signal into overlapping segments, computes the FFT of each, and averages the squared magnitudes. The result is a power value (in µV²/Hz) for each frequency bin, for each channel.

### 2. Which frequency bands are used in the PSD computation?

| Band | Frequency Range | Associated Brain Activity |
|------|----------------|--------------------------|
| Delta (δ) | 0.5 – 4 Hz | Deep sleep, unconscious processes |
| Theta (θ) | 4 – 8 Hz | Drowsiness, light sleep, memory |
| Alpha (α) | 8 – 13 Hz | Relaxed wakefulness, eyes closed |
| Beta (β) | 13 – 30 Hz | Active thinking, focus, motor planning |
| Gamma (γ) | 30 – 50 Hz | Higher cognitive functions, perception |

The PSD is computed for each of these 5 bands across all 32 channels, resulting in a **32 × 5 feature matrix** per epoch (32 channels × 5 frequency bands).

### 3. Why is PSD useful for Parkinson's detection?

PD causes characteristic changes in brain oscillatory activity:
- **Increased theta power**: Slowing of brain rhythms is a hallmark of PD, reflecting cortical dysfunction.
- **Decreased beta power**: PD is associated with abnormal beta oscillations in motor circuits, which are linked to the bradykinesia and rigidity symptoms.
- **Altered alpha rhythms**: Changes in alpha band power reflect disrupted resting-state cortical activity.
- These spectral biomarkers provide a **compact, informative representation** of neural activity that captures disease-related changes better than raw EEG voltage values.

### 4. What is Phase Locking Value (PLV)?

PLV measures the **consistency of the phase difference between two EEG channels across time** (or across trials). It quantifies **functional connectivity** — whether two brain regions are synchronizing their oscillatory activity. PLV ranges from 0 to 1:
- **PLV = 1**: The phase difference between two channels is perfectly constant (complete synchronization).
- **PLV = 0**: The phase difference is completely random (no synchronization).

Mathematically, PLV is computed as:
```
PLV = |1/N × Σ exp(j × (φ_a(t) - φ_b(t)))|
```
where φ_a(t) and φ_b(t) are the instantaneous phases of channels a and b at time t, extracted using the Hilbert transform.

### 5. What type of brain information does PLV capture?

PLV captures **functional connectivity** — the degree to which different brain regions communicate or synchronize their activity. Unlike PSD, which measures local activity at each electrode, PLV measures **pairwise relationships between channels**. This provides information about:
- Inter-regional neural communication
- Network-level brain organization
- Synchronization patterns across brain regions

For 32 channels computed across 5 frequency bands, PLV produces a **32 × 32 × 5 connectivity matrix** per epoch (though only the upper triangle is needed since PLV is symmetric, and the diagonal is always 1).

### 6. Why do the authors combine PSD and PLV features instead of using only one?

PSD and PLV capture **complementary** aspects of brain activity:
- **PSD** captures **local spectral power** — how active each brain region is in each frequency band. It's a channel-level (node-level) feature.
- **PLV** captures **inter-regional connectivity** — how synchronized pairs of brain regions are. It's an edge-level (network-level) feature.

PD affects both local neural activity (e.g., slowing of rhythms) and network connectivity (e.g., disrupted communication between motor cortex regions). By combining both features, the model gets a richer, more complete picture of the disease's neural signature, leading to better classification performance than either feature alone.

### Feature Comparison Table

| Feature | What It Measures | Mathematical Idea | Why Useful for PD |
|---------|-----------------|-------------------|-------------------|
| PSD (Power Spectral Density) | Power distribution of EEG signal across frequency bands at each channel | Decompose signal into frequency components using FFT/Welch's method; compute average power per band | PD causes characteristic spectral changes: increased theta (cortical slowing), altered beta (motor dysfunction), changed alpha (resting-state disruption) |
| PLV (Phase Locking Value) | Phase synchronization consistency between pairs of EEG channels | Compute instantaneous phase via Hilbert transform; measure consistency of phase difference across time | PD disrupts functional brain networks; abnormal connectivity patterns between motor, frontal, and other regions serve as disease biomarkers |

### Feature Dimensions Summary

```
Per 1-second epoch:

PSD Feature:
┌──────────────────────┐
│  32 channels × 5 bands│  → Shape: 32 × 5 = 160 features
│  (δ, θ, α, β, γ)     │
└──────────────────────┘

PLV Feature:
┌──────────────────────┐
│  32 × 32 channels     │
│  × 5 bands            │  → Shape: 32 × 32 × 5 connectivity matrix
│  (symmetric matrix)   │
└──────────────────────┘

Combined → Fed into MCPNet encoder
```

---

# SECTION 6 – Model Architecture (MCPNet)

### Questions

### 1. What does MCPNet stand for?

**Multiscale Convolutional Prototype Network.** It combines multiscale convolutional feature extraction with prototype-based few-shot classification.

### 2. What are the three main components of MCPNet?

1. **Multiscale CNN Encoder**: Extracts features at multiple temporal/spatial scales from the input PSD and PLV features using parallel convolutional branches with different kernel sizes.
2. **Prototype Computation Module**: Computes class prototypes (representative embeddings) for each class (PD and HC) by averaging the encoded feature vectors of the support set samples.
3. **Prototype Calibration Strategy**: Refines the prototypes using a small number of labeled samples from the test subject to adapt the model to the specific characteristics of the new subject.

### 3. What is the purpose of the multiscale CNN encoder?

The multiscale CNN encoder processes the input features using **multiple parallel convolutional branches**, each with a different kernel size. This allows the network to capture patterns at different scales simultaneously:
- **Small kernels** capture fine-grained, local patterns (e.g., sharp spectral peaks, narrow connectivity features).
- **Large kernels** capture broader, more global patterns (e.g., overall spectral trends, wide-range connectivity patterns).

The outputs from all branches are concatenated to form a rich, multi-resolution feature representation. This is important because PD-related EEG changes manifest at multiple scales.

### 4. Why are different convolution kernel sizes used?

Different kernel sizes act as **multi-resolution filters**:
- **Small kernels (e.g., 3×3)**: Detect localized, fine-grained features — specific channel interactions, narrow frequency band changes.
- **Medium kernels (e.g., 5×5)**: Capture intermediate-scale patterns — regional connectivity, moderate spectral trends.
- **Large kernels (e.g., 7×7)**: Detect global, broad patterns — whole-brain connectivity changes, overall spectral shifts.

Since EEG biomarkers for PD exist at multiple spatial and spectral scales, using a single kernel size would miss patterns at other resolutions. The multiscale approach ensures the model captures the full range of disease-related features.

### 5. What is prototype learning?

Prototype learning is a **metric-based few-shot learning** approach. Instead of learning a direct mapping from input to class label (like traditional classifiers), the model:
1. Learns an **embedding function** (the encoder) that maps inputs to a feature space.
2. Computes a **prototype** (centroid) for each class by averaging the embeddings of a few labeled examples (the support set).
3. Classifies a new sample by measuring its **distance to each prototype** in the embedding space — the sample is assigned to the class whose prototype is nearest.

The key insight is that the model doesn't need to retrain for new subjects; it just needs a few labeled examples to compute prototypes and then classifies based on proximity.

### 6. How are class prototypes calculated?

For each class c (PD or HC), the prototype is the **mean embedding vector** of all support set samples belonging to that class:

```
Prototype_c = (1/|S_c|) × Σ f_θ(x_i)   for all x_i in S_c
```

Where:
- S_c = set of support samples from class c
- f_θ = the multiscale CNN encoder (parameterized by θ)
- x_i = an input sample (PSD + PLV features)
- |S_c| = number of support samples in class c

For a 2-way (PD vs HC) K-shot setting, each class has K support samples, so each prototype is the average of K encoded feature vectors.

### MCPNet Architecture Diagram

```
                         Input Features
                    (PSD: 32×5, PLV: 32×32×5)
                              │
                              ▼
              ┌───────────────────────────────┐
              │     MULTISCALE CNN ENCODER     │
              │                               │
              │  ┌─────────┐ ┌─────────┐ ┌─────────┐
              │  │ Branch 1│ │ Branch 2│ │ Branch 3│
              │  │ Small   │ │ Medium  │ │ Large   │
              │  │ Kernel  │ │ Kernel  │ │ Kernel  │
              │  │ (3×3)   │ │ (5×5)   │ │ (7×7)   │
              │  └────┬────┘ └────┬────┘ └────┬────┘
              │       │          │          │
              │       └──────────┼──────────┘
              │                  │
              │          Concatenate
              │                  │
              │                  ▼
              │         Feature Embedding
              │           (vector z)
              └───────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
          Support Set                   Query Set
          Samples                       Samples
                │                            │
                ▼                            │
   ┌────────────────────────┐                │
   │  PROTOTYPE COMPUTATION │                │
   │                        │                │
   │  PD Prototype =        │                │
   │    mean(PD support     │                │
   │    embeddings)         │                │
   │                        │                │
   │  HC Prototype =        │                │
   │    mean(HC support     │                │
   │    embeddings)         │                │
   └───────────┬────────────┘                │
               │                             │
               ▼                             ▼
   ┌─────────────────────────────────────────────┐
   │         PROTOTYPE CALIBRATION               │
   │  Refine prototypes using a few labeled      │
   │  samples from the test subject              │
   └──────────────────┬──────────────────────────┘
                      │
                      ▼
   ┌─────────────────────────────────────────────┐
   │         DISTANCE COMPUTATION                │
   │  d(query, PD_prototype) vs                  │
   │  d(query, HC_prototype)                     │
   │                                             │
   │  Classify query → nearest prototype         │
   └─────────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Prediction:  │
              │  PD or HC     │
              └───────────────┘
```

---

# SECTION 7 – Experimental Design

### Questions

### 1. What is Leave-One-Subject-Out (LOSO) validation?

LOSO is a validation strategy where, in each fold, **one subject's data is entirely held out as the test set**, and the model is trained on all remaining subjects. This process is repeated N times (once per subject), and results are averaged. For 87 subjects across three datasets, this means 87 iterations of training and testing.

```
Fold 1: Train on Subjects 2–87, Test on Subject 1
Fold 2: Train on Subjects 1, 3–87, Test on Subject 2
...
Fold 87: Train on Subjects 1–86, Test on Subject 87
```

### 2. Why is LOSO more suitable for EEG datasets?

- **Prevents data leakage**: No epochs from the test subject appear in training.
- **Simulates clinical reality**: In practice, a deployed system would encounter entirely new patients. LOSO mimics this scenario.
- **Accounts for inter-subject variability**: Forces the model to learn generalizable disease features rather than subject-specific patterns.
- **More honest evaluation**: Provides realistic performance estimates, unlike k-fold CV which can give misleadingly high accuracy.

### 3. What is N-way K-shot learning?

N-way K-shot is the standard few-shot learning setup:
- **N-way**: The number of classes to distinguish. In this paper, N = 2 (PD vs HC).
- **K-shot**: The number of labeled examples provided per class in the support set. The paper experiments with K = 1, 5, 10, 20.

So a "2-way 5-shot" task means: given 5 labeled PD examples and 5 labeled HC examples (the support set), classify new unlabeled examples (the query set).

### 4. What is a support set?

The support set is the **small set of labeled examples** provided to the model at test time for each classification episode. It serves as the "reference" from which class prototypes are computed. In a 2-way K-shot setting:
- K samples labeled as PD
- K samples labeled as HC

The model uses these to compute the PD prototype and HC prototype in the embedding space. The support set can be thought of as a "few-shot training set" that the model uses to adapt to the current classification task.

### 5. What is a query set?

The query set contains the **unlabeled samples that need to be classified**. After the model computes prototypes from the support set, each query sample is embedded by the encoder and classified based on its distance to the prototypes. The query set is essentially the "test set" within each episode.

In practice for LOSO: when subject S is held out, a few of subject S's epochs form the support set, and the remaining epochs form the query set.

### 6. How does the prototype calibration strategy work?

Prototype calibration is a refinement step that adapts the generic prototypes to the specific test subject:

1. **Initial prototypes** are computed from the training subjects' support samples.
2. A small number of labeled samples from the **test subject** (calibration samples) are embedded and used to **shift/adjust** the prototypes.
3. The calibrated prototype is a weighted combination of the original prototype and the mean embedding of the calibration samples:

```
Calibrated_Prototype_c = α × Original_Prototype_c + (1 - α) × mean(calibration_embeddings_c)
```

This allows the model to adapt to the specific EEG characteristics of the new subject (their unique brain patterns, recording conditions, etc.) without full retraining. It bridges the gap between the general knowledge from training subjects and the specific characteristics of the test subject.

### Experimental Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATASET PREPARATION                       │
│  UC (31 subjects) + UNM (28 subjects) + Iowa (28 subjects)  │
│                    = 87 total subjects                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 LOSO CROSS-VALIDATION                        │
│                                                             │
│  For each subject S (1 to 87):                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ TRAINING SET: All subjects except S                  │   │
│  │                                                      │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │  Episodic Training                          │     │   │
│  │  │                                             │     │   │
│  │  │  For each episode:                          │     │   │
│  │  │   1. Sample support set (K per class)       │     │   │
│  │  │   2. Sample query set                       │     │   │
│  │  │   3. Encode all via Multiscale CNN          │     │   │
│  │  │   4. Compute prototypes from support        │     │   │
│  │  │   5. Classify queries by nearest prototype  │     │   │
│  │  │   6. Compute loss & update encoder          │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ TEST SET: Subject S only                             │   │
│  │                                                      │   │
│  │  1. Take K epochs as support (for calibration)       │   │
│  │  2. Remaining epochs = query set                     │   │
│  │  3. Compute initial prototypes from training support │   │
│  │  4. Calibrate prototypes using subject S's support   │   │
│  │  5. Classify all query epochs                        │   │
│  │  6. Record accuracy for subject S                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              AGGREGATE RESULTS                               │
│  Average accuracy across all 87 LOSO folds                  │
│  Report: Accuracy, Sensitivity, Specificity, F1-score       │
└─────────────────────────────────────────────────────────────┘
```

---

# SECTION 8 – Critical Analysis of the Paper

### Questions

### 1. What are the main strengths of the paper?

- **Rigorous evaluation protocol**: Uses LOSO validation instead of k-fold CV, providing realistic and trustworthy performance estimates that are clinically meaningful.
- **Cross-dataset evaluation**: Tests the model across three independently collected datasets, demonstrating robustness to different recording conditions, equipment, and subject populations.
- **Complementary feature design**: Combines PSD (local spectral power) and PLV (inter-regional connectivity), capturing both node-level and network-level disease signatures.
- **Few-shot learning framework**: Addresses the practical reality of small EEG datasets by using prototype networks that can classify new subjects with only a handful of labeled samples.
- **Prototype calibration**: Introduces an adaptation mechanism that personalizes the model to each test subject, accounting for inter-subject variability.
- **Multiscale feature extraction**: The multi-kernel CNN encoder captures disease patterns at multiple resolutions, improving feature richness.

### 2. What are the limitations of the proposed method?

- **Small total sample size**: Only 87 subjects across all three datasets. While LOSO is rigorous, conclusions from such a small cohort may not generalize to the broader PD population.
- **Resting-state only**: The datasets contain only resting-state EEG. PD symptoms are primarily motor, and task-based EEG (during movement) might reveal additional biomarkers.
- **Channel substitution (Pz → Fz)**: Replacing a parietal channel with a frontal one in the Iowa dataset introduces spatial inconsistency that may affect PLV features.
- **Binary classification only**: The model distinguishes PD vs. HC but doesn't address severity staging, medication effects, or differential diagnosis (e.g., PD vs. other neurodegenerative diseases).
- **Epoch non-independence**: The 1-second epochs from the same subject are not truly independent, which may still introduce some bias in the few-shot episode sampling during training.
- **Limited interpretability**: While prototype networks are more interpretable than black-box classifiers, the paper doesn't provide detailed analysis of which EEG features or brain regions drive the classification decisions.

### 3. Is the prototype calibration strategy realistic for real-world deployment?

**Partially.** The calibration strategy requires a small number of **labeled** samples from the test subject at inference time. In a clinical setting, this means:

- **Pros**: Only a few labeled epochs are needed (not full retraining). If a clinician can label even 5–10 seconds of a patient's EEG, calibration becomes feasible.
- **Cons**: It still requires ground truth labels at test time — but in practice, we don't yet know whether a new patient has PD (that's what we're trying to determine). This creates a **chicken-and-egg problem**. The calibration samples could come from a preliminary clinical assessment, but this somewhat defeats the purpose of automated detection.
- **Possible workaround**: Use unlabeled calibration via transductive or semi-supervised methods, or use the initial uncalibrated prediction to bootstrap calibration iteratively.

### 4. What improvements could be proposed?

- **Larger and more diverse datasets**: Include datasets from multiple countries, age groups, PD stages, and medication states.
- **Task-based EEG**: Incorporate motor task EEG alongside resting-state to capture movement-related biomarkers.
- **Multi-class classification**: Extend to PD severity levels (Hoehn & Yahr stages) or differential diagnosis.
- **Explainability analysis**: Add attention maps or saliency methods to identify which channels, frequency bands, and connectivity patterns are most discriminative.
- **Unsupervised calibration**: Replace the labeled calibration step with domain adaptation or self-supervised methods that don't require test-time labels.
- **Temporal modeling**: Add recurrent or transformer-based components to capture temporal dynamics across epochs, rather than treating each epoch independently.

### 5 Bullet Points of Critical Analysis

1. **Evaluation rigor is the paper's strongest contribution** — by using LOSO instead of k-fold CV and testing across three datasets, the results are among the most trustworthy in the PD-EEG detection literature, even though accuracy numbers are lower than inflated k-fold results reported by others.

2. **The prototype calibration assumption is clinically problematic** — requiring labeled test samples assumes partial knowledge of the diagnosis, which conflicts with the goal of automated screening. Future work should explore label-free adaptation strategies.

3. **Dataset size remains a fundamental bottleneck** — with only 87 subjects, the statistical power to detect subtle effects is limited. The few-shot framework mitigates this computationally but cannot fully compensate for the underlying data scarcity.

4. **Feature engineering (PSD + PLV) vs. end-to-end learning** — while the handcrafted features are well-motivated, they impose assumptions about which signal properties matter. An end-to-end approach learning directly from raw EEG could potentially discover novel biomarkers, though it would need more data.

5. **Reproducibility and clinical translation gap** — the paper would benefit from released code, pre-trained models, and a clear pathway to clinical validation (prospective studies, regulatory considerations, real-time processing requirements).

---

# Summary of All Deliverables Checklist

| Section | Deliverable | Status |
|---------|------------|--------|
| 1 | Problem understanding answers | ✅ Complete |
| 2 | Dataset summary table | ✅ Complete |
| 2 | Brief explanation of datasets | ✅ Complete |
| 3 | Preprocessing pipeline flowchart | ✅ Complete |
| 3 | Preprocessing answers | ✅ Complete |
| 4 | Tensor structure diagram | ✅ Complete |
| 4 | Epoch segmentation answers | ✅ Complete |
| 5 | PSD vs PLV comparison table | ✅ Complete |
| 5 | Feature extraction answers | ✅ Complete |
| 6 | Architecture diagram | ✅ Complete |
| 6 | Architecture answers | ✅ Complete |
| 7 | Experimental pipeline diagram | ✅ Complete |
| 7 | Experimental design answers | ✅ Complete |
| 8 | 5 bullet points of critical analysis | ✅ Complete |
| 8 | Critical analysis answers | ✅ Complete |
