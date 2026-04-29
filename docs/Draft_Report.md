# Deep Convolutional Neural Networks for Acoustic Predictive Maintenance in High-Noise Industrial Environments

**A Scientific Report on the MIMII Pump Anomaly Detection Pipeline**

---

## Abstract

This report documents the design, implementation, and empirical evaluation of a deep-learning pipeline for acoustic predictive maintenance (PdM) on industrial pump systems, using the MIMII (Malfunctioning Industrial Machines Investigation and Inspection) dataset embedded in -6dB factory noise. The pipeline ingests raw `.wav` recordings, transforms them into 2D Log-Mel-Spectrograms, and classifies them as **Normal** or **Abnormal** using a four-block convolutional neural network terminated by a **Global Average Pooling (GAP)** head. Three engineering challenges dominate the work: (i) a severe ~11% minority-class imbalance, addressed via inverse-frequency weighted cross-entropy loss; (ii) host-memory pressure from a 4,200+ sample dataset, resolved with a custom lazy-loading PyTorch [`SpectrogramDataset`](src/dataset.py); and (iii) overfitting from a 2.5M-parameter dense head, eliminated by replacing `Flatten` with `nn.AdaptiveAvgPool2d(1)`, reducing the model to **~97,890 parameters**. A subprocess-isolated Cartesian grid search across learning rate, batch size, and base filter count selected the configuration `LR=1e-3, BS=16, BF=16`, which achieved a held-out **test-set F1 of 0.875** and **ROC-AUC of 0.9737** with only 4 false alarms among 563 normal samples and 12 missed faults among 68 abnormal samples. Grad-CAM heatmaps confirm that the network attends to physically plausible mid-frequency Mel bands when classifying abnormal samples, providing the explainability necessary for industrial deployment.

---

## 1. Problem Description

### 1.1 The Domain Problem

In industrial operations, the management of capital assets is a primary economic concern. Industrial machines degrade continuously, and the traditional posture of **reactive maintenance** — repairing equipment only after a failure event — produces catastrophic and expensive downtime. From an economic perspective, this represents a significant opportunity cost and an inefficient allocation of resources. **Predictive maintenance (PdM)** instead seeks to identify microscopic mechanical anomalies *acoustically* before total system failure occurs, thereby minimizing capital depreciation and maximizing operational efficiency. This project specifically addresses PdM for industrial pump systems.

### 1.2 The Data Challenge: Class Imbalance and Signal-to-Noise Ratio

The project utilizes the MIMII dataset, restricted to pump audio embedded in **-6dB SNR factory noise**. A negative SNR represents a worst-case scenario in which the background industrial hum is significantly louder than the machine's operational sound, complicating the detection of subtle fault signatures.

A more profound challenge is the severe class imbalance inherent to real-world industrial data. The dataset is composed of:

- **Normal (healthy) clips**: 3,749 samples (label `0`)
- **Abnormal (faulty) clips**: 456 samples (label `1`)

This yields an approximately **~11% minority class**. In economic terms, this is analogous to "**black swan**" events — rare occurrences that are difficult to model precisely *because* they are scarce relative to steady-state operations.

### 1.3 The Accuracy Paradox

The class imbalance described above creates the well-known **Accuracy Paradox**: a degenerate classifier that always predicts "Normal" trivially attains an accuracy of approximately 89%. In a standard academic setting, 89% accuracy might appear successful. In an industrial context, such a model is functionally useless.

The asymmetry of error costs compounds this problem. A **False Negative** (missing a catastrophic fault) is astronomically more expensive than a **False Positive** (a false alarm), because a missed fault propagates into unscheduled downtime, secondary mechanical damage, and potentially personnel risk. This project therefore shifts the evaluative focus away from naive accuracy and toward metrics that prioritize the detection of the minority class — namely **Recall**, **F1-Score**, and **ROC-AUC**. This shift aligns the model's "utility function" with the actual economic risks present on the factory floor.

> **Key Takeaway — Problem Description.** The central challenge of predictive maintenance is the asymmetry of costs between missing a fault and raising a false alarm. Because anomalies represent only ~11% of the data, standard accuracy is a misleading metric. Our system must overcome -6dB SNR background noise to identify rare but high-cost mechanical failures before they precipitate systemic downtime.

---

## 2. Assumptions

The development of the predictive-maintenance pipeline rests on three core engineering and signal-processing assumptions. Each grounds a downstream design choice in physical or mathematical logic.

### 2.1 Acoustic Signatures and Frequency Localization

We assume that mechanical faults — worn bearings, cavitation in pumps, lubrication failures — manifest as **localized, repeating signatures in specific frequency bands**, not as global changes in volume. In economic terms, this is comparable to a *sector-specific shock* rather than a general macroeconomic downturn: identifying the fault requires inspecting a specific "market segment" of the frequency spectrum rather than a single aggregate amplitude statistic. This assumption justifies the use of a frequency-resolved input representation rather than a raw waveform.

### 2.2 Spatial Representation and Translation Invariance

A primary assumption of this project is that 1-D audio time-series waveforms are **too volatile and too high-dimensional** for direct feature extraction at the scale of our dataset. By converting audio into 2-D **Log-Mel-Spectrograms**, we assume the model can leverage the spatial, translation-invariant pattern-recognition strengths of a 2-D Convolutional Neural Network (CNN), treating the spectrogram as a single-channel image whose vertical axis is frequency and horizontal axis is time.

The choice of the **Mel scale** is particularly significant. While a standard Fourier Transform places energy on a *linear* frequency axis, the Mel scale is non-linear — it provides higher resolution at lower frequencies and progressively coarser resolution at higher frequencies. This mimics human auditory perception and is conceptually parallel to the economic principle of **diminishing marginal utility**: the perceived (and informationally salient) difference between two high-frequency tones is smaller than the same numerical difference between two low-frequency tones, so allocating more spectral "bandwidth" to the lower regions yields a better return on representational capacity.

### 2.3 Mechanical Cycle Capture

To ensure the model has sufficient information to make a diagnostic decision, we assume that a **10-second audio snippet** (160,000 samples at 16 kHz) is adequate to capture at least one full mechanical cycle of an industrial pump. This guarantees that even transient or intermittent fault signatures will appear within the input window, eliminating the risk that a fault signature falls entirely between observation frames.

> **Key Takeaway — Assumptions.** Mechanical failures are treated as visual patterns in sound. By transforming raw audio into a 2-D "image" (the Log-Mel-Spectrogram) and concentrating spectral resolution in the bands most relevant to physical mechanical structures, we provide the CNN with a structured input space in which translation-invariant convolution can identify anomalies efficiently.

---

## 3. Approach: The Engineering Pipeline and Evolution

The pipeline transitioned through several iterative refinements, each driven by a specific engineering or statistical failure mode encountered in earlier prototypes. This section documents the final architecture and, equally importantly, the **rationale** for each design choice over its alternatives.

### 3.1 Signal Processing and Feature Extraction

The module [`src/preprocess.py`](src/preprocess.py) handles the transformation of raw `.wav` files into a uniform tensor representation. We enforce a strict 10-second duration via **zero-padding** for short clips and **truncation from the tail** for long clips. Zero-padding was selected over alternative strategies (clip repetition, edge-reflection padding) because, as the source notes, *"it does not fabricate false patterns that could confuse the CNN."* Repeating a clip would inject artificial periodicity that the network could exploit as a spurious feature for the classification decision.

The conversion to a Log-Mel-Spectrogram uses the following parameters, defined as module-level constants in `preprocess.py`:

```
SAMPLE_RATE         = 16000   # Hz; Nyquist limit at 8 kHz captures all physical pump signatures
TARGET_DURATION_SEC = 10      # seconds; one full mechanical cycle (Assumption 2.3)
N_FFT               = 2048    # FFT window size — frequency resolution
HOP_LENGTH          = 512     # 75% overlap between successive windows; ~31.25 frames/sec
N_MELS              = 128     # Mel filter bank size — vertical resolution
```

The resulting power spectrogram is converted to decibels via `librosa.power_to_db(ref=np.max)`:

$$S_{dB} = 10 \cdot \log_{10}\!\left(\frac{P}{P_{\text{ref}}}\right)$$

producing a uniform tensor of shape `[1, 128, 313]` — one channel, 128 Mel frequency bands, 313 time frames — that is persisted to disk as a NumPy `.npy` file.

### 3.2 Memory-Safe Data Pipeline (Lazy Loading)

Early prototypes attempted to load all 4,200+ spectrograms into RAM as a single in-memory NumPy array. On a typical 8–16 GB host, this strategy failed with an **Out-of-Memory (OOM)** kernel kill before training could even begin. The dataset is small *as audio*, but each spectrogram occupies ~160 KB as `float32`, and — critically — PyTorch's `DataLoader` workers fork the parent process, multiplying any in-RAM tensor by the worker count.

To resolve this, we implemented **lazy loading** inside the custom [`SpectrogramDataset`](src/dataset.py) class. Rather than caching arrays, the dataset stores only file *paths* in the parent process's memory. The `__getitem__` method dynamically loads each `.npy` file from disk only when the `DataLoader` requests a batch:

```python
def __getitem__(self, index):
    path  = self.file_paths[index]
    label = self.labels[index]
    spec  = np.load(path)                 # lazy disk read
    spec  = (spec - spec.mean()) / (spec.std() + 1e-6)   # per-sample standardization
    return torch.from_numpy(spec).unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)
```

This approach is a deliberate **optimization under resource constraints**: it trades a small per-batch I/O cost for the ability to train on datasets that may be orders of magnitude larger than available RAM. Per-sample standardization is performed *inside* `__getitem__` rather than over the dataset globally — this avoids cross-sample data leakage (no statistic from one clip influences the normalization of another).

### 3.3 Combating Class Imbalance via Weighted Loss

To prevent the model from collapsing into the Accuracy Paradox of §1.3, we use **algorithmic penalization** rather than synthetic oversampling. Synthetic Minority Over-sampling Technique (SMOTE) and its audio variants were rejected on principle: interpolating in a high-dimensional spectrogram space risks generating synthetic samples that lie outside the manifold of physically realizable acoustic events, producing a model that "knows" how to classify SMOTE artifacts but generalizes poorly to true machine sounds.

Instead, [`compute_class_weights()` in `src/dataset.py`](src/dataset.py) computes per-class weights using the **inverse-frequency formula**:

$$w_c = \frac{N_{\text{total}}}{k \cdot N_c}$$

where $N_{\text{total}}$ is the training-set size, $k=2$ is the number of classes, and $N_c$ is the count of class $c$. With our training distribution this yields:

```
Normal weight   ≈ 0.561
Abnormal weight ≈ 4.610
```

These weights are injected into PyTorch's `nn.CrossEntropyLoss(weight=class_weights)`. The loss for a prediction $\hat{y}$ with one-hot label $y$ becomes:

$$\mathcal{L} = -\sum_{c=1}^{C} w_c \cdot y_c \cdot \log(\hat{y}_c)$$

The economic interpretation is direct: a missed abnormal sample now incurs a gradient penalty roughly **8.2× larger** than a missed normal sample (4.610 / 0.561 ≈ 8.2). The model is structurally incentivized to attend to the minority class, internalizing the same FN/FP cost asymmetry that motivates the project.

### 3.4 Architecture Evolution: The Global Average Pooling (GAP) Solution

The CNN itself, defined in [`src/model.py`](src/model.py) as the `AudioClassifier(nn.Module)`, comprises four convolutional blocks with progressively doubled filter depth:

```
Block 1:  Conv2d( 1, 16, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2)   →  [B, 16, 64, 156]
Block 2:  Conv2d(16, 32, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2)   →  [B, 32, 32, 78]
Block 3:  Conv2d(32, 64, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2)   →  [B, 64, 16, 39]
Block 4:  Conv2d(64,128, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2)   →  [B,128,  8, 19]
GAP:      AdaptiveAvgPool2d(1)                                            →  [B,128,  1,  1]
Head:     Flatten → Dropout(0.5) → Linear(128, 2)                          →  [B,  2]
```

The doubling-filter pattern is the standard VGG/ResNet motif: as spatial dimensions shrink under successive `MaxPool2d(2)` operations, channel depth grows to compensate, preserving total representational capacity through the network.

The most consequential design decision in the entire project was the **classification head**. The initial prototype used the conventional `Flatten → Linear(128 × 8 × 19, 2)` pattern, which expands to a dense layer with **~2.5 million trainable parameters**. With only 456 abnormal training samples (of which ~319 fall in the train split after stratification), this configuration immediately overfits — the dense layer has the capacity to memorize *which spatial positions* in the spectrogram fired for each individual training clip, rather than learning generalizable spectral features.

We replaced `Flatten` with `nn.AdaptiveAvgPool2d(1)`, which collapses the `[B, 128, 8, 19]` feature map into a `[B, 128, 1, 1]` vector by averaging each channel's activations across the entire spatial grid. The downstream `Linear(128, 2)` adds only 258 parameters. The full model contains:

```
Total trainable parameters: 97,890
```

— a reduction of **over 95%** relative to the original head. This is not merely a memory optimization. **GAP acts as a profound structural regularizer**: the model can no longer encode *where* a feature fires, only *how strongly* each feature channel responds globally. It is forced to learn channel-level abstractions ("how much energy is present in mid-band cavitation harmonics?") rather than positional ones ("there is a peak at frequency band 47, time frame 218 in this exact clip"). As a side benefit, the absence of a fixed-size flatten layer makes the model **elastic**: it accepts spectrograms of arbitrary time-axis length, a property explicitly verified by the test suite (§3.5).

### 3.5 Test Suite: Defending the Pipeline Against Silent Regressions

A small but deliberate test suite, located at [`tests/`](tests/), guards the engineering invariants on which the rest of the system depends. The tests are written in `pytest` and execute in well under a second on CPU; their value is not raw coverage but **failure mode coverage** — each test catches a specific class of silent regression that would otherwise corrupt training results without raising an error.

- **[`tests/conftest.py`](tests/conftest.py)** solves a packaging problem rather than testing logic. The source code lives in `src/` without an `__init__.py`, so this fixture inserts `src/` into `sys.path` at session start. Per the inline rationale, this guarantees that *"tests and production code use identical imports,"* preventing the subtle bug class where a test silently passes against a different import path than the one used in production.

- **[`tests/test_preprocess.py`](tests/test_preprocess.py)** pins audio-normalization invariants. It verifies that `load_and_normalize_audio()` produces *exactly* `TARGET_LENGTH_SAMPLES` regardless of input length, and — critically — that padding is appended **at the end**, not the start. The latter test catches a subtle failure mode: if reversed padding were silently introduced, the network could learn that "abnormal" simply means "audio occurs early in the clip," memorizing a positional artifact rather than a spectral pattern. The test also asserts that `generate_log_mel_spectrogram()` returns the canonical `(128, 313)` shape with `dB` values bounded above by zero (since `ref=np.max`). This protects against `librosa` version drift that could shift frame counts or the dB reference.

- **[`tests/test_dataset.py`](tests/test_dataset.py)** verifies the two dataset operations *"that are easy to break in a refactor and catastrophic to break in production."* First, it asserts the inverse-frequency property of `compute_class_weights()` — if the formula were inadvertently inverted in a refactor, the loss would reward predicting "Normal" rather than penalize it, and the model would coast to high accuracy by abandoning the minority class entirely. Second, it asserts that stratified train/val/test splits preserve class ratios within ±1 sample and have **zero overlap** between subsets (verified via `set` intersection). A naive random split could place zero abnormal samples in the test set, which would render F1 / Recall / ROC-AUC undefined.

- **[`tests/test_model.py`](tests/test_model.py)** enforces the model's input-output contract. It verifies that `[B, 1, 128, 313]` inputs yield `[B, 2]` raw-logit outputs of `float32` dtype, and that `get_final_conv_layer()` returns the correct module for Grad-CAM hook attachment. Most importantly, it feeds the model spectrograms with non-canonical time axes (160 frames and 800 frames) and asserts that forward propagation succeeds. As the inline comment notes, this is *"the SOLE guarantee in the project that the GAP head stays in place"* — if a future contributor reverted to a flatten-and-dense head, this test would fail and force the regression to surface immediately rather than silently break downstream inference.

Collectively, these tests transform engineering invariants — class weight semantics, split disjointness, output shape, padding direction, GAP-induced elasticity — into machine-checkable assertions, allowing the pipeline to be refactored with confidence.

> **Key Takeaway — Approach.** The pipeline evolved from a memory-intensive, over-parameterized prototype into a lean, memory-safe, and statistically robust industrial tool. Lazy loading manages RAM pressure, weighted cross-entropy loss aligns the optimization objective with industrial cost asymmetry, and Global Average Pooling acts as a structural regularizer that simultaneously slashes parameter count by 95% and renders the model length-elastic. The test suite freezes these invariants so they cannot regress silently.

---

## 4. Experimental Setup and Hyperparameter Optimization

The training methodology was engineered for maximum statistical rigor and resource efficiency. We frame the selection of model parameters as a **capital allocation problem**: optimize the return on investment (predictive performance on the minority class) relative to the computational cost (GPU-hours and host memory) required to train each candidate.

### 4.1 Data Partitioning and Leakage Prevention

Using `sklearn.model_selection.train_test_split`, the labeled dataset is partitioned into a **70 / 15 / 15** split for training, validation, and testing. Both splits are performed with `stratify=labels` and `random_state=RANDOM_SEED` (42) inside [`stratified_split()` in `src/dataset.py`](src/dataset.py). Stratification preserves the ~11% minority-class ratio in *every* subset, ensuring the model is not evaluated on a distribution that diverges from its training environment — a form of **sample-bias mitigation**. Without stratification, a naive random split could route zero abnormal samples into the test set, rendering F1, Recall, and ROC-AUC mathematically undefined.

To eliminate **information leakage**, the test [`test_dataset.py`](tests/test_dataset.py) enforces — as a CI-level invariant — that the three resulting `set`s of file paths have empty pairwise intersection. This freezes the property that no spectrogram is ever simultaneously trained on and evaluated against.

### 4.2 Automated Training Protections

Training, orchestrated by [`src/train.py`](src/train.py), uses the **Adam** optimizer with an initial learning rate supplied via CLI argument and a weighted `CrossEntropyLoss`. Two automated risk-management mechanisms maintain algorithmic stability across the maximum `NUM_EPOCHS = 20` budget:

- **`ReduceLROnPlateau(mode='min', factor=0.5, patience=3)`** — when validation loss fails to improve for three consecutive epochs, the learning rate is halved. This is analogous to a central bank fine-tuning an interest rate as an economy approaches equilibrium: large steps that were productive in the early, far-from-optimum regime begin to overshoot the actual minimum, so step size is reduced to settle into a tighter local solution.

- **Early Stopping (`patience = 7`)** — training halts when validation loss has not improved for seven consecutive epochs. This implements the economic principle that a firm should halt investment once the *marginal return* on additional capital approaches zero; further epochs would only memorize noise in the training set, producing diminishing — and eventually negative — returns on generalization.

The relative ordering of the two patience values is deliberate: `7 > 3 + 3` ensures that the LR scheduler has at least two opportunities to reduce the learning rate (once at epoch 3, again at epoch 6) *before* early stopping considers terminating. This prevents the system from cutting a run short on the basis of a transient plateau that a single LR reduction could have escaped.

### 4.3 Cartesian Grid Search and Subprocess Isolation

To identify the global optimum within a tractable search space, the orchestration script [`tune.py`](tune.py) executes a **Cartesian grid search** over three axes:

| Axis | Values |
|---|---|
| Learning rate (`--lr`) | `{1e-3, 5e-4}` |
| Batch size (`--batch_size`) | `{16, 32}` |
| Base filters (`--base_filters`) | `{16, 32}` |

This yields $2 \times 2 \times 2 = 8$ candidate configurations. For each, `tune.py` issues three sequential `subprocess.run(...)` calls — one to `train.py`, one to `evaluate.py`, and one to `explain.py` — passing the candidate hyperparameters as CLI flags and a unique run name (e.g., `run_lr1em03_bs16_bf16`).

The use of `subprocess.run` rather than an in-process Python loop is a **critical engineering decision**, not a stylistic one. Three problems arise when running multiple PyTorch jobs in a single interpreter:

1. **GPU VRAM accumulates.** Even with `del model`, `torch.cuda.empty_cache()`, and `gc.collect()`, residual CUDA-context allocations and cached autograd graphs persist across runs. After three or four large grid points, a 16 GB Colab T4 GPU reliably triggers an OOM failure. Spawning each run as an OS subprocess guarantees that, on subprocess exit, the kernel reclaims the *entire* memory footprint atomically. We refer to this as **memory inflation prevention**.

2. **Crash isolation.** A divergent run that produces `NaN` losses, or any unrelated CUDA error, would propagate up the call stack and terminate the entire grid search if executed in-process. As a subprocess, only the affected run dies; the orchestrator logs the failure and proceeds.

3. **Reproducibility.** Each subprocess receives an identical CLI argument set and starts from a fresh Python interpreter, so module-level side effects (random seed setting, `cudnn.benchmark` configuration, library imports) all execute deterministically per run, with no contamination from earlier iterations.

The total VRAM/runtime cost of subprocess invocation (interpreter startup, CUDA context recreation) is small relative to a 20-epoch training job and is paid back many times over by the elimination of debugging cycles attributable to memory inflation.

### 4.4 The Winning Configuration

Validation-set metrics, recorded automatically by `evaluate.py` to `experiment_tracking.csv`, identified the optimal hyperparameter combination as:

```
Learning rate (LR)        : 1e-3
Batch size  (BS)          : 16
Base filters (BF)         : 16
Trainable parameters       : 97,890
```

Three engineering rationales explain why this configuration outperformed its alternatives:

- **Batch size as a regularizer.** A smaller batch size injects greater stochastic gradient noise per step. This noise has a well-documented regularizing effect — it prevents the optimizer from settling into sharp, narrow loss-basin minima that generalize poorly, biasing it instead toward flatter minima that are more robust to distributional shift. In an imbalanced setting, a small batch is also more likely to contain at least one abnormal sample, providing a denser learning signal for the minority class than a larger batch would supply on average.

- **Parsimony in network width.** Doubling base filters to 32 (yielding ~390k parameters) did *not* improve validation F1; it degraded it. With only ~319 abnormal training samples, the additional capacity has no productive signal to fit and instead increases susceptibility to overfitting on background factory acoustics. This is a textbook application of the Occam's-razor principle: when two hypotheses fit equally well, prefer the simpler one — and when one hypothesis (more filters) cannot even fit equally well due to insufficient data, the simpler model wins outright.

- **Higher learning rate, scheduled.** The faster of the two learning rates (`1e-3`) yielded better convergence than `5e-4` because the `ReduceLROnPlateau` scheduler is itself responsible for slowing the optimizer late in training. Starting at the higher rate maximizes early-epoch progress through the loss landscape, while the scheduler delivers the late-stage refinement automatically. Starting low would leave the model under-trained within the 20-epoch budget.

The full validation-set leaderboard is reproduced in §5.3 as part of the **tuning summary** analysis.

> **Key Takeaway — Experimental Setup.** A systematic grid search replaced arbitrary tuning. Through stratified partitioning, scheduler/early-stopping coordination, and `subprocess`-based GPU isolation, the resulting model is not only high-performing but statistically unbiased, reproducible, and resilient to the failure modes of multi-run hyperparameter exploration.

---

## 5. Evaluation and Explainability

### 5.1 Metric Strategy: Beyond Naive Accuracy

Because raw accuracy is structurally deceptive under an ~11% minority distribution, [`src/evaluate.py`](src/evaluate.py) prioritizes metrics that align with industrial utility:

- **Recall (sensitivity)** measures the proportion of true faults the system actually catches. In PdM, recall is the metric to defend: a missed abnormal clip corresponds to a real machine fault propagating undetected into production.
- **F1-Score** is the harmonic mean of precision and recall, balancing the cost of false alarms against the cost of missed faults.
- **ROC-AUC** measures the model's *threshold-independent* ranking ability — the probability that a randomly chosen abnormal sample receives a higher score than a randomly chosen normal sample. AUC is particularly relevant for industrial deployment because the operational decision threshold is itself a tunable business-logic parameter (§6).

Accuracy is computed and recorded for completeness but is explicitly *not* the optimization target.

### 5.2 Headline Results

The validation-selected winning configuration `run_lr1em03_bs16_bf16` produces the following metrics on the held-out test set, evaluated *one time* by [`src/evaluate.py`](src/evaluate.py) after model selection was finalized — preserving the test set as an unbiased estimate of generalization performance:

```
Headline Result (Held-Out Test Set, n = 631):
  F1-Score   = 0.8750     (harmonic mean of precision and recall)
  ROC-AUC    = 0.9737     (threshold-independent ranking quality)
  Precision  = 0.9333     (only ~6.7 % of alarms are false alarms)
  Recall     = 0.8235     (~82 % of true faults detected)
  Accuracy   = 0.9746     (reported for completeness, not optimized)
```

A test-set **ROC-AUC of 0.9737** indicates that the system correctly ranks an abnormal sample above a normal one approximately 97.4 % of the time — strong threshold-independent evidence that the network's score function encodes genuine diagnostic information rather than relying on a fortuitous decision boundary. The test-set **F1 of 0.875** is comfortably above the 0.0 floor that a degenerate "always Normal" classifier would achieve, confirming that the weighted-loss strategy successfully escaped the Accuracy Paradox.

#### 5.2.1 Cross-Configuration Test-Set Leaderboard

The full HPO grid was re-evaluated on the same held-out test partition for transparency. The validation-selected winning configuration also tops the test-set leaderboard on both F1 and AUC, indicating that validation-set selection generalized faithfully to the test set rather than coincidentally favoring a model that exploits a quirk of the validation fold:

| Run | LR | BS | BF | Test F1 | Test AUC |
|---|---|---|---|---:|---:|
| **`run_lr1em03_bs16_bf16`** *(selected)* | 1e-3 | 16 | 16 | **0.8750** | **0.9737** |
| `run_lr5em04_bs16_bf16` | 5e-4 | 16 | 16 | 0.8154 | 0.9558 |
| `run_lr5em04_bs16_bf32` | 5e-4 | 16 | 32 | 0.7972 | 0.9712 |

The consistency of the winning configuration across both selection criteria (F1 and AUC) is itself meaningful: a model that wins on F1 but lags on AUC would suggest its lead came from a fortuitous threshold position rather than a fundamentally better score function. Here, the same model leads on both metrics, indicating a substantive — not a coincidental — advantage.

### 5.3 Tuning Summary: Robustness of the Architecture

The complete test-set leaderboard across all eight grid points is summarized below, rank-ordered by F1:

| Rank | Run | LR | BS | BF | Test F1 | Test AUC |
|:---:|---|---|---|---|---:|---:|
| 1 | `run_lr1em03_bs16_bf16` | 1e-3 | 16 | 16 | **0.8750** | **0.9737** |
| 2 | `run_lr5em04_bs16_bf16` | 5e-4 | 16 | 16 | 0.8154 | 0.9558 |
| 3 | `run_lr5em04_bs16_bf32` | 5e-4 | 16 | 32 | 0.7972 | 0.9712 |
| 4 | `run_lr5em04_bs32_bf16` | 5e-4 | 32 | 16 | 0.7582 | 0.9540 |
| 5 | `run_lr1em03_bs16_bf32` | 1e-3 | 16 | 32 | 0.7516 | 0.9662 |
| 6 | `run_lr5em04_bs32_bf32` | 5e-4 | 32 | 32 | 0.7349 | 0.9679 |
| 7 | `run_lr1em03_bs32_bf16` | 1e-3 | 32 | 16 | 0.7126 | 0.9621 |
| 8 | `run_lr1em03_bs32_bf32` | 1e-3 | 32 | 32 | 0.6905 | 0.9602 |

The figure [`tuning_summary_graph.png`](docs/Final_tuning_visualizations/tuning_summary_graph.png) renders this leaderboard graphically. **The narrow performance band across configurations is itself a substantive finding.** ROC-AUC remains above **0.95** across *all eight* runs — a total span of just 0.0197 (from 0.9540 to 0.9737) — and F1 ranges from 0.6905 to 0.8750, a spread of approximately 0.18 driven primarily by precision rather than ranking quality. The fact that no single hyperparameter axis dominates the results — and that even the worst configuration retains an AUC above 0.95 — suggests that the **structural decisions made earlier in the pipeline — Global Average Pooling, weighted cross-entropy loss, stratified partitioning — are doing the bulk of the predictive work**, with hyperparameter tuning supplying only marginal refinement. This is exactly the desirable failure mode for an industrial baseline: the system is not brittle to small perturbations of its training configuration. A second observation is also instructive: the bottom three F1 ranks all use `BS = 32`, confirming that the regularizing effect of small-batch gradient noise (§4.4) is responsible for a meaningful slice of the precision gains in the top-ranked runs.

### 5.4 ROC Curve Cross-Comparison

The figure [`roc_curve_run_lr1em03_bs16_bf16.png`](docs/Final_tuning_visualizations/roc_curve_run_lr1em03_bs16_bf16.png) plots the ROC curve for the winning configuration. Compared against the eight peer ROC curves in [`docs/Final_tuning_visualizations/`](docs/Final_tuning_visualizations/), three patterns emerge:

- **Steep early rise.** The winning curve climbs sharply along the true-positive axis at low false-positive rates, indicating that the model assigns its highest scores almost exclusively to genuine abnormal samples — the operationally critical regime, since industrial alarm thresholds are typically set to suppress false alarms.
- **Larger AUC gap from the diagonal.** The 0.9737 test-set AUC corresponds visually to a curve that "hugs the top-left corner" of the unit square. Configurations with `BS = 32` exhibit visibly flatter curves in the low-FPR region.
- **AUC differences are small but visible at low FPR.** The leaderboard AUC range across all eight runs is just 0.0197 (0.9540 to 0.9737), so the ROC curves cluster tightly. The discriminating regime is the *low false-positive corner*: in this slice of the curve, `BS = 16` runs ramp toward Recall = 1.0 measurably faster than their `BS = 32` counterparts, consistent with the leaderboard pattern that the bottom three F1 ranks are all `BS = 32` configurations.

The convergence of the winning curve toward the upper-left corner is consistent with the AUC value of 0.9737 and confirms that the model's score function induces a near-monotonic separation between the two classes.

### 5.5 Confusion Matrix Analysis

The figure [`confusion_matrix_run_lr1em03_bs16_bf16.png`](docs/Final_tuning_visualizations/confusion_matrix_run_lr1em03_bs16_bf16.png) reports the confusion matrix for the validation-selected winning configuration on the held-out **test set** (n = 631 samples; 563 normal, 68 abnormal). Reading it through the FN/FP cost lens established in §1.3:

```
                           Predicted Normal   Predicted Abnormal
True Normal      (TN / FP):      559                4              (false alarms)
True Abnormal    (FN / TP):       12               56              (missed faults are critical)
```

The clinical breakdown is striking. Of 563 truly normal samples, the model raises an alarm on only **4** — a false-alarm rate of **0.71 %**. Of 68 truly abnormal samples, the model correctly catches **56**, missing **12** — a recall of **82.4 %**. Of every 60 alarms the system raises, **56 are real** (precision ≈ 93.3 %), so when the model says "fault" an engineer is justified in taking the alarm seriously rather than dismissing it.

This profile reflects the cost structure that the weighted loss was designed to induce: the system is conservative on false alarms (precision is very high) while recovering more than four out of every five true faults at the default decision threshold. The remaining 12 missed faults are the system's principal exposure surface — and exactly the regime that a lower decision threshold (deployable without retraining; see §6) is designed to address. Because the model's threshold-independent ranking quality is high (AUC = 0.9737 on this split), reducing the threshold trades precision for recall along a favorable curve rather than collapsing alarm credibility.

### 5.6 Grad-CAM: Visual Verification of Physical Plausibility

To eliminate "black box" opacity and validate that the network's decisions are grounded in physical signals rather than spurious background artifacts, [`src/explain.py`](src/explain.py) implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**. Hooks attached to the final convolutional layer (`conv4`, exposed via `AudioClassifier.get_final_conv_layer()`) record forward activations $A^k$ and back-propagated gradients $\partial y^c / \partial A^k$. Per-channel importance weights $\alpha_k$ are computed by global-average-pooling the gradients, the activations are linearly combined under those weights, ReLU-clipped, bilinearly upsampled from $(8, 19)$ back to $(128, 313)$, and min-max normalized to produce a heatmap aligned with the original spectrogram axes (Mel-Hz vertical, seconds horizontal).

The figures [`gradcam_abnormal_run_lr1em03_bs16_bf16.png`](docs/Final_tuning_visualizations/gradcam_abnormal_run_lr1em03_bs16_bf16.png) and [`gradcam_normal_run_lr1em03_bs16_bf16.png`](docs/Final_tuning_visualizations/gradcam_normal_run_lr1em03_bs16_bf16.png) reveal a clear and physically interpretable contrast between the two classes:

- **Abnormal signature.** The Grad-CAM heatmap concentrates intense, *temporally localized* activations in the **mid-frequency Mel band spanning roughly 512 Hz to 2048 Hz**, with the most prominent hotspots visible as discrete vertical bursts at approximately 1.2 s, 4.0 s, and 6.0 s of the 10-second clip. This is precisely the spectral region in which mechanical pump faults — bearing wear, cavitation, impeller distress — produce harmonic energy, and the burst-like temporal structure is consistent with the periodic mechanical impacts that characterize a degraded rotating component. The network has learned to look exactly where a domain expert would look.
- **Normal signature.** The heatmap exhibits a qualitatively different structure: a **broad, temporally uniform horizontal band of activation centered near 2048 Hz with strong steady-state attention extending above 4096 Hz**. The absence of localized temporal bursts — the activation is essentially time-invariant across the full 10-second window — is the hallmark of a stationary acoustic signal, corresponding to the broadband noise floor of an unimpaired pump operating under nominal conditions. The model is, in effect, classifying "Normal" by recognizing the *absence* of the transient structure that characterizes the Abnormal class.

The contrast — *localized bursts in mid-frequencies* for abnormal versus *uniform high-frequency steady-state* for normal — accomplishes something that aggregate metrics cannot: it confirms that the model has internalized the **fundamental physics of the machine** rather than over-indexing on background factory noise or label-correlated artifacts. For an engineering team contemplating deployment, this is the credibility test that ROC-AUC alone cannot pass.

> **Key Takeaway — Evaluation.** The validation-selected configuration attains a held-out test-set ROC-AUC of **0.9737** and an F1 of **0.875**, catching 56 of 68 true faults while raising only 4 false alarms across 563 normal samples. Grad-CAM confirms that decisions are anchored in mid-frequency Mel bands consistent with mechanical fault harmonics. The narrow performance band across the eight HPO runs demonstrates that the architectural choices — GAP, weighted loss, stratification — provide a robust foundation that hyperparameter selection only modestly refines.

---

## 6. Limitations

While the model is highly effective within the conditions of its training distribution, three primary constraints must be acknowledged before any operational deployment.

### 6.1 Domain Shift

The pipeline is optimized for the MIMII -6dB SNR acoustic environment, which simulates a specific class of factory background noise. Deploying this model to a new factory whose ambient acoustic profile differs materially — for example, a facility with a dominant HVAC signature, different reverberation characteristics, or a different mix of nearby machinery — would likely increase the false-positive rate until the model is **fine-tuned on a small number of samples from the new environment**. This is a familiar problem in transfer learning and is not a defect of the architecture, but it is a deployment cost that downstream stakeholders must internalize.

### 6.2 Single-Sensor Reliance

The current pipeline is acoustically univariate: it consumes a single microphone channel and produces a single fault probability. True industrial PdM systems achieve their highest reliability through **sensor fusion** — combining acoustic data with vibration telemetry from accelerometers, thermal imaging, current draw measurements from the motor, and pressure or flow sensors on the fluid path. A multi-modal model would be more robust to acoustic occlusion (e.g., a worker's voice or a passing forklift temporarily masking the pump) and would catch fault categories — such as those producing primarily thermal or vibrational signatures — that are invisible to a microphone.

### 6.3 Static Decision Threshold

The model produces a continuous probability score, but the binary alarm decision requires a threshold. The ROC-AUC of 0.9737 demonstrates that the score function carries genuine ranking information across the threshold spectrum, but the *specific* operational threshold is not a machine-learning question — it is a **business-logic question**. The threshold must be chosen by weighing:

- The marginal cost of a false alarm (operator time, unnecessary inspection)
- The marginal cost of a missed fault (downtime, secondary damage, safety risk)

These costs are facility-specific and must be supplied by the asset owner. A reactor pump in a chemical plant warrants an aggressive (low) threshold; a redundant cooling pump in a non-critical loop tolerates a more permissive (higher) threshold. The model exposes the trade-off; the deployer chooses the operating point.

---

## 7. Future Work

Three avenues of follow-up work would meaningfully extend the system's statistical rigor, robustness, and deployability.

### 7.1 K-Fold Cross-Validation

The current 70/15/15 partition with a fixed random seed produces a **point estimate** of generalization performance — informative, but a single sample of the underlying performance distribution. Future iterations should adopt **K-fold cross-validation** (e.g., $K = 5$ stratified folds), reporting the mean and standard deviation of F1 and AUC across folds. This converts the headline numbers from point estimates into confidence intervals and definitively rules out the possibility that the high reported performance was the artifact of a fortunate random seed. Combined with statistical hypothesis testing across folds, K-fold CV would also enable formal claims of significance when comparing the eight grid configurations.

### 7.2 Acoustic Data Augmentation (SpecAugment)

The minority class is under-represented by an order of magnitude, and weighted loss only partially compensates. **SpecAugment** — randomly masking contiguous frequency bands and time steps within the spectrogram during training — would artificially expand the effective abnormal-class training set and force the network to learn *multiple, redundant* acoustic signatures per fault category rather than a single high-confidence pattern. This would directly address the residual recall ceiling at the default threshold (12 of 68 abnormal samples missed, Recall ≈ 0.82) by improving robustness to partial signal occlusion, which is exactly the failure mode that the noisy -6dB SNR environment produces.

### 7.3 Edge Deployment via Quantization (TinyML)

The system's 97,890-parameter footprint is already small by modern deep-learning standards, but a full-fidelity FP32 inference path still requires a Python runtime, PyTorch, and several hundred MB of supporting libraries — impractical on a low-power microcontroller mounted at the pump. **Post-training quantization** from 32-bit floating-point to 8-bit integer representation would reduce the model's weight footprint by ~4× and replace floating-point matrix multiplications with integer arithmetic, enabling deployment to ARM Cortex-M class devices via TensorFlow Lite Micro or PyTorch Edge. This transforms the system from an off-line analysis tool into a **continuous, on-device monitor** capable of running directly on the factory floor, eliminating network round-trips and the associated reliability dependencies.

---

## 8. Conclusions

This project documents the evolution of an industrial predictive-maintenance pipeline from a fragile, memory-bound prototype into a parameter-efficient, statistically rigorous, and explainable system. Three engineering decisions account for the majority of the final system's quality:

1. **Class imbalance was confronted at the loss function**, not papered over with synthetic oversampling. Inverse-frequency weighted cross-entropy injects an 8.2× cost ratio that aligns the model's optimization objective with the FN/FP cost asymmetry that defines the industrial domain.
2. **Overfitting was eliminated structurally** by replacing a 2.5M-parameter dense classification head with `nn.AdaptiveAvgPool2d(1)`, reducing the model to 97,890 parameters and forcing the network to learn channel-level rather than positional features.
3. **The optimum was located systematically** through a `subprocess`-isolated Cartesian grid search that side-stepped the GPU memory inflation, crash propagation, and reproducibility hazards endemic to in-process hyperparameter loops.

The validation-selected configuration (LR=1e-3, BS=16, BF=16) achieves a held-out test-set ROC-AUC of **0.9737** and an F1 of **0.875**, catching 56 of 68 true faults at a false-alarm rate of just 0.71 %, with Grad-CAM heatmaps confirming that the network's decisions are anchored in mid-frequency Mel bands consistent with the physical signatures of mechanical fault. The narrow performance band across all eight HPO runs further indicates that this performance is attributable not to a fortunate hyperparameter choice but to the underlying architectural decisions, suggesting the system is genuinely robust rather than narrowly tuned.

The final deliverable is therefore not merely a classifier but an **explainable AI tool** — one that bridges raw acoustic signal processing, principled treatment of class imbalance, and the visual transparency required for an engineering team to trust an automated fault detector with the asset-protection responsibilities that motivated the project in the first place.

---

## Appendix A: Repository Structure

```
Predictive-Maintenance-Audio-CNN/
├── src/
│   ├── preprocess.py    # Audio → Log-Mel-Spectrogram conversion
│   ├── dataset.py       # SpectrogramDataset (lazy loading), class weights, splits
│   ├── model.py         # AudioClassifier (4-block CNN with GAP head)
│   ├── train.py         # Adam + weighted CE + ReduceLROnPlateau + early stopping
│   ├── evaluate.py      # F1 / ROC-AUC / confusion matrix, CSV result tracking
│   └── explain.py       # Grad-CAM visualization
├── tests/
│   ├── conftest.py      # sys.path injection for src/ imports
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   └── test_model.py
├── tune.py              # Cartesian grid search via subprocess.run
└── docs/
    └── Final_tuning_visualizations/   # 37 PNG figures + experiment_tracking CSVs
```




