"""
dataset.py — PyTorch Data Pipeline for Log-Mel-Spectrograms
============================================================
This script is the SECOND stage of our Predictive Maintenance Audio CNN project.
After preprocess.py has produced .npy Log-Mel-Spectrogram files in
data/processed/normal/ and data/processed/abnormal/, this script builds the
PyTorch infrastructure that feeds those files into the CNN during training.

WHAT THIS FILE PROVIDES:
    1. File discovery + binary label assignment (normal=0, abnormal=1).
    2. Stratified train/val/test splitting (70/15/15) via scikit-learn.
    3. Class weight computation — critical because MIMII is typically imbalanced
       (Way more "normal" files than "abnormal" ones). Without class weighting,
       the model can achieve ~95% accuracy by simply predicting "normal" every
       time, which is useless for anomaly detection.
    4. A custom PyTorch Dataset class that uses LAZY LOADING — spectrograms are
       read from disk one at a time inside __getitem__ rather than loaded all at
       once into RAM. This lets us train on datasets far larger than memory. This was a major issue early on.
    5. A factory function that wraps the three Datasets into DataLoader objects
       with appropriate shuffling and batching behavior.

WHY LAZY LOADING?
    A single 10-second Log-Mel-Spectrogram at (128, 313) in float32 is:
        128 * 313 * 4 bytes = ~160 KB
    The full MIMII dataset can contain thousands of files per machine type.
    With 5,000 files that's ~800 MB — manageable, but multiplied by machine
    types and augmentations it balloons fast. On Colab's free tier (12 GB RAM)
    and shared with PyTorch's own overhead, eagerly loading everything is a
    recipe for OOM crashes. 
    **Lazy loading keeps memory use flat regardless of
    dataset size: only the current batch lives in RAM at any moment.**

WHY THE SINGLE CHANNEL DIMENSION [1, H, W]?
    PyTorch Conv2D layers expect inputs of shape [batch, channels, height, width].
    A grayscale image has 1 channel, a color image has 3 (RGB). Our Mel-spectrogram
    is effectively a grayscale image (one value per time-frequency cell), so it
    needs exactly 1 channel. The raw .npy file has shape (n_mels, n_time_frames),
    so we must insert a channel axis at position 0 → (1, n_mels, n_time_frames).
    PyTorch's DataLoader then batches these into shape [batch, 1, n_mels, n_time].

USAGE:
    Run directly to preview the pipeline on your current data:
        python src/dataset.py

    Or import in a training script / notebook:
        from src.dataset import get_dataloaders, compute_class_weights
        train_loader, val_loader, test_loader, class_weights = get_dataloaders()
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os               # Filesystem path operations
import glob             # Pattern-matching for finding .npy files
import numpy as np      # Loading .npy arrays + class weight math
import torch            # PyTorch core (tensors, device management)
from torch.utils.data import Dataset, DataLoader  # Dataset API + batch iterator
from sklearn.model_selection import train_test_split  # Stratified splitting


# ============================================================================
# CONFIGURATION — Tunable constants at module level so Colab overrides are easy.
# ============================================================================

PROCESSED_DATA_DIR = os.path.join("data", "processed")
# Parent directory containing the per-class subfolders populated by preprocess.py.

CLASS_LABELS = ["normal", "abnormal"]
# The two class subfolder names. Their POSITION in this list defines the
# integer label assigned to each class:
#   index 0 → "normal"   → label 0
#   index 1 → "abnormal" → label 1
# Binary classification convention: class 1 is the "positive" / minority class
# we want to detect (the abnormal events).

TRAIN_RATIO = 0.70   # 70% of data used to fit the model's weights
VAL_RATIO = 0.15     # 15% used to tune hyperparameters / detect overfitting
TEST_RATIO = 0.15    # 15% held out, never touched until final evaluation
# Ratios must sum to 1.0. We'll split in two stages (train vs temp, then val vs test)
# because sklearn's train_test_split is a binary split function.

RANDOM_SEED = 42
# A fixed seed makes the split reproducible across runs. Without this, every
# execution would produce a different train/val/test partition, making it
# impossible to compare results between experiments.

BATCH_SIZE = 32
# Number of spectrograms per forward/backward pass. 32 is a safe default that
# fits most GPUs. We'll tune this on Colab — larger batches give more stable
# gradients but require more VRAM.

NUM_WORKERS = 0
# Number of subprocess workers for data loading.
# IMPORTANT: On Windows, multi-worker DataLoaders (NUM_WORKERS > 0) require the
# entry script to be guarded with `if __name__ == "__main__":` because Windows
# uses "spawn" (not "fork") for multiprocessing. For local development we keep
# this at 0 (single-threaded, main process). Bump to 2–4 on Colab/Linux for
# faster I/O throughput.


# ============================================================================
# STEP 1 — FILE DISCOVERY & LABEL ASSIGNMENT
# ============================================================================

def gather_file_paths_and_labels(processed_dir=PROCESSED_DATA_DIR,
                                  class_labels=CLASS_LABELS):
    """
    Scan the processed data directory and collect every .npy file path paired
    with its integer class label.

    Parameters
    ----------
    processed_dir : str
        Parent directory containing per-class subfolders of .npy files.
        Default: "data/processed".
    class_labels : list of str
        Ordered list of class subfolder names. The index of each name is used
        as the integer label (normal=0, abnormal=1).

    Returns
    -------
    file_paths : list of str
        Absolute or relative paths to every discovered .npy file.
    labels : list of int
        Integer labels corresponding 1:1 with file_paths. 0=normal, 1=abnormal.
    """

    file_paths = []  # Will be populated with every .npy path, one per file.
    labels = []      # Parallel list: labels[i] is the class of file_paths[i].

    # Iterate over each class in order. enumerate gives us (index, name) pairs,
    # and we USE the index as the integer label so normal=0 and abnormal=1.
    for label_index, class_name in enumerate(class_labels):

        # Build the path to this class's subfolder, e.g. "data/processed/normal".
        class_dir = os.path.join(processed_dir, class_name)

        # glob returns a list of all matching file paths. We sort for determinism
        # so the order doesn't depend on filesystem quirks — important because
        # the sklearn random seed only gives reproducible splits if the input
        # order is also deterministic.
        class_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))

        print(f"  [{class_name}] Found {len(class_files)} .npy files "
              f"(label={label_index})")

        # Append all paths and their shared label to the master lists.
        # We use extend (not append) because we're adding many items at once.
        file_paths.extend(class_files)
        labels.extend([label_index] * len(class_files))
        # [label_index] * len(class_files) creates a list like [0, 0, 0, ...]
        # with one entry per file — same length as class_files.

    return file_paths, labels


# ============================================================================
# STEP 2 — STRATIFIED TRAIN / VAL / TEST SPLIT
# ============================================================================

def stratified_split(file_paths, labels, train_ratio=TRAIN_RATIO,
                     val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                     random_seed=RANDOM_SEED):
    """
    Split the dataset into training, validation, and test sets in a
    STRATIFIED fashion — preserving the class ratio in each subset.

    WHY STRATIFIED?
        Our dataset is imbalanced (many more normal than abnormal samples).
        A random split could by chance assign ZERO abnormal files to the
        validation or test set, making the metrics meaningless. Stratified
        splitting guarantees that the proportion of each class is the same
        across train/val/test subsets, so each split is a representative
        miniature of the whole dataset.

    WHY TWO-STAGE SPLIT?
        sklearn.train_test_split is a BINARY splitter — it divides one set into
        two. To get three subsets we apply it twice:
          Stage 1: all data       → train (70%)      + temp (30%)
          Stage 2: temp (30%)     → val (15% of all) + test (15% of all)
        We pass the test_size relative to the SECOND stage's input, which is
        why it's computed as test_ratio / (val_ratio + test_ratio) = 0.5.

    Parameters
    ----------
    file_paths : list of str
        All .npy file paths.
    labels : list of int
        Corresponding integer labels (same length as file_paths).
    train_ratio : float
        Fraction assigned to the training set. Default: 0.70.
    val_ratio : float
        Fraction assigned to the validation set. Default: 0.15.
    test_ratio : float
        Fraction assigned to the test set. Default: 0.15.
    random_seed : int
        Seed for the random number generator to make the split reproducible.

    Returns
    -------
    splits : dict
        A dictionary with six keys:
            "train_paths", "train_labels",
            "val_paths",   "val_labels",
            "test_paths",  "test_labels"
        Each is a list of the same length within its split.
    """

    # --- Sanity check: ratios must sum to ~1.0 ---
    # A floating-point tolerance of 1e-6 accounts for harmless rounding error
    # (e.g. 0.7 + 0.15 + 0.15 may not equal 1.0 exactly in binary float).
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {total}"
    )

    # --- Stage 1: Split off the training set ---
    # test_size here is the fraction that does NOT go to training (i.e. val + test).
    # stratify=labels tells sklearn to preserve the class proportions.
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=(val_ratio + test_ratio),  # 0.30 by default
        stratify=labels,                      # preserve class ratio
        random_state=random_seed              # reproducibility
    )

    # --- Stage 2: Split the remaining 30% into val (50%) and test (50%) ---
    # Since the remaining chunk is 30% of the total and we want val=15%, test=15%,
    # we need a 50/50 split of this subset.
    # test_size is test_ratio expressed as a fraction of the temp set:
    #     test_ratio / (val_ratio + test_ratio) = 0.15 / 0.30 = 0.50
    relative_test_size = test_ratio / (val_ratio + test_ratio)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,     # stratify within the temp set too
        random_state=random_seed
    )

    # --- Print a summary so the user can visually confirm the stratification ---
    print(f"\n  Split sizes (stratified, seed={random_seed}):")
    for name, lbls in [("train", train_labels),
                       ("val",   val_labels),
                       ("test",  test_labels)]:
        # Count how many samples of each class ended up in this split.
        n_normal = lbls.count(0)
        n_abnormal = lbls.count(1)
        total_split = len(lbls)
        print(f"    {name:6s}: {total_split:5d} total "
              f"({n_normal} normal, {n_abnormal} abnormal, "
              f"abnormal={100*n_abnormal/max(total_split,1):.1f}%)")

    # Return everything in a dict — more readable at the call site than a
    # 6-tuple, and the keys document themselves.
    return {
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths":   val_paths,   "val_labels":   val_labels,
        "test_paths":  test_paths,  "test_labels":  test_labels,
    }


# ============================================================================
# STEP 3 — CLASS WEIGHT COMPUTATION
# ============================================================================

def compute_class_weights(train_labels, num_classes=2):
    """
    Compute class weights to counteract class imbalance in the training set.

    THE MATH (inverse-frequency weighting):
        For each class c:
            weight[c] = N_total / (num_classes * N_c)
        Where:
            N_total     = total number of training samples
            num_classes = number of classes (here 2: normal + abnormal)
            N_c         = number of training samples in class c

        This formula is exactly what sklearn's compute_class_weight(
        class_weight='balanced', ...) uses. Intuition:
          - If a class is UNDER-represented (small N_c), its weight becomes LARGER,
            so mistakes on that class incur a bigger loss and the model is forced
            to pay attention to it.
          - If classes are perfectly balanced (N_c = N_total / num_classes),
            every weight equals 1.0 — no effect.

    WORKED EXAMPLE:
        Suppose training set has 900 normal and 100 abnormal (N_total=1000):
            weight[0] = 1000 / (2 * 900) ≈ 0.556
            weight[1] = 1000 / (2 * 100) = 5.000
        So abnormal mistakes cost ~9x more than normal mistakes during training.

    HOW TO USE THESE WEIGHTS:
        Pass them to PyTorch's loss function:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        CrossEntropyLoss will automatically multiply each sample's loss by the
        weight of its true class before averaging across the batch.

    Parameters
    ----------
    train_labels : list of int
        Integer labels for the training set only. We compute weights from the
        TRAINING SET, not the full dataset, to avoid leaking information about
        the val/test class distribution into training.
    num_classes : int
        Total number of classes. Default: 2 (binary classification).

    Returns
    -------
    class_weights : np.ndarray, shape (num_classes,)
        A float32 array where class_weights[c] is the weight for class c.
    """

    # Convert to numpy for vectorized operations.
    train_labels_arr = np.array(train_labels)
    n_total = len(train_labels_arr)

    # Pre-allocate the weights array with float32 (matches PyTorch's default).
    class_weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        # Count how many training samples belong to class c.
        # np.sum(boolean_array) counts True values — a common NumPy idiom.
        n_c = int(np.sum(train_labels_arr == c))

        if n_c == 0:
            # Guard against division by zero: if a class is missing from training,
            # we can't compute its weight. Set to 0 and warn the user — this
            # usually means something went wrong upstream (empty directory,
            # failed preprocessing, etc.).
            print(f"  WARNING: class {c} has 0 samples in training set. "
                  f"Setting weight to 0.")
            class_weights[c] = 0.0
        else:
            class_weights[c] = n_total / (num_classes * n_c)

    print(f"\n  Class weights (inverse frequency):")
    for c in range(num_classes):
        class_name = CLASS_LABELS[c] if c < len(CLASS_LABELS) else f"class_{c}"
        print(f"    {class_name} (label={c}): weight = {class_weights[c]:.4f}")

    return class_weights


# ============================================================================
# STEP 4 — CUSTOM PYTORCH DATASET (LAZY LOADING)
# ============================================================================

class SpectrogramDataset(Dataset):
    """
    A custom PyTorch Dataset that serves Log-Mel-Spectrograms lazily from disk.

    PyTorch's Dataset API requires subclasses to implement:
        __len__(self)       -> total number of samples
        __getitem__(self, i) -> the i-th (input, label) pair

    The DataLoader calls __getitem__ many times (optionally in parallel via
    worker subprocesses) and stacks the returned tensors into batches.

    WHY LAZY LOADING?
        We do NOT read all .npy files in __init__. Instead, we store just the
        file PATHS and labels — a few KB of memory total. When the DataLoader
        asks for sample i, we open that one file, convert it to a tensor, and
        return it. The file drops out of memory when the batch is consumed.

        Benefits:
          - Memory footprint stays constant regardless of dataset size.
          - Scales to datasets much larger than RAM.
          - Enables on-the-fly augmentation (you can transform the spectrogram
            differently each epoch without caching).

        Trade-off:
          - Disk I/O happens every epoch. On slow disks this can bottleneck
            training. Mitigation: set num_workers>0 in the DataLoader so I/O
            runs in parallel with GPU compute.

    OUTPUT TENSOR SHAPE: (1, n_mels, n_time_frames)
        PyTorch Conv2D expects [batch, channels, H, W]. After the DataLoader
        stacks a batch, shape becomes [batch, 1, n_mels, n_time]. Our spectrogram
        is inherently single-channel (grayscale-equivalent), so channels=1.
        We explicitly insert this axis with .unsqueeze(0) because the raw .npy
        file has shape (n_mels, n_time) with no channel dimension.
    """

    def __init__(self, file_paths, labels, transform=None):
        """
        Store file paths and labels. DO NOT load any arrays here.

        Parameters
        ----------
        file_paths : list of str
            Paths to .npy spectrogram files for this split.
        labels : list of int
            Integer class labels, parallel to file_paths.
        """
        # Defensive check: mismatched lengths would silently produce wrong pairings of inputs to labels — a nightmare bug to debug.
        assert len(file_paths) == len(labels), (
            f"file_paths and labels must have the same length, "
            f"got {len(file_paths)} vs {len(labels)}"
        )
        # Store as instance attributes. Python lists are references (not copies), so this costs essentially no memory.
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform #Added to store augmentation functions.

    def __len__(self):
        """
        Return the number of samples in this dataset.

        PyTorch's DataLoader uses this to know how many iterations constitute
        one epoch. Must be implemented.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load, transform, and return the (spectrogram_tensor, label_tensor)
        pair at position `idx`.

        This is called by the DataLoader for EACH sample every epoch.
        Keep it fast and avoid side effects (no printing, no writes to shared
        state) — it may run in parallel worker subprocesses.

        Parameters
        ----------
        idx : int
            Index into the dataset. 0 <= idx < len(self).
        Returns
        -------
        spectrogram : torch.FloatTensor, shape (1, n_mels, n_time_frames)
            The Log-Mel-Spectrogram as a float32 tensor with a channel axis.
        label : torch.LongTensor, scalar
            The integer class label (0 or 1) as a long tensor.
            PyTorch's CrossEntropyLoss expects long/int64 targets.
        """

        # --- Step A: Load the .npy file from disk ---
        # np.load is very fast — it just mmaps the binary contents into memory.
        # The resulting array has shape (n_mels, n_time_frames) and dtype float32
        # (as produced by preprocess.py via librosa.power_to_db).
        spectrogram_np = np.load(self.file_paths[idx])

        # --- Step B: Convert NumPy → PyTorch tensor ---
        # torch.from_numpy shares memory with the numpy array (zero copy) when
        # dtypes are compatible. We then cast to float32 explicitly to guarantee
        # the dtype regardless of what was saved — PyTorch models almost always
        # want float32 inputs (float64 would be wasteful, and mixing precisions
        # causes dtype errors).
        spectrogram = torch.from_numpy(spectrogram_np).float()

        #--- Step B.5: Per-sample Standardization
        # Centers the data around zero to stabilize the graident descent. We compute mean and std per sample to avoid data leakage across samples. This is a common practice for spectrogram inputs.
        mean = spectrogram.mean()
        std = spectrogram.std()

        if std > 1e-6: # prevents division by zero
            spectrogram = (spectrogram - mean) / std
        
        # --- Step C: Add the channel dimension ---
        # Current shape: (n_mels, n_time_frames)
        # Target shape:  (1, n_mels, n_time_frames)
        # unsqueeze(0) inserts a new axis of size 1 at position 0.
        #
        # After the DataLoader stacks a batch of these, the batch tensor has
        # shape (batch_size, 1, n_mels, n_time_frames), exactly what nn.Conv2d
        # expects as input.
        spectrogram = spectrogram.unsqueeze(0)

        # --- Step D: Wrap the label in a tensor ---
        # dtype=torch.long is required by nn.CrossEntropyLoss for class indices.
        # If you accidentally pass float, PyTorch raises a cryptic error.
        if self.transform:
            spectrogram = self.transform(spectrogram) #Apply augmentation if provided.
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spectrogram, label


# ============================================================================
# STEP 5 — DATALOADER FACTORY
# ============================================================================

def get_dataloaders(processed_dir=PROCESSED_DATA_DIR,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    random_seed=RANDOM_SEED):
    """
    Build the full PyTorch data pipeline end-to-end.

    Orchestrates: file discovery → stratified split → class weights →
    three SpectrogramDatasets → three DataLoaders.

    Parameters
    ----------
    processed_dir : str
        Parent directory of processed .npy files. Default: "data/processed".
    batch_size : int
        Number of samples per batch. Default: 32.
    num_workers : int
        Number of subprocess workers for parallel data loading. Default: 0
        (safe on Windows; bump to 2–4 on Linux/Colab).
    random_seed : int
        Seed for reproducible splitting. Default: 42.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Shuffled, batched iterator over the training split.
    val_loader : torch.utils.data.DataLoader
        Un-shuffled iterator over the validation split.
    test_loader : torch.utils.data.DataLoader
        Un-shuffled iterator over the test split.
    class_weights : np.ndarray, shape (2,)
        Inverse-frequency weights for the loss function, computed from the
        TRAINING set only.
    """

    print("=" * 60)
    print("BUILDING DATA PIPELINE")
    print("=" * 60)

    # --- Step 1: Discover files and assign labels ---
    print("\n[1/4] Gathering file paths & labels...")
    file_paths, labels = gather_file_paths_and_labels(processed_dir=processed_dir)

    if len(file_paths) == 0:
        # Nothing to do — bail out early with a clear error message.
        raise RuntimeError(
            f"No .npy files found under '{processed_dir}'. "
            f"Run src/preprocess.py first to generate spectrograms."
        )

    # --- Step 2: Stratified split into train/val/test ---
    print("\n[2/4] Performing stratified train/val/test split...")
    splits = stratified_split(file_paths, labels, random_seed=random_seed)

    # --- Step 3: Compute class weights from the TRAINING set only ---
    # IMPORTANT: never use val/test labels to compute weights — that would
    # leak information from the held-out sets into the training signal.
    print("\n[3/4] Computing class weights from training set...")
    class_weights = compute_class_weights(splits["train_labels"])

    # --- Step 4: Wrap each split in a SpectrogramDataset and DataLoader ---
    print("\n[4/4] Constructing Datasets and DataLoaders...")

    train_dataset = SpectrogramDataset(splits["train_paths"], splits["train_labels"])
    val_dataset   = SpectrogramDataset(splits["val_paths"],   splits["val_labels"])
    test_dataset  = SpectrogramDataset(splits["test_paths"],  splits["test_labels"])

    # DataLoader wraps a Dataset with batching, shuffling, and parallel I/O.
    #
    # Why shuffle=True for train?
    #   Stochastic Gradient Descent assumes that consecutive batches are
    #   approximately i.i.d. samples from the data distribution. If we fed the
    #   batches in a fixed order (e.g. all normals first, then all abnormals),
    #   the model would see highly correlated batches and training would be
    #   unstable. Shuffling ensures each batch is a random mix of classes.
    #
    # Why shuffle=False for val/test?
    #   Evaluation is order-independent — we compute aggregate metrics over
    #   the whole split. Shuffling would only slow things down and make debugging
    #   harder (error #47 would correspond to a different file each run).
    #
    # pin_memory=True speeds up GPU transfers by allocating the batch in
    # page-locked host memory; it's a free performance win when a GPU is
    # available and a no-op when training on CPU.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"\n  DataLoaders ready:")
    print(f"    train: {len(train_dataset):5d} samples, "
          f"{len(train_loader):4d} batches of size {batch_size}")
    print(f"    val:   {len(val_dataset):5d} samples, "
          f"{len(val_loader):4d} batches of size {batch_size}")
    print(f"    test:  {len(test_dataset):5d} samples, "
          f"{len(test_loader):4d} batches of size {batch_size}")

    print("\n" + "=" * 60)
    print("DATA PIPELINE READY")
    print("=" * 60)

    return train_loader, val_loader, test_loader, class_weights


# ============================================================================
# ENTRY POINT — Quick sanity check when run directly
# ============================================================================

if __name__ == "__main__":
    # When invoked as `python src/dataset.py`, build the pipeline and pull one
    # batch to verify shapes/dtypes. This is a cheap smoke test that catches
    # most integration issues before the model sees any data.

    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    print("\n--- Smoke test: fetching one batch from the training loader ---")
    first_batch = next(iter(train_loader))
    spectrograms, labels = first_batch

    # Expected shapes with defaults:
    #   spectrograms: (batch_size, 1, 128, ~313) — float32
    #   labels:       (batch_size,)              — int64
    print(f"  Spectrogram batch shape: {tuple(spectrograms.shape)}  "
          f"(dtype={spectrograms.dtype})")
    print(f"  Labels batch shape:      {tuple(labels.shape)}          "
          f"(dtype={labels.dtype})")
    print(f"  Labels in this batch:    {labels.tolist()}")
    print(f"  Class weights tensor:    {class_weights.tolist()}")
