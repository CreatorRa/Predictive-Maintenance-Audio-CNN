"""
evaluate.py — Final Test-Set Evaluation + Confusion Matrix Visualization
============================================================
This script is the FIFTH stage of our Predictive Maintenance Audio CNN
project. It loads the best checkpoint produced by src/train.py, runs it
over the held-out test set ONE TIME, and produces:

    1. The four headline metrics for the abnormal class:
         - Accuracy   (overall fraction correct — reported with caveats)
         - Precision  (of clips we flagged as abnormal, how many really were?)
         - Recall     (of clips that really were abnormal, how many did we catch?)
         - F1-Score   (harmonic mean of precision and recall)
    2. A 2x2 Confusion Matrix saved as a heatmap PNG in docs/visualizations/.

WHY A SEPARATE EVALUATION SCRIPT?
    The test set is "spent" the moment we look at it for tuning purposes —
    every decision we make based on test-set performance leaks information
    from the test set into our model selection, inflating the reported
    metrics. Keeping evaluation in its own script (run ONCE at the end of
    the project) preserves the test set as a true unbiased estimate of
    how the model will perform on never-before-seen industrial audio.

WHY A CONFUSION MATRIX?
    The four predictions/outcome combinations carry very different real-world
    costs in predictive maintenance. A confusion matrix lays them all out
    in a single 2x2 grid so we can see exactly where the model fails:

                          PREDICTED
                       Normal    Abnormal
        TRUE  Normal     TN         FP        ← FP = false alarm: wasted inspection
              Abnormal   FN         TP        ← FN = MISSED FAULT: machine breaks down

    For predictive maintenance, FALSE NEGATIVES (FN) are the dangerous error
    mode — an undetected fault means a machine fails unexpectedly, which can
    halt production lines, damage equipment, or harm operators. A FALSE
    POSITIVE (FP) just means we send a technician to check a healthy machine
    — annoying and costly, but not catastrophic. So when reading the matrix,
    we pay extra attention to the FN cell. A model with high accuracy but
    high FN is far worse than a model with slightly lower accuracy but near-
    zero FN.

USAGE:
    From the project root, AFTER running src/train.py (which produces
    models/best_model.pth):
        python src/evaluate.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os                # Filesystem paths + makedirs for the visualization dir
import sys               # Used to make the project root importable

import numpy as np       # Confusion matrix is a 2D NumPy array
import torch             # PyTorch core (loading state_dict, device transfers)

import matplotlib.pyplot as plt   # Figure container for the heatmap PNG
import seaborn as sns             # Pretty heatmap with annotations

# scikit-learn metrics. Importing each one explicitly (vs. `from sklearn
# import metrics`) makes it obvious at the call site exactly which metric
# we're computing — useful when reading the script later.
from sklearn.metrics import (
    accuracy_score,    # Fraction of samples where prediction == label
    precision_score,   # TP / (TP + FP) for the chosen class
    recall_score,      # TP / (TP + FN) for the chosen class
    f1_score,          # Harmonic mean of precision and recall
    confusion_matrix,  # Raw 2x2 (or NxN) count grid
)

# --- Make `from src.xxx import ...` work when running this file directly. ---
# Same trick used in src/train.py: prepend the project root to sys.path so
# the absolute imports below resolve regardless of how the script is launched.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders        # Builds the test DataLoader
from src.model import AudioClassifier           # The CNN architecture
from src.train import (                         # Reuse training-time settings
    get_device,
    BASE_FILTERS,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_PATH,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = ["Normal", "Abnormal"]
# Display labels for the confusion matrix axes. The ORDER must match the
# integer label assignments from src/dataset.py (normal=0, abnormal=1) so
# that index i in this list corresponds to predicted/true label == i.

POSITIVE_LABEL = 1
# The integer label of the "positive" class for binary metrics. We treat
# "abnormal" as the positive class because catching anomalies is the actual
# objective of predictive maintenance — Precision/Recall/F1 are computed
# with this class as the focus.

VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "docs", "visualizations")
# Where confusion-matrix and other report-ready figures live. Created on
# demand if missing.

CONFUSION_MATRIX_PATH = os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")
# Output path for the heatmap PNG. Overwritten every time evaluate.py runs.


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(device, checkpoint_path=CHECKPOINT_PATH,
                       base_filters=BASE_FILTERS):
    """
    Build a fresh AudioClassifier and load the trained weights from disk.

    Parameters
    ----------
    device : torch.device
        Where the model will live (cuda / mps / cpu).
    checkpoint_path : str
        Path to the .pth file produced by src/train.py. Default: the
        CHECKPOINT_PATH constant exported by train.py.
    base_filters : int
        Width of the first conv layer. MUST match whatever value was used
        when the checkpoint was saved — otherwise the state_dict's tensor
        shapes won't line up with the freshly-built model and load_state_dict
        will raise a size-mismatch error. Default: train.py's BASE_FILTERS.

    Returns
    -------
    model : AudioClassifier
        The model with trained weights loaded, moved to `device`, and
        switched into evaluation mode.
    """

    # Fail loudly and early if the checkpoint isn't there. Without this
    # guard, torch.load would raise a confusing FileNotFoundError mid-run.
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint_path}'. "
            f"Run src/train.py first to train and save the best model."
        )

    # Build the SAME architecture used during training. The state_dict
    # only contains parameter values keyed by layer name — we still need
    # the matching nn.Module to load them into.
    model = AudioClassifier(base_filters=base_filters)

    # torch.load deserializes the state_dict from disk. We pass
    # map_location=device so the tensors land on the correct device
    # immediately — this is the right pattern for cross-device checkpoints
    # (e.g., trained on Colab GPU, evaluated on local CPU).
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move the module's parameters/buffers to `device` and switch into
    # eval mode. eval() disables dropout and tells BatchNorm to use its
    # accumulated running statistics — both essential for deterministic,
    # apples-to-apples test-time inference.
    model.to(device)
    model.eval()

    return model


# ============================================================================
# INFERENCE LOOP
# ============================================================================

def run_inference(model, loader, device):
    """
    Run the model over every batch in `loader` and collect the predictions.

    Parameters
    ----------
    model : nn.Module
        AudioClassifier already in eval mode and on `device`.
    loader : DataLoader
        Test DataLoader (un-shuffled — we want deterministic ordering for
        any per-sample debugging later).
    device : torch.device
        Compute device.

    Returns
    -------
    y_true : np.ndarray, shape (N,)
        Integer ground-truth labels for every sample in the test set.
    y_pred : np.ndarray, shape (N,)
        Integer predicted labels (argmax over the 2 logits).
    """

    all_labels = []
    all_preds = []

    # torch.no_grad() disables autograd's tape recording for everything
    # inside the block — saves memory and CPU/GPU cycles. We never need
    # gradients during inference.
    with torch.no_grad():
        for spectrograms, labels in loader:
            # non_blocking=True can overlap the host→device copy with
            # compute when the DataLoader uses pin_memory=True (which our
            # dataset.py DOES set). It's a no-op on CPU.
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass — returns raw logits of shape [B, num_classes].
            logits = model(spectrograms)

            # argmax(dim=1) picks the class with the highest logit per
            # sample. We don't need softmax probabilities here; argmax is
            # invariant to monotonic transforms.
            preds = logits.argmax(dim=1)

            # Move to CPU and convert to plain Python lists. sklearn lives
            # in NumPy land and won't accept GPU tensors. Using .tolist()
            # avoids holding many small tensors in memory across iterations.
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # Convert once at the end to NumPy arrays — every sklearn metric
    # function takes them happily and the conversion is cheap.
    return np.array(all_labels), np.array(all_preds)


# ============================================================================
# METRICS
# ============================================================================

def compute_and_print_metrics(y_true, y_pred, positive_label=POSITIVE_LABEL):
    """
    Compute and print the four headline metrics for the test set.

    Precision/Recall/F1 are computed BINARY-style with `positive_label`
    treated as the class of interest. For us that's "abnormal" (label 1) —
    catching anomalies is what predictive maintenance is for.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth labels.
    y_pred : np.ndarray, shape (N,)
        Predicted labels.
    positive_label : int
        Which integer label counts as the "positive" class for binary
        precision/recall/F1. Default: 1 (abnormal).

    Returns
    -------
    metrics : dict
        Keys: "accuracy", "precision", "recall", "f1". Returned so callers
        can log or re-display the numbers without re-computing them.
    """

    # accuracy_score is class-agnostic — it just counts matches. Reported
    # for completeness (and as a sanity-check against the validation
    # accuracies seen during training) but NOT the metric of record.
    accuracy = accuracy_score(y_true, y_pred)

    # zero_division=0 → if the denominator (TP+FP for precision, TP+FN for
    # recall) is zero, return 0.0 instead of raising a warning. Correct
    # interpretation: a model that produced no positive predictions has
    # undefined precision; we score it 0 instead of NaN-ing the report.
    precision = precision_score(y_true, y_pred, pos_label=positive_label,
                                zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label,
                          zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label,
                  zero_division=0)

    # Pretty-print so the user can paste the block straight into a report.
    print()
    print("=" * 60)
    print("FINAL TEST-SET METRICS")
    print("=" * 60)
    print(f"  Total test samples: {len(y_true)}")
    print()
    print(f"  Accuracy:  {accuracy:.4f}    (fraction of all samples correct)")
    print(f"  Precision: {precision:.4f}    (of clips flagged abnormal, how "
          f"many really were)")
    print(f"  Recall:    {recall:.4f}    (of true abnormal clips, how many "
          f"we caught)")
    print(f"  F1-Score:  {f1:.4f}    (harmonic mean of precision & recall)")
    print()
    print("  ⤷ For predictive maintenance, RECALL is the metric to defend:")
    print("    a missed abnormal clip = a real machine fault going undetected.")
    print("=" * 60)

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_and_save_confusion_matrix(y_true, y_pred,
                                    output_path=CONFUSION_MATRIX_PATH,
                                    class_names=CLASS_NAMES):
    """
    Build a 2x2 confusion matrix from the test predictions and save it
    as an annotated seaborn heatmap PNG.

    LAYOUT (sklearn convention with labels=[0, 1]):
                            PREDICTED
                         Normal    Abnormal
            TRUE Normal    TN         FP
                 Abnormal  FN         TP

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth labels.
    y_pred : np.ndarray, shape (N,)
        Predicted labels.
    output_path : str
        Where to write the PNG. Parent directory is created if missing.
        Default: docs/visualizations/confusion_matrix.png.
    class_names : list of str
        Display labels for the axes, ordered by integer class index.
        Default: ["Normal", "Abnormal"].

    Returns
    -------
    cm : np.ndarray, shape (2, 2)
        The raw confusion matrix counts. Returned so the caller can
        inspect/log specific cells (e.g., the FN count).
    """

    # Pin the label order to [0, 1] so the matrix is always laid out
    # Normal-then-Abnormal regardless of which classes happened to appear
    # in y_true. Without this argument sklearn would infer the order from
    # the data and could produce inconsistent layouts across runs.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Make sure the output directory exists. exist_ok=True is idempotent.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build the figure. (8, 6) inches is large enough for the annotations
    # to be readable at typical report-PDF zoom levels.
    plt.figure(figsize=(8, 6))

    # seaborn.heatmap renders the 2D NumPy array as a colored grid.
    #   annot=True       → write the raw count in each cell.
    #   fmt='d'          → format the annotation as a decimal integer
    #                      (otherwise seaborn would print them as floats).
    #   cmap='Blues'     → colorblind-friendly sequential colormap; darker
    #                      = larger count, which makes the diagonal
    #                      (correct predictions) visually pop.
    #   xticklabels/yticklabels → use the human-readable class names from
    #                      class_names, in the same order as labels=[0,1]
    #                      above.
    #   cbar=True        → keep the colorbar; useful when comparing two
    #                      confusion matrices side by side.
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )

    # Axis labels and title — clarity matters when this PNG ends up in a
    # course report next to four other figures.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix — Test Set")
    # tight_layout prevents axis labels from being clipped on save.
    plt.tight_layout()

    # bbox_inches='tight' = save the smallest bounding box that contains
    # everything (no extra whitespace). dpi=150 gives a crisp render at
    # report-print resolution without bloating the file size.
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Free the figure's memory. Important if this script ever gets called
    # in a loop (e.g., evaluating multiple checkpoints).
    plt.close()

    # Also dump the matrix to stdout in plain text — handy when the user
    # runs the script over SSH without a way to view the PNG immediately.
    print()
    print("Confusion Matrix (rows = True, cols = Predicted):")
    print(f"                  Pred Normal   Pred Abnormal")
    print(f"  True Normal   |  {cm[0, 0]:>10d}    {cm[0, 1]:>13d}    "
          f"← FP={cm[0, 1]} (false alarms)")
    print(f"  True Abnormal |  {cm[1, 0]:>10d}    {cm[1, 1]:>13d}    "
          f"← FN={cm[1, 0]} (MISSED FAULTS)")
    print()
    print(f"  Heatmap saved: {output_path}")

    return cm


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """
    Wire everything together: device → test loader → model checkpoint →
    inference → metrics → confusion matrix.

    Runs when you launch `python src/evaluate.py` from the project root.
    """

    print("=" * 60)
    print("EVALUATION — AudioClassifier on Held-Out Test Set")
    print("=" * 60)

    # ---- 1. Device ----
    print("\n[1/4] Selecting compute device...")
    device = get_device()

    # ---- 2. Data ----
    # We only need the test loader. The train and val loaders are still
    # built (the function builds all three together so the splits stay
    # reproducible), but we deliberately discard them so we never
    # accidentally let test-time decisions leak back into training.
    print("\n[2/4] Building test DataLoader...")
    _train, _val, test_loader, _weights = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # ---- 3. Model ----
    print("\n[3/4] Loading trained model from checkpoint...")
    model = load_trained_model(device)
    print(f"  Loaded weights from: {CHECKPOINT_PATH}")

    # ---- 4. Inference + metrics + visualization ----
    print("\n[4/4] Running inference on the test set...")
    y_true, y_pred = run_inference(model, test_loader, device)

    compute_and_print_metrics(y_true, y_pred)
    plot_and_save_confusion_matrix(y_true, y_pred)

    print("\nEvaluation complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================
# Guarded so importing this module (e.g., from a notebook or another script)
# doesn't trigger a full evaluation run.

if __name__ == "__main__":
    main()
