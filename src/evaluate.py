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

import argparse          # CLI args — replaces shared constants imported from train.py
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
    roc_curve,         # (FPR, TPR, thresholds) sweep across all decision thresholds
    auc,               # Area-under-curve given (x, y) — used with roc_curve outputs
    roc_auc_score,     # Convenience: ROC-AUC directly from (y_true, y_probs)
)

# --- Make `from src.xxx import ...` work when running this file directly. ---
# Same trick used in src/train.py: prepend the project root to sys.path so
# the absolute imports below resolve regardless of how the script is launched.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders        # Builds the test DataLoader
from src.model import AudioClassifier           # The CNN architecture
from src.train import (                         # Reuse only the truly-shared bits
    get_device,
    NUM_WORKERS,
    CHECKPOINT_DIR,
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
# Where confusion-matrix and ROC PNGs live. Created on demand if missing.

EXPERIMENT_TRACKING_CSV = os.path.join(PROJECT_ROOT, "docs", "experiment_tracking.csv")
# Master HPO log. Each call to evaluate.py appends one row (header on first write).


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """
    CLI args mirror those of train.py so a single (lr, batch_size, base_filters,
    run_name) tuple propagates through the entire HPO pipeline:
        train.py    → produces best_model_{run_name}.pth
        evaluate.py → loads best_model_{run_name}.pth, writes per-run figures
                      and APPENDS a row to docs/experiment_tracking.csv
        explain.py  → loads best_model_{run_name}.pth, writes per-run heatmaps

    base_filters MUST match the value used when the checkpoint was saved.
    Mismatches cause load_state_dict to raise a shape error — that's a feature,
    not a bug, because it loudly catches mis-tagged runs.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AudioClassifier checkpoint."
    )
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate the model was trained with. "
                             "Recorded in the CSV; not used for inference.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for the test DataLoader.")
    parser.add_argument("--base_filters", type=int, default=16,
                        help="Conv-stack width — MUST match training value.")
    parser.add_argument("--run_name", type=str, default="default_run",
                        help="Run tag — selects which checkpoint to load and "
                             "is embedded in every output filename.")
    return parser.parse_args()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(device, checkpoint_path, base_filters):
    """
    Build a fresh AudioClassifier and load the trained weights from disk.

    Parameters
    ----------
    device : torch.device
        Where the model will live (cuda / mps / cpu).
    checkpoint_path : str
        Path to the .pth file produced by src/train.py. Caller builds this
        path from the run_name CLI arg so HPO runs stay distinct.
    base_filters : int
        Width of the first conv layer. MUST match whatever value was used
        when the checkpoint was saved — otherwise the state_dict's tensor
        shapes won't line up with the freshly-built model and load_state_dict
        will raise a size-mismatch error.

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
    Run the model over every batch in `loader` and collect both the hard
    class predictions AND the soft positive-class probabilities.

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
        Integer predicted labels (argmax over the 2 logits — equivalent to
        thresholding the positive-class probability at 0.5).
    y_probs : np.ndarray, shape (N,)
        Predicted probability of the POSITIVE (abnormal) class for every
        sample, in [0, 1]. Required for ROC-AUC and any threshold-sweep
        analysis (precision-recall curves, custom alarm thresholds, etc.).
    """

    all_labels = []
    all_preds = []
    all_probs = []   # Positive-class probabilities for ROC-AUC.

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

            # ---- Soft probabilities via softmax ----
            # The model returns RAW logits (we deliberately left the softmax
            # out of the architecture so that nn.CrossEntropyLoss could
            # apply log-softmax internally during training). For ROC-AUC
            # we now need actual probabilities — apply softmax over the
            # class dimension to get a [B, 2] tensor whose rows sum to 1.
            #
            # We then slice column [:, 1] to extract only the POSITIVE
            # class (abnormal) probability. ROC analysis is a one-versus-
            # rest concept: we treat the model's confidence in "abnormal"
            # as a continuous score and sweep a threshold across it.
            probs = torch.softmax(logits, dim=1)
            pos_probs = probs[:, 1]

            # argmax(dim=1) picks the class with the highest logit per
            # sample. Equivalent to "is positive-class probability >= 0.5?"
            # because softmax preserves the argmax. We keep this for the
            # confusion matrix and the precision/recall/F1 metrics, which
            # all need hard class labels.
            preds = logits.argmax(dim=1)

            # Move to CPU and convert to plain Python lists. sklearn lives
            # in NumPy land and won't accept GPU tensors. Using .tolist()
            # avoids holding many small tensors in memory across iterations.
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(pos_probs.cpu().tolist())

    # Convert once at the end to NumPy arrays — every sklearn metric
    # function takes them happily and the conversion is cheap.
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ============================================================================
# METRICS
# ============================================================================

def compute_and_print_metrics(y_true, y_pred, y_probs,
                              positive_label=POSITIVE_LABEL):
    """
    Compute and print the five headline metrics for the test set.

    Precision/Recall/F1 are computed BINARY-style with `positive_label`
    treated as the class of interest. For us that's "abnormal" (label 1) —
    catching anomalies is what predictive maintenance is for.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth labels.
    y_pred : np.ndarray, shape (N,)
        Predicted labels (argmax over logits).
    y_probs : np.ndarray, shape (N,)
        Predicted probability of the positive class for every sample.
        Required for the threshold-independent ROC-AUC metric — the
        precision/recall/F1 above all assume a fixed 0.5 threshold, but
        ROC-AUC summarizes the model's ranking quality across EVERY
        possible threshold.
    positive_label : int
        Which integer label counts as the "positive" class for binary
        precision/recall/F1. Default: 1 (abnormal).

    Returns
    -------
    metrics : dict
        Keys: "accuracy", "precision", "recall", "f1", "roc_auc". Returned
        so callers can log or re-display the numbers without re-computing
        them.
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

    # ---- ROC-AUC: a threshold-independent summary of ranking quality ----
    # roc_auc_score takes the TRUE labels and the POSITIVE-CLASS PROBABILITIES
    # (NOT the hard 0/1 predictions) and returns the area under the ROC curve.
    #
    # Interpretation (the "Mann-Whitney" framing):
    #   AUC = P(model assigns a higher abnormal-probability to a randomly
    #         chosen abnormal sample than to a randomly chosen normal one)
    #
    # Reference points:
    #   AUC = 1.0 → perfect ranking (every abnormal sample scores higher
    #               than every normal one).
    #   AUC = 0.5 → no better than random coin-flip.
    #   AUC < 0.5 → systematically WORSE than random — usually a sign that
    #               the labels are flipped.
    roc_auc = roc_auc_score(y_true, y_probs)

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
    print(f"  ROC-AUC:   {roc_auc:.4f}    (ranking quality across ALL "
          f"thresholds)")
    print()
    print("  ⤷ For predictive maintenance, RECALL is the metric to defend:")
    print("    a missed abnormal clip = a real machine fault going undetected.")
    print("  ⤷ ROC-AUC is threshold-independent — useful when we want to")
    print("    tune the alarm sensitivity AFTER training (see roc_curve.png).")
    print("=" * 60)

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
    }


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_and_save_confusion_matrix(y_true, y_pred,
                                    output_path,
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
        Caller is responsible for embedding the run_name in this path so
        HPO experiments do not overwrite each other.
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
# ROC CURVE VISUALIZATION
# ============================================================================

def plot_and_save_roc_curve(y_true, y_probs, output_path):
    """
    Compute the ROC curve from positive-class probabilities and save it as
    a PNG plot annotated with the AUC score.

    WHY EVALUATE ACROSS ALL THRESHOLDS (NOT JUST ARGMAX)?
        The hard predictions used for accuracy/precision/recall/F1 implicitly
        threshold the model's probability output at 0.5 — every sample whose
        positive-class probability is at or above 0.5 is flagged abnormal,
        every other sample is flagged normal. That single threshold is rarely
        the right operational choice for predictive maintenance:

          - If missing a fault is catastrophic (machine failure costs $$$$),
            an operator wants to LOWER the threshold so MORE samples get
            flagged as abnormal — recall goes up, precision goes down,
            more false alarms but fewer missed faults.
          - If false alarms are extremely expensive (sending a tech to every
            flagged machine), they want to RAISE the threshold so ONLY the
            most confident predictions trigger — precision goes up, recall
            goes down.

        The ROC curve plots that entire trade-off explicitly. For every
        possible threshold t in [0, 1], it computes:
            TPR(t) = Recall   = TP / (TP + FN)
                                "fraction of real anomalies we catch"
            FPR(t) = Fall-out = FP / (FP + TN)
                                "fraction of healthy machines we falsely alarm"
        and plots TPR vs. FPR as the threshold sweeps from 1.0 down to 0.0.

        - Top-left corner (FPR=0, TPR=1) = perfect classifier.
        - Diagonal y=x line             = random guessing baseline.
        - Curve hugging the top-left    = strong model.
        - Curve along the diagonal      = useless model.

        AUC (Area Under the ROC Curve) condenses the entire curve to a
        single threshold-independent number. AUC=1.0 is perfect; AUC=0.5 is
        no better than coin flips. Because AUC doesn't depend on a chosen
        threshold, it's the most honest single-number summary of model
        quality on imbalanced data — and it lets stakeholders defend the
        model regardless of where they ultimately set the alarm sensitivity.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth binary labels (0 = normal, 1 = abnormal).
    y_probs : np.ndarray, shape (N,)
        Predicted probability of the POSITIVE (abnormal) class for every
        sample. NOT hard predictions — sklearn needs continuous scores so
        it can sweep the threshold.
    output_path : str
        Where to save the PNG. Parent directory is created if missing.
        Caller embeds the run_name so HPO experiments stay distinct.

    Returns
    -------
    roc_auc : float
        The computed ROC-AUC value. Returned so the caller can log it
        without re-computing.
    """

    # roc_curve returns three parallel arrays of length n_thresholds:
    #   fpr        - False Positive Rate at each threshold
    #   tpr        - True  Positive Rate at each threshold
    #   thresholds - the corresponding probability cutoffs (sorted high→low)
    # sklearn picks all "interesting" thresholds (one per unique prob value
    # plus an extra at the top) so the curve is exact, not subsampled.
    # The third return (per-threshold cutoffs) is unused here — we plot
    # the curve in (FPR, TPR) space rather than annotating threshold values.
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)

    # auc(fpr, tpr) integrates the curve via the trapezoid rule. Identical
    # to roc_auc_score(y_true, y_probs) for binary problems, but we use
    # auc() here so the area is computed from EXACTLY the points we'll
    # also plot — guaranteeing the legend value matches the visible curve.
    roc_auc = auc(fpr, tpr)

    # Make sure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build the figure. Square-ish aspect for an ROC plot is conventional
    # because the axes share a [0, 1] range.
    plt.figure(figsize=(7, 6))

    # ---- Main curve: model performance across all thresholds ----
    # color='C0' picks the first matplotlib default color (blue). lw=2 makes
    # the curve clearly visible at report-print resolution.
    plt.plot(fpr, tpr, color='C0', lw=2,
             label=f"Model ROC (AUC = {roc_auc:.4f})")

    # ---- Diagonal: random-classifier baseline ----
    # A model that assigns probabilities uniformly at random produces a
    # straight diagonal from (0,0) to (1,1). Plotting it gives the reader
    # an instant visual reference: if our curve is hugging the diagonal,
    # the model has learned nothing.
    # linestyle='--' (dashed) and a muted gray make this read as "reference"
    # rather than competing with the actual model curve.
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label="Random guessing (AUC = 0.5)")

    # ---- Axis bounds, labels, title, legend ----
    # Slightly extend the upper x bound so the curve doesn't bump against
    # the right edge of the plot.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate  (FP / (FP + TN))")
    plt.ylabel("True Positive Rate  (Recall = TP / (TP + FN))")
    plt.title("ROC Curve — Test Set")
    # loc='lower right' is the conventional placement for ROC legends —
    # keeps it out of the high-TPR / low-FPR region where the curve lives
    # on a good model.
    plt.legend(loc="lower right")

    # Light grid helps readers eyeball specific (FPR, TPR) operating points
    # if they want to choose a custom alarm threshold from the figure.
    plt.grid(alpha=0.3)

    # tight_layout prevents axis labels from being clipped on save.
    plt.tight_layout()

    # bbox_inches='tight' = save the smallest bounding box that contains
    # everything (no extra whitespace). dpi=150 gives a crisp render at
    # report-print resolution without bloating the file size.
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Free the figure's memory.
    plt.close()

    print()
    print(f"ROC curve saved: {output_path}")
    print(f"  ROC-AUC = {roc_auc:.4f}  "
          f"(0.5 = random, 1.0 = perfect)")

    return roc_auc


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """
    Wire everything together: args → device → test loader → checkpoint →
    inference → metrics → confusion matrix → ROC curve → CSV row.
    """

    # ---- 0. Parse CLI args ----
    args = parse_args()

    # Per-run paths — every output filename is namespaced by run_name so
    # HPO experiments produce a parallel set of figures we can compare later.
    checkpoint_path        = os.path.join(CHECKPOINT_DIR, f"best_model_{args.run_name}.pth")
    confusion_matrix_path  = os.path.join(VISUALIZATION_DIR, f"confusion_matrix_{args.run_name}.png")
    roc_curve_path         = os.path.join(VISUALIZATION_DIR, f"roc_curve_{args.run_name}.png")

    print("=" * 60)
    print("EVALUATION — AudioClassifier on Held-Out Test Set")
    print(f"  Run name:    {args.run_name}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print("=" * 60)

    # ---- 1. Device ----
    print("\n[1/5] Selecting compute device...")
    device = get_device()

    # ---- 2. Data ----
    # We only need the test loader. The train and val loaders are still built
    # (the function builds all three together so the splits stay reproducible),
    # but we deliberately discard them so we never let test-time decisions
    # leak back into training.
    print("\n[2/5] Building test DataLoader...")
    _train, _val, test_loader, _weights = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
    )

    # ---- 3. Model ----
    print("\n[3/5] Loading trained model from checkpoint...")
    model = load_trained_model(device, checkpoint_path, args.base_filters)
    print(f"  Loaded weights from: {checkpoint_path}")

    # ---- 4. Inference + metrics + visualization ----
    # run_inference returns three arrays — true labels, hard predictions
    # (for discrete metrics + confusion matrix), and positive-class
    # probabilities (for the threshold-independent ROC-AUC + curve).
    print("\n[4/5] Running inference on the test set...")
    y_true, y_pred, y_probs = run_inference(model, test_loader, device)

    metrics = compute_and_print_metrics(y_true, y_pred, y_probs)
    plot_and_save_confusion_matrix(y_true, y_pred, output_path=confusion_matrix_path)
    plot_and_save_roc_curve(y_true, y_probs, output_path=roc_curve_path)

    # ---- 5. Append this run's metrics to the master CSV (Task 2) ----
    print("\n[5/5] Logging results to experiment-tracking CSV...")
    append_results_to_csv(args, metrics, EXPERIMENT_TRACKING_CSV)

    print("\nEvaluation complete.")


# ============================================================================
# EXPERIMENT TRACKING — Append one CSV row per HPO run (Task 2)
# ============================================================================

def append_results_to_csv(args, metrics, csv_path):
    """
    Append a single line summarizing this run to docs/experiment_tracking.csv.

    WHY A FLAT CSV?
        It's the simplest format pandas / Excel / plain editors all read.
        Across a 12-cell grid search we end up with 12 rows — perfectly
        analyzable in a spreadsheet without any infra. tune.py can also tail
        the file mid-run to monitor progress.

    SCHEMA (one row per training run):
        run_name, lr, batch_size, base_filters,
        accuracy, precision, recall, f1, roc_auc

    The first three identify the experiment; the last five record outcomes.
    Sorting / filtering by F1 or ROC-AUC after the fact is then a one-liner.

    Parameters
    ----------
    args : argparse.Namespace
        Carries lr, batch_size, base_filters, run_name from parse_args().
    metrics : dict
        Output of compute_and_print_metrics — has keys accuracy, precision,
        recall, f1, roc_auc.
    csv_path : str
        Where to write. Parent dir is created on demand. Header is emitted
        only when the file does not yet exist.
    """
    import csv  # Local import — only this function needs it.

    # Make sure docs/ exists before opening the file in append mode.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Detect first-time write so we know whether to emit the header.
    # We check existence BEFORE opening so the open() itself doesn't create
    # an empty file and break the test on a retry.
    file_is_new = not os.path.isfile(csv_path)

    # newline='' is the official csv-module recommendation on Windows —
    # without it, the writer's '\r\n' line terminator gets translated by the
    # underlying text-mode handle, producing blank lines between rows.
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Emit header only on first creation. If the user wipes the CSV
        # between runs and re-runs evaluate.py, the header is restored
        # automatically.
        if file_is_new:
            writer.writerow([
                "run_name", "lr", "batch_size", "base_filters",
                "accuracy", "precision", "recall", "f1", "roc_auc",
            ])

        writer.writerow([
            args.run_name,
            args.lr,
            args.batch_size,
            args.base_filters,
            f"{metrics['accuracy']:.6f}",
            f"{metrics['precision']:.6f}",
            f"{metrics['recall']:.6f}",
            f"{metrics['f1']:.6f}",
            f"{metrics['roc_auc']:.6f}",
        ])

    print(f"  Appended row for run '{args.run_name}' → {csv_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================
# Guarded so importing this module (e.g., from a notebook or another script)
# doesn't trigger a full evaluation run.

if __name__ == "__main__":
    main()
