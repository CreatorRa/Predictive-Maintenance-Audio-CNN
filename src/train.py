"""
train.py — End-to-End Training Loop for AudioClassifier
============================================================
This script is the FOURTH stage of our Predictive Maintenance Audio CNN
project. It wires together the data pipeline (src/dataset.py) and the model
architecture (src/model.py), then runs a standard PyTorch training loop with
class-weighted cross-entropy loss and best-model checkpointing.

WHAT THIS FILE DOES (high level):
    1. Detects the best available compute device (CUDA → MPS → CPU).
    2. Builds train/val/test DataLoaders and pulls the per-class weights
       computed from the training set.
    3. Instantiates an AudioClassifier and moves it to the chosen device.
    4. Defines the loss function (weighted CrossEntropyLoss) and optimizer (Adam).
    5. For NUM_EPOCHS:
         - Runs a TRAINING phase (gradients on, dropout/BatchNorm in train mode)
         - Runs a VALIDATION phase (gradients off, dropout disabled, BN frozen)
         - Logs per-epoch loss and accuracy for both phases
         - If validation loss improved, saves the current state_dict to
           models/best_model.pth
    6. After each epoch, hands the validation loss to a ReduceLROnPlateau
       scheduler, which halves the learning rate whenever progress stalls.
    7. Tracks consecutive epochs without validation-loss improvement and
       triggers EARLY STOPPING once the streak reaches EARLY_STOPPING_PATIENCE,
       breaking out of the loop early.
    8. Reports the best validation loss observed and where its checkpoint lives.

WHY THIS DESIGN?
    - Class-weighted loss: MIMII has many more "normal" than "abnormal"
      recordings; without weighting, a model that always predicts "normal"
      could trivially score >95% accuracy yet be useless for anomaly
      detection. Weighting the loss by inverse class frequency forces the
      model to pay proportionally more attention to the minority class.
    - Adam optimizer: combines momentum and per-parameter adaptive learning
      rates. It's a strong default that works well out-of-the-box for most
      CNNs, sparing us from hand-tuning learning rate schedules early on.
    - ReduceLROnPlateau scheduler: when validation loss stops improving for
      a few epochs, we halve the learning rate. Big steps early on cover
      the loss landscape quickly; smaller steps later let us settle into a
      narrow minimum that big steps would overshoot. This is one of the
      cheapest ways to squeeze extra accuracy out of a model that has
      "almost converged."
    - Early stopping: if validation loss hasn't improved for
      EARLY_STOPPING_PATIENCE epochs, we stop training. Two wins:
        1. Saves Colab compute (and our GPU credits) — we don't waste
           epochs that aren't going to help.
        2. Prevents overfitting — past the convergence point, validation
           loss starts climbing again as the model memorizes training
           noise. Early stopping freezes us at the best generalization
           point we ever saw.
    - Checkpoint by *validation loss* (not training loss): training loss
      decreases monotonically as the model overfits, so it's a poor signal
      of generalization. Validation loss is computed on data the model
      never sees during weight updates, so it's a faithful proxy for how
      the model will perform on the held-out test set.

USAGE:
    From the project root:
        python src/train.py

    Hyperparameters live as constants at the top of this file — edit those
    to experiment with learning rate, epoch count, batch size, or model size.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse           # CLI arg parsing — replaces hardcoded HPO constants.
import os                # Filesystem paths + makedirs for the checkpoint dir
import sys               # Used to make the project root importable
import time              # Per-epoch timing for the progress log

import torch             # PyTorch core
import torch.nn as nn    # nn.CrossEntropyLoss
import torch.optim as optim  # Adam optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau  # LR scheduler

from sklearn.metrics import f1_score  # Imbalance-aware evaluation metric

# --- Make `from src.xxx import ...` work when running this file directly. ---
# When you launch with `python src/train.py`, Python adds src/ (the script's
# directory) to sys.path, NOT the project root. That breaks imports like
# `from src.dataset import ...`. We patch this by inserting the project root
# (the parent of src/) at position 0 so absolute imports resolve cleanly
# regardless of how the script is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders   # Builds train/val/test DataLoaders + weights
from src.model import AudioClassifier      # Our 4-block CNN with GAP head


# ============================================================================
# FIXED HYPERPARAMETERS — These stay constant across HPO runs.
# ============================================================================
# We deliberately do NOT expose these as CLI args because the HPO grid only
# tunes learning rate, batch size, and base filters. Holding everything else
# fixed isolates the effect of the swept parameters — a clean controlled
# experiment.

NUM_EPOCHS = 20
# One "epoch" = one full pass over the training set. 20 is a reasonable
# starting point — enough for the model to converge on a small dataset, not
# so many that we waste compute. Early stopping will trim this further when
# validation loss plateaus.

NUM_WORKERS = 0
# DataLoader worker processes. Keep 0 on Windows to avoid spawn-related
# headaches; 2–4 on Linux/Colab makes I/O notably faster.

EARLY_STOPPING_PATIENCE = 7
# How many consecutive epochs we tolerate without a validation-loss
# improvement before halting training. Tuning rule of thumb:
#   - Set noticeably larger than the LR scheduler's patience (3) so the
#     scheduler gets at least one chance to drop the LR and rescue
#     progress before early stopping fires.
#   - Too small → we kill runs that would have recovered.
#   - Too large → we waste epochs once the model has plateaued and
#     started overfitting.
# 7 is a balanced default for our 20-epoch budget.

LR_SCHEDULER_FACTOR = 0.5
# When the scheduler triggers, the new LR = current LR * factor. 0.5
# (halving) is a common, gentle choice — aggressive enough to make a
# noticeable difference, conservative enough not to stall training.

LR_SCHEDULER_PATIENCE = 3
# How many epochs the scheduler waits with no validation-loss
# improvement before halving the LR. Must be smaller than
# EARLY_STOPPING_PATIENCE so the LR drop has a chance to help before
# early stopping kicks in.

# --- Output paths ---
# CHECKPOINT_DIR is shared across runs; the FILENAME inside it is built
# inside main() from --run_name so HPO experiments don't clobber each other.
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models")


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """
    Build the CLI arg parser for train.py.

    WHY ARGPARSE INSTEAD OF HARDCODED CONSTANTS?
        Task: support automated grid search via tune.py. The orchestrator
        loops over (lr, batch_size, base_filters) combinations and shells
        out to this script. Argparse is the cleanest contract between the
        orchestrator and this script — no environment variables, no editing
        constants between runs, no risk of a stale value carrying over.

    The defaults match the previous hardcoded constants exactly so the bare
    command `python src/train.py` reproduces the pre-HPO behaviour.
    """
    parser = argparse.ArgumentParser(
        description="Train AudioClassifier with configurable hyperparameters."
    )
    # --lr: Adam step size. Grid search will sweep this on a log scale.
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam. Default: 1e-3.")
    # --batch_size: forwarded to get_dataloaders. Bigger = more stable
    # gradients but more VRAM. Sweep candidates: 16, 32, 64.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size. Default: 32.")
    # --base_filters: width of the first conv layer. Doubles every block.
    # Sweep candidates: 16 (small), 32 (medium), 64 (large).
    parser.add_argument("--base_filters", type=int, default=16,
                        help="Width of first conv layer. Default: 16.")
    # --run_name: human-readable tag for THIS specific experiment. Used to
    # name the checkpoint file so different HPO runs don't overwrite each
    # other's weights. tune.py auto-generates one per combination.
    parser.add_argument("--run_name", type=str, default="default_run",
                        help="Unique tag for this run. Used in checkpoint "
                             "and visualization filenames so HPO runs do "
                             "not overwrite each other. Default: default_run.")
    return parser.parse_args()


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def get_device():
    """
    Pick the best available compute device for training.

    Preference order:
        1. CUDA  — NVIDIA GPUs (Colab, most workstations).
        2. MPS   — Apple Silicon (M1/M2/M3 Macs) via Metal Performance Shaders.
        3. CPU   — Fallback. Slow for training but always available.

    Returns
    -------
    torch.device
        The selected device. Pass to `.to(device)` on tensors and modules.
    """
    if torch.cuda.is_available():
        # CUDA is by far the fastest option when present. We also print the
        # GPU name so the user can verify which card actually got picked
        # (useful on multi-GPU workstations).
        device = torch.device("cuda")
        print(f"  Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS is Apple's GPU backend. Significantly faster than CPU on M-series
        # Macs but with occasional op-coverage gaps versus CUDA.
        device = torch.device("mps")
        print(f"  Using device: MPS (Apple Silicon)")
    else:
        # CPU fallback. Training a CNN here is slow — fine for the local
        # micro-batch smoke test, not for the full Colab run.
        device = torch.device("cpu")
        print(f"  Using device: CPU (no GPU detected — training will be slow)")
    return device


# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass over the training set, updating model weights.

    Parameters
    ----------
    model : nn.Module
        The AudioClassifier instance (already on `device`).
    loader : DataLoader
        Yields (spectrograms, labels) batches from the training set.
    criterion : nn.Module
        Loss function — here, weighted nn.CrossEntropyLoss.
    optimizer : torch.optim.Optimizer
        Updates model parameters from the gradients computed by loss.backward().
    device : torch.device
        Where to send each batch (GPU/MPS/CPU).

    Returns
    -------
    avg_loss : float
        Mean training loss across all batches in this epoch.
    accuracy : float
        Fraction of correct predictions across all training samples.
    """

    # model.train() flips the model into TRAINING mode. This affects layers
    # that behave differently in train vs. eval:
    #   - Dropout: actually drops activations (vs. passing through in eval).
    #   - BatchNorm: uses the *current batch's* statistics for normalization
    #     and updates the running mean/variance buffers (vs. using the
    #     stored running stats in eval).
    # Forgetting this call is a classic bug that causes silent under-training.
    model.train()

    running_loss = 0.0       # Sum of (loss * batch_size) — divided at the end.
    correct = 0              # Count of correctly classified samples.
    total = 0                # Count of samples seen this epoch.

    for spectrograms, labels in loader:
        # Move the batch onto the compute device. non_blocking=True lets
        # PyTorch overlap the host→device transfer with GPU compute when
        # the DataLoader uses pin_memory=True (which it does — see dataset.py).
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---- 1. Zero the parameter gradients ----
        # PyTorch ACCUMULATES gradients into .grad attributes by default
        # (so you can split a logical batch across multiple forward passes
        # if it doesn't fit in memory). For our standard one-batch-at-a-time
        # training, we must clear those accumulated gradients before each
        # backward pass — otherwise gradient updates from previous batches
        # would corrupt this batch's update.
        # set_to_none=True is slightly faster than zeroing in place.
        optimizer.zero_grad(set_to_none=True)

        # ---- 2. Forward pass ----
        # Computes raw class logits of shape [B, num_classes].
        logits = model(spectrograms)

        # ---- 3. Compute the loss ----
        # CrossEntropyLoss internally applies log-softmax to the logits and
        # compares against the integer class indices. With our class weights
        # baked in, mistakes on the minority class are penalized more heavily.
        loss = criterion(logits, labels)

        # ---- 4. Backward pass ----
        # Computes d(loss)/d(parameter) for every parameter in the model and
        # stores the result in each parameter's .grad attribute. This is
        # automatic differentiation — the same operations that were recorded
        # during the forward pass are now traversed in reverse.
        loss.backward()

        # ---- 5. Optimizer step ----
        # Reads each parameter's .grad and applies the Adam update rule
        # (gradient + momentum + per-parameter adaptive scaling), modifying
        # parameter values in place. After this call the model is "smarter"
        # by one tiny step.
        optimizer.step()

        # ---- 6. Bookkeeping for epoch-level metrics ----
        # loss.item() converts the 0-D loss tensor to a Python float. We
        # multiply by batch size so we can divide by total samples at the
        # end and get a true per-sample mean (correct even if the last
        # batch is smaller than the rest).
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        # Predicted class = argmax over logits. We compare against ground
        # truth and count matches. .item() pulls the scalar out of the tensor.
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Run the model over a held-out split (val or test) WITHOUT updating
    weights. Returns mean loss, accuracy, and F1-score for the abnormal
    (positive) class.

    Parameters
    ----------
    model : nn.Module
        AudioClassifier on `device`.
    loader : DataLoader
        Yields batches from the validation or test set.
    criterion : nn.Module
        Same weighted CrossEntropyLoss used for training.
    device : torch.device
        Compute device.

    Returns
    -------
    avg_loss : float
        Mean loss across all batches.
    accuracy : float
        Fraction of correct predictions. Reported for completeness only —
        see the Accuracy Paradox note below.
    f1 : float
        F1-score for the positive (abnormal) class. The metric we actually
        care about for model selection.
    """

    # model.eval() flips dropout off and tells BatchNorm to use its
    # accumulated running statistics rather than the current batch's stats.
    # This makes evaluation deterministic and correct on any batch size.
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # ====================================================================
    # WHY WE COLLECT EVERY PREDICTION + LABEL — THE ACCURACY PARADOX
    # ====================================================================
    # Plain accuracy is dangerously misleading on imbalanced datasets like
    # MIMII. If 90% of clips are "normal", a broken model that ALWAYS
    # predicts "normal" achieves 90% accuracy while catching ZERO anomalies
    # — total failure for predictive maintenance, but a metric that looks
    # great. This is "the accuracy paradox": the rare event we actually
    # want to detect is exactly the event accuracy is most willing to ignore.
    #
    # F1-Score solves this by combining two complementary metrics:
    #   - Precision = TP / (TP + FP)
    #         "Of the clips we flagged as abnormal, how many really were?"
    #   - Recall    = TP / (TP + FN)
    #         "Of the clips that really were abnormal, how many did we catch?"
    #   - F1        = 2 * (Precision * Recall) / (Precision + Recall)
    #         The harmonic mean — punished severely if EITHER is poor.
    #
    # The "always normal" model gets Precision = 0/0 (no positive predictions
    # at all) and Recall = 0 (caught nothing), so F1 = 0. That's the honest
    # signal accuracy refuses to give us.
    #
    # To compute F1 globally over the validation epoch we have to accumulate
    # every prediction and every label across all batches — F1 is NOT a
    # batch-additive metric the way loss/accuracy are (you can't average
    # batch-level F1s and get the dataset-level F1 in general).
    all_preds = []   # All predicted class indices, one per validation sample.
    all_labels = []  # Corresponding ground-truth labels, parallel to all_preds.

    # torch.no_grad() disables autograd's gradient tracking for everything
    # inside this block. Two big wins:
    #   1. Lower memory use — no autograd graph is allocated.
    #   2. Faster — no bookkeeping on every operation.
    # We don't need gradients during evaluation because we're not training.
    with torch.no_grad():
        for spectrograms, labels in loader:
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(spectrograms)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

            # Stash this batch's predictions and labels for the global F1
            # calculation after the loop. .cpu() pulls tensors off the GPU
            # (sklearn lives in NumPy land) and .tolist() avoids holding a
            # bunch of small tensors in memory across iterations.
            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total

    # pos_label=1 → compute F1 specifically for the "abnormal" class. This
    #   is the binary-classification convention we set up in dataset.py
    #   (normal=0, abnormal=1). For predictive maintenance, F1 of the
    #   positive class is the metric that matters: missing an anomaly
    #   (false negative) means a machine fault goes undetected.
    # zero_division=0 → if precision OR recall is undefined (e.g., the model
    #   produced zero positive predictions in this epoch), return 0.0 instead
    #   of raising a warning. That's the correct interpretation: a model that
    #   never flags anomalies should score 0 on F1, not crash the run.
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    return avg_loss, accuracy, f1


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """
    Wire everything together: args → device → data → model → optimizer →
    loop → checkpoint. Runs when you execute `python src/train.py [...]`.
    """

    # ---- 0. Parse CLI args (replaces the old hardcoded constants) ----
    # We resolve hyperparameters here, at the top, so every subsequent step
    # references args.lr / args.batch_size / args.base_filters / args.run_name.
    args = parse_args()

    # Build a per-run checkpoint path. Embedding run_name in the filename is
    # CRITICAL during HPO — without it, every grid-search combination would
    # save into best_model.pth and clobber the previous run's weights, making
    # later evaluation impossible.
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{args.run_name}.pth")

    print("=" * 60)
    print("TRAINING — AudioClassifier on Log-Mel-Spectrograms")
    print(f"  Run name:     {args.run_name}")
    print(f"  Checkpoint:   {checkpoint_path}")
    print("=" * 60)

    # ---- 1. Device ----
    print("\n[1/5] Selecting compute device...")
    device = get_device()

    # ---- 2. Data ----
    print("\n[2/5] Building DataLoaders...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
    )
    # test_loader is built but not used here — final test-set evaluation
    # belongs in a separate evaluate.py script so we can choose when to
    # "spend" our one look at the held-out test set.
    _ = test_loader

    # ---- 3. Model ----
    print("\n[3/5] Instantiating model...")
    model = AudioClassifier(base_filters=args.base_filters)
    # Move EVERY parameter buffer of the model onto the compute device.
    # Must happen before constructing the optimizer — Adam captures parameter
    # references when initialized, and they need to already point at GPU
    # memory for CUDA training to work.
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ---- 4. Loss & Optimizer ----
    print("\n[4/5] Setting up loss function and optimizer...")

    # Convert the NumPy class_weights array (computed in dataset.py) to a
    # torch.float32 tensor and ship it to the same device as the model.
    # CrossEntropyLoss requires the weight tensor on the same device as its
    # inputs — otherwise PyTorch raises "Expected all tensors to be on the
    # same device" during the loss call.
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32,
                                         device=device)

    print(f"  Class weights (on {device}): {class_weights_tensor.tolist()}")

    # nn.CrossEntropyLoss with weight=class_weights_tensor multiplies each
    # sample's loss by the weight of its TRUE class before averaging across
    # the batch. Result: gradient updates carry larger magnitude when we
    # mis-classify a rare-class sample, pulling the model toward a more
    # balanced decision boundary.
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Adam: adaptive moment estimation. Maintains per-parameter learning
    # rates that adjust based on the first and second moments of recent
    # gradients. It's the de-facto default optimizer for new CNN projects
    # because it works well across a wide range of architectures and
    # hyperparameters without manual tuning.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Learning Rate Scheduler ----
    # ReduceLROnPlateau watches a metric (here, validation loss) and divides
    # the learning rate by `factor` whenever the metric fails to improve for
    # `patience` consecutive epochs.
    #
    # WHY DROP THE LR ON PLATEAUS?
    #   Imagine the loss landscape as a hilly terrain. A large LR takes big
    #   steps — great for descending the steep slopes early in training,
    #   but once we're near the bottom of a valley those big steps overshoot
    #   the actual minimum and bounce around it. Halving the LR shrinks our
    #   steps so we can settle into a tighter, lower point. This is one of
    #   the cheapest ways to extract the last few percentage points of
    #   accuracy from a model that has "almost converged."
    #
    # mode='min'      → metric we want to MINIMIZE (validation loss).
    # factor=0.5      → new_lr = old_lr * 0.5 each time we trigger.
    # patience=3      → wait 3 epochs of no-improvement before triggering.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    # ---- 5. Training loop ----
    print("\n[5/5] Starting training loop...")
    print(f"  Hyperparameters: lr={args.lr}, epochs={NUM_EPOCHS}, "
          f"batch_size={args.batch_size}, base_filters={args.base_filters}")
    print(f"  LR scheduler: ReduceLROnPlateau(factor={LR_SCHEDULER_FACTOR}, "
          f"patience={LR_SCHEDULER_PATIENCE})")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")

    # Make sure the checkpoint directory exists. exist_ok=True is idempotent.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Track the best (lowest) validation loss seen so far. We start at +inf
    # so the very first epoch's val loss is guaranteed to be an improvement
    # and trigger a checkpoint save.
    best_val_loss = float("inf")
    best_epoch = -1

    # ---- Early stopping bookkeeping ----
    # Counts how many epochs in a ROW we've gone without beating
    # best_val_loss. Reset to 0 every time we improve. When it hits
    # EARLY_STOPPING_PATIENCE, we break out of the training loop.
    #
    # WHY EARLY STOPPING?
    #   - Saves compute. Once validation loss has stopped improving for
    #     several epochs, additional training is unlikely to help —
    #     finishing the remaining epochs just burns Colab credits.
    #   - Prevents overfitting. Past the convergence point, the model
    #     starts memorizing training-set noise, training loss keeps
    #     dropping, but validation loss climbs back up. Stopping at the
    #     plateau preserves the best generalization point we ever reached
    #     (which is already saved in best_model.pth via the checkpoint).
    epochs_since_improvement = 0

    print()
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>10} | {'Val Acc':>9} | {'Val F1':>7} | "
          f"{'LR':>9} | {'Time':>6} | Note")
    print("-" * 106)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Run one training pass and one validation pass. evaluate() now
        # returns three values — loss, accuracy, and the imbalance-aware
        # F1-score for the abnormal class (see the long comment in
        # evaluate() for the "Accuracy Paradox" rationale).
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        # ---- Learning rate scheduler step ----
        # Capture the current LR BEFORE calling scheduler.step() so we can
        # detect (by comparing before vs. after) whether the scheduler
        # decided to reduce it on this epoch. We read from
        # optimizer.param_groups[0]['lr'] because Adam stores the
        # currently-active LR there — that's the value the scheduler
        # actually mutates.
        lr_before = optimizer.param_groups[0]['lr']

        # ReduceLROnPlateau is a "metric-driven" scheduler — unlike StepLR
        # or CosineAnnealingLR which step on epoch count, this one needs
        # the value of the metric we're monitoring (val_loss). It compares
        # this against its internal "best so far" and increments its own
        # patience counter when there's no improvement.
        scheduler.step(val_loss)

        lr_after = optimizer.param_groups[0]['lr']

        # ---- Checkpoint if validation loss improved ----
        # Strictly less-than comparison: ties don't trigger a re-save (saves
        # disk I/O and avoids overwriting a good checkpoint with an identical
        # one). Saving state_dict (not the full model) is the recommended
        # PyTorch idiom — it's smaller, more portable, and decouples the
        # weights from the model class definition.
        improved_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            improved_marker = "✓ saved"
            # Reset the early-stopping counter — we're making progress again.
            epochs_since_improvement = 0
        else:
            # No improvement this epoch. Bump the counter; we'll check it
            # against EARLY_STOPPING_PATIENCE after logging.
            epochs_since_improvement += 1

        # ---- Annotate the log row when the LR actually dropped ----
        # We only mention it on the epoch the change happens, so the log
        # stays uncluttered. lr_after < lr_before is a strict comparison —
        # ReduceLROnPlateau never raises the LR, only lowers it.
        if lr_after < lr_before:
            lr_drop_note = f"LR↓ {lr_before:.2e}→{lr_after:.2e}"
            improved_marker = (improved_marker + " | " + lr_drop_note
                               if improved_marker else lr_drop_note)

        epoch_secs = time.time() - epoch_start

        # Per-epoch log row. Tabular format makes it easy to scan the
        # training curve at a glance. Val F1 is the column to actually
        # watch — accuracy is shown for sanity-checking but is misleading
        # on imbalanced data.
        print(f"{epoch:>5d} | {train_loss:>10.4f} | {train_acc:>9.2%} | "
              f"{val_loss:>10.4f} | {val_acc:>9.2%} | {val_f1:>7.4f} | "
              f"{lr_after:>9.2e} | {epoch_secs:>5.1f}s | {improved_marker}")

        # ---- Early stopping check ----
        # Done AFTER the log row so the final epoch is still printed before
        # we announce the stop. Reaching the patience limit means
        # validation loss has plateaued long enough that further epochs
        # are unlikely to help — bail out and save the remaining compute.
        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print("-" * 106)
            print(f"\nEarly stopping triggered: validation loss has not "
                  f"improved for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
            print(f"  Stopped at epoch {epoch} of {NUM_EPOCHS}.")
            break

    # ---- Wrap-up ----
    print("-" * 95)
    print(f"\nTraining complete.")
    print(f"  Best validation loss: {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"  Best checkpoint:      {checkpoint_path}")
    print("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================
# Guarded with __main__ so importing this module (e.g., from a notebook)
# doesn't accidentally kick off a 20-epoch training run.

if __name__ == "__main__":
    main()
