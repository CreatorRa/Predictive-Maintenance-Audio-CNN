"""
explain.py — Grad-CAM Visualization for Audio Anomaly Detection
============================================================
This script is the SIXTH and FINAL stage of our Predictive Maintenance Audio
CNN project. After preprocess.py produced spectrograms, dataset.py served
them, model.py defined the CNN, train.py trained it, and evaluate.py reported
quantitative metrics, this script answers the qualitative question:

        "WHY did the model classify THIS clip as Normal/Abnormal?"

It loads our best checkpoint, runs Grad-CAM (Gradient-weighted Class Activation
Mapping) on one Normal and one Abnormal test sample, and saves side-by-side
PNGs comparing the raw Log-Mel-Spectrogram to a heatmap that highlights the
exact time/frequency regions the model relied on to make its decision.

WHAT IS GRAD-CAM AND WHY DO WE NEED IT?
    A trained CNN is a black box: it spits out a label and a probability, but
    not a reason. For predictive maintenance, "trust me, that pump is failing"
    is not good enough — a maintenance crew is about to spend money and time
    on the recommendation. We need to SEE what convinced the model.

    Grad-CAM (Selvaraju et al., 2017) opens the box by tracing the gradient
    of the predicted class's logit BACKWARD from the output all the way to
    the activations of the LAST convolutional layer. The recipe is:

        1. Forward pass: feed the spectrogram through the CNN and record
           the activations A^k of every channel k in the final conv layer.
           Shape: (channels, H, W) — for our model, (128, 8, 19).

        2. Backward pass on the chosen class score y^c: compute
           dA^k / d(y^c) for every cell of every channel. Average those
           gradients across the spatial (H, W) dimensions to get one
           scalar weight α_k per channel:
                α_k = mean over (i, j) of  ∂y^c / ∂A^k_{i,j}
           Intuitively, α_k says "how much does increasing channel k's
           activation push up the score for class c?"

        3. Weighted sum: combine the channels into a single 2D heatmap
                L = ReLU( Σ_k  α_k * A^k )
           ReLU keeps only POSITIVE contributions — regions that argue
           FOR the class — because regions arguing AGAINST it are
           noise we don't want to highlight.

        4. Upsample the (8, 19) heatmap back to the original (128, 313)
           spectrogram size with bilinear interpolation, normalize to
           [0, 1], and overlay it on the input as a translucent jet
           colormap.

    The result tells us, pixel-by-pixel: "the model paid attention HERE,
    in this frequency band, at this time, to make its decision." A bright
    blob over a specific Mel band at a specific moment in time is a
    direct, visual answer to "why?"

WHY THE LAST CONV LAYER?
    Earlier conv layers see low-level patterns (edges, simple textures).
    Later conv layers see semantic, class-discriminative features (the
    actual "this is an abnormal squeal" detector). The last conv layer is
    the deepest layer that still has spatial structure — fully-connected
    layers below it have collapsed (H, W) into a flat vector and lost the
    "where in the spectrogram?" information we need for a heatmap. Hence
    AudioClassifier.get_final_conv_layer() returns self.conv4: it is the
    sweet spot between "rich semantics" and "spatial layout still intact."

WHY THIS MATTERS FOR PREDICTIVE MAINTENANCE
    Engineers will only trust an automated fault detector if they can
    verify its reasoning. Grad-CAM lets us do exactly that:

      - DOMAIN-KNOWLEDGE SANITY CHECK: a real bearing fault produces
        characteristic high-frequency harmonics. If our Grad-CAM heatmap
        for "Abnormal" lights up the corresponding Mel bands, an engineer
        sees physics-consistent reasoning and trust grows. If instead it
        lights up the silent regions or background hum, we know the model
        learned a spurious correlation (e.g. mic placement artifacts).

      - DEBUGGING DATASET LEAKS: if the heatmap consistently fires in the
        first 100ms of every clip, the model is probably memorizing a
        recording-system artifact at clip onset, not a real fault. Grad-CAM
        catches this kind of label leakage that pure metrics cannot.

      - EXPLAINABILITY FOR DEPLOYMENT: in regulated industries (aerospace,
        automotive, healthcare-adjacent manufacturing), an opaque model
        often cannot be deployed at all. A model that ships a Grad-CAM
        explanation alongside every alert is far easier to certify.

    In short: metrics tell us IF the model is right; Grad-CAM tells us WHY,
    and that is the only honest path to deployable predictive maintenance.

USAGE:
    From the project root, AFTER training (so models/best_model.pth exists):
        python src/explain.py

    Outputs:
        docs/visualizations/gradcam_normal.png
        docs/visualizations/gradcam_abnormal.png
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse                    # CLI args for HPO compatibility
import os                          # Filesystem operations (mkdirs, path joins)
import sys                         # Used to make the project root importable
import numpy as np                 # Heatmap math + image array manipulation
import torch                       # PyTorch core for model, tensors, device
import matplotlib.pyplot as plt    # Plotting and saving the PNG figures

# librosa.display.specshow gives us spectrogram axes in physical units —
# seconds on the x-axis and Hertz (Mel-scaled) on the y-axis — instead of
# raw array indices (0..313 frames, 0..128 mel bins). Physical units are
# essential for an engineering report because they let a domain expert
# point at the heatmap and say "the model fired at 5.2 seconds, around
# 4 kHz" instead of "the model fired around column 162, row 90."
import librosa
import librosa.display

# pytorch-grad-cam is a third-party library that wraps the Grad-CAM hook
# bookkeeping (forward + backward hooks, gradient capture, ReLU + upsampling)
# behind a clean class-based API. We use TWO things from it:
#   - GradCAM:           the algorithm itself, instantiated with our model
#                        and a list of target conv layers.
#   - show_cam_on_image: a helper that overlays a [0,1] heatmap onto an
#                        RGB image as a translucent jet colormap.
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ClassifierOutputTarget is the standard "I want to explain class index c"
# wrapper. Passing it to GradCAM tells the library which logit to back-prop
# from. (Without a target, the library would back-prop from the argmax
# class, which is often what we want anyway — but being explicit avoids
# surprises when the model is confidently wrong on a sample.)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Make `from src.xxx import ...` work when running this file directly. ---
# Same trick used in src/train.py and src/evaluate.py: prepend the project
# root to sys.path so absolute imports below resolve regardless of how the
# script is launched.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Our project modules. We deliberately reuse the SAME helpers that train.py
# and evaluate.py use so explanation runs in lock-step with training/eval —
# any change to the checkpoint dir or device-picking logic propagates
# automatically.
from src.model import AudioClassifier
from src.dataset import get_dataloaders
# get_device and CHECKPOINT_DIR live in train.py. Per-run --base_filters and
# --run_name now come from the CLI, not module-level constants.
from src.train import get_device, CHECKPOINT_DIR
# SAMPLE_RATE and HOP_LENGTH come from preprocess.py because they govern
# the time axis of the spectrogram: each column corresponds to HOP_LENGTH
# samples at SAMPLE_RATE Hz. librosa.display.specshow uses both to convert
# column indices into seconds, so the values here MUST match whatever
# preprocess.py used when generating the .npy files.
from src.preprocess import SAMPLE_RATE, HOP_LENGTH


# ============================================================================
# CONFIGURATION — Output dir + class display names. Per-file paths are built
# inside main() once we know the run_name from CLI args.
# ============================================================================

# Directory where we drop the two PNGs. We reuse the same folder evaluate.py
# writes to so all "report assets" live in one place.
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "docs", "visualizations")

# Display labels. Index matches the integer class label (normal=0, abnormal=1).
# Matches the convention from dataset.py / evaluate.py.
CLASS_NAMES = ["Normal", "Abnormal"]


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """
    Mirror train.py / evaluate.py — selects which checkpoint to load and
    embeds run_name in the heatmap filenames so HPO experiments don't
    clobber one another's explanation figures.
    """
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for a trained AudioClassifier."
    )
    parser.add_argument("--base_filters", type=int, default=16,
                        help="Conv-stack width — MUST match training value.")
    parser.add_argument("--run_name", type=str, default="default_run",
                        help="Run tag — selects checkpoint and namespaces "
                             "the output PNGs.")
    # batch_size / lr are accepted but unused — keeping the signature uniform
    # across train/evaluate/explain lets tune.py shell out with the same args.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Accepted for CLI uniformity; ignored here.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Accepted for CLI uniformity; ignored here.")
    return parser.parse_args()


# ============================================================================
# STEP 1 — LOAD THE TRAINED MODEL
# ============================================================================

def load_model_for_explain(device, checkpoint_path, base_filters):
    """
    Rebuild the AudioClassifier architecture, load the trained weights from
    disk, push it to `device`, and switch it into eval() mode.

    WHY eval() MODE?
        eval() does two important things for explanation:
          1. Disables Dropout — at training time, Dropout randomly zeros 50%
             of activations. We want DETERMINISTIC heatmaps for the report,
             not heatmaps that change every run.
          2. Switches BatchNorm to its accumulated running mean/variance
             instead of the current batch's statistics. This is critical
             because we run Grad-CAM on a single sample (batch size 1) —
             batch-statistics on a batch of 1 are nonsensical.

    WHY NOT torch.no_grad()?
        Grad-CAM REQUIRES gradient computation — it back-props from the
        chosen class logit through the conv stack to capture gradients on
        the final conv layer. Wrapping the forward pass in no_grad() would
        silently break the algorithm (you'd get a "tensor does not require
        grad" error or, worse, an all-zero heatmap). So we use eval() to
        disable dropout/BN tracking but leave autograd ENABLED.

    Parameters
    ----------
    device : torch.device
        Where the model should live (cuda / mps / cpu).
    checkpoint_path : str
        Path to the .pth file saved by train.py.
    base_filters : int
        Conv-stack width. MUST equal the value used at training time —
        otherwise the saved state_dict won't fit the rebuilt architecture
        and load_state_dict() raises a shape-mismatch error.

    Returns
    -------
    model : AudioClassifier
        Architecture rebuilt, weights loaded, on `device`, in eval mode.
    """

    # Helpful upfront error: if the checkpoint is missing the user almost
    # certainly forgot to run train.py first. A clear message saves a
    # confusing stack trace.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Run src/train.py first to produce it."
        )

    print(f"  Loading model from: {checkpoint_path}")
    print(f"  Architecture:       AudioClassifier(base_filters={base_filters})")

    # Step A: instantiate the SAME architecture we trained. base_filters
    # must match exactly — see the docstring above for why.
    model = AudioClassifier(base_filters=base_filters)

    # Step B: load the saved weights. map_location=device is the safe
    # cross-device idiom — it lets us load a checkpoint trained on Colab
    # GPU and run it on a local Mac MPS or CPU without errors.
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Step C: ship the parameters to the target device. Must come AFTER
    # load_state_dict (loading into the wrong device is allowed but slow).
    model = model.to(device)

    # Step D: eval mode — Dropout off, BatchNorm using running stats.
    # See the docstring for the full reasoning.
    model.eval()

    print(f"  Model ready on device: {device}")
    return model


# ============================================================================
# STEP 2 — FIND ONE NORMAL AND ONE ABNORMAL TEST SAMPLE
# ============================================================================

def find_one_per_class(test_loader):
    """
    Walk through the test DataLoader and return ONE sample of each class.

    WHY THE TEST LOADER (not train/val)?
        Explanation should illustrate model behavior on data the model has
        NEVER seen during training. If we pulled samples from the training
        set, the heatmaps would be biased toward the patterns the model
        memorized rather than the ones it generalized.

    WHY ITERATE BATCH-BY-BATCH?
        The DataLoader yields batches of size 32, but Grad-CAM is a per-
        sample algorithm. We unpack the batch, scan each sample's label,
        and stop the moment we have one of each class. This is much faster
        than materializing the entire test split into memory.

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        Un-shuffled iterator over the test split.

    Returns
    -------
    normal_spec : torch.Tensor, shape (1, 1, 128, 313)
        One Normal (label 0) spectrogram, with batch dim already added so
        it can be fed straight into the model / GradCAM.
    abnormal_spec : torch.Tensor, shape (1, 1, 128, 313)
        One Abnormal (label 1) spectrogram, same shape conventions.

    Raises
    ------
    RuntimeError
        If the loader is exhausted before both classes are found — usually
        means the test split is missing one of the classes (broken split
        or an empty class folder upstream).
    """

    # We use None as a sentinel for "haven't found one yet". Using a typed
    # default like an empty tensor would be ambiguous.
    normal_sample   = None
    abnormal_sample = None

    # Iterate over every batch in the test set. The loader already pre-
    # standardizes spectrograms (mean/std normalize per sample) inside
    # SpectrogramDataset.__getitem__, so the tensors here are exactly
    # what the model was trained on — no extra normalization needed.
    for spectrograms, labels in test_loader:
        # spectrograms: (B, 1, 128, 313) float32
        # labels:       (B,)             int64

        # Walk through each sample in this batch.
        for i in range(labels.size(0)):
            label = int(labels[i].item())
            # spectrograms[i] is (1, 128, 313). We add a batch dim with
            # unsqueeze(0) so the result is (1, 1, 128, 313) — the exact
            # 4D shape AudioClassifier.forward and GradCAM expect.
            single = spectrograms[i].unsqueeze(0)

            if label == 0 and normal_sample is None:
                normal_sample = single
            elif label == 1 and abnormal_sample is None:
                abnormal_sample = single

            # Early exit: as soon as we have one of each, stop scanning.
            # The test split can be thousands of samples; this saves
            # essentially all of them.
            if normal_sample is not None and abnormal_sample is not None:
                return normal_sample, abnormal_sample

    # If we fell through the loop, one or both classes are missing.
    raise RuntimeError(
        "Could not find one sample of each class in the test loader. "
        "Check that data/processed/normal/ and data/processed/abnormal/ "
        "both contain .npy files."
    )


# ============================================================================
# STEP 3 — GENERATE A GRAD-CAM HEATMAP FOR A SINGLE SAMPLE
# ============================================================================

def generate_gradcam(model, target_layers, spectrogram, target_class, device):
    """
    Run Grad-CAM on one spectrogram and return both the raw spectrogram
    (as a 2D NumPy array) and the corresponding [0, 1] heatmap.

    KEY IMPLEMENTATION DETAILS:
      - requires_grad=True is REQUIRED on the input tensor. pytorch_grad_cam
        internally calls .backward() to capture gradients on the target
        layer's activations. Without requires_grad on the input, autograd
        won't build the graph and the library raises an error.
      - We move the tensor to `device` BEFORE setting requires_grad. The
        order matters: requires_grad on a CPU tensor that's then moved to
        GPU produces a fresh tensor that's no longer a leaf — autograd
        gets confused.
      - pytorch_grad_cam returns a NumPy array of shape (B, H, W) where
        H and W are the ORIGINAL input dims (128, 313 here) — it has
        already done the bilinear upsample and [0,1] normalization for us.

    Parameters
    ----------
    model : AudioClassifier
        Trained model, in eval() mode, on `device`.
    target_layers : list of nn.Module
        The conv layers Grad-CAM should hook into. For our architecture
        this is exactly [model.get_final_conv_layer()].
    spectrogram : torch.Tensor, shape (1, 1, 128, 313)
        A single spectrogram with the batch dimension already added.
    target_class : int
        Which class's logit to back-prop from. 0 = Normal, 1 = Abnormal.
        We deliberately use the GROUND TRUTH label (not the model's
        prediction) so we visualize "what convinced the model THIS sample
        was {actual class}" — the most honest framing for an explanation.
    device : torch.device
        Compute device.

    Returns
    -------
    spec_2d : np.ndarray, shape (128, 313)
        The original spectrogram as a plain 2D NumPy array (channel and
        batch dims stripped). Used as the background image for the overlay.
    heatmap : np.ndarray, shape (128, 313), values in [0, 1]
        The Grad-CAM saliency map — same spatial size as the input,
        already upsampled and normalized by the library.
    """

    # --- Step A: move the tensor to the right device and enable autograd ---
    # Grad-CAM needs gradients to flow back through the input. Setting
    # requires_grad here makes the input a leaf node that autograd will
    # accumulate gradients on during the backward pass.
    input_tensor = spectrogram.to(device)
    input_tensor.requires_grad_(True)

    # --- Step B: build the GradCAM object ---
    # GradCAM registers forward AND backward hooks on every layer in
    # target_layers. The hooks capture activations on the forward pass
    # and gradients on the backward pass. Using `with ... as cam:` makes
    # sure the hooks are removed afterwards even if an exception fires —
    # leaving them attached would leak memory and corrupt subsequent
    # forward passes during evaluation.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # ClassifierOutputTarget(c) tells the library "back-prop from
        # logit c". We wrap it in a list because GradCAM expects one
        # target per sample in the batch (we have batch size 1, so one
        # target).
        targets = [ClassifierOutputTarget(target_class)]

        # cam(...) does, internally:
        #   1. forward pass → capture activations A^k via forward hook
        #   2. compute scalar = sum of selected logits for the targets
        #   3. backward pass → capture gradients ∂scalar/∂A^k via backward hook
        #   4. spatially-average the gradients to per-channel weights α_k
        #   5. weighted-sum the activations: L = ReLU( Σ α_k * A^k )
        #   6. bilinearly upsample L from (8, 19) → (128, 313)
        #   7. min-max normalize each sample's heatmap to [0, 1]
        # Returned shape: (B, H, W) NumPy array. With B=1 we squeeze to (H, W).
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Strip the batch dimension. grayscale_cam[0] is the (128, 313) heatmap.
    heatmap = grayscale_cam[0]

    # --- Step C: extract the original spectrogram as a 2D array ---
    # We pull from the ORIGINAL spectrogram tensor (not input_tensor) so
    # we get a clean, gradient-free copy for plotting. Detach is safety:
    # it severs the autograd graph so .numpy() can succeed even if the
    # tensor still references gradient buffers.
    # Squeeze removes the (batch=1, channel=1) dims, leaving (128, 313).
    spec_2d = spectrogram.detach().cpu().numpy().squeeze()

    return spec_2d, heatmap


# ============================================================================
# STEP 4 — RENDER THE SIDE-BY-SIDE COMPARISON FIGURE
# ============================================================================

def plot_and_save_explanation(spec_2d, heatmap, class_name, output_path):
    """
    Render a single Matplotlib figure with two subplots:
        Left:  the original Log-Mel-Spectrogram (viridis colormap)
        Right: the same spectrogram with the Grad-CAM heatmap overlaid

    WHY SIDE-BY-SIDE?
        Showing the heatmap alone hides what region of the spectrogram it
        sits over. Showing the spectrogram alone hides the model's
        attention. Putting them next to each other lets a domain expert
        compare "what's there" against "what the model looked at" in one
        glance — exactly the comparison needed to validate or refute the
        model's reasoning.

    OVERLAY MECHANICS:
        show_cam_on_image expects:
          - rgb_img: an RGB float image in [0, 1], shape (H, W, 3).
          - mask:    the [0, 1] heatmap, shape (H, W).
        It maps the mask through a jet colormap, blends it with the RGB
        image at use_rgb=True (RGB ordering, not BGR), and returns a
        uint8 image we can imshow directly.

    To build the rgb_img we need to convert our 2D spectrogram (which is
    in dB units, often roughly in [-80, 0]) into a [0, 1] float RGB. The
    cleanest way is:
        1. Min-max normalize the dB values to [0, 1].
        2. Stack the result on itself 3 times to get a (H, W, 3) "grayscale
           as RGB" image. show_cam_on_image will paint the jet heatmap
           over this gray background.

    Parameters
    ----------
    spec_2d : np.ndarray, shape (128, 313)
        The standardized spectrogram (already mean/std-normalized inside
        SpectrogramDataset). We re-normalize to [0, 1] purely for display.
    heatmap : np.ndarray, shape (128, 313), values in [0, 1]
        Grad-CAM saliency map.
    class_name : str
        Used in the figure title and subplot titles ("Normal" or "Abnormal").
    output_path : str
        Where to save the PNG. Parent directory is created on demand.
    """

    # --- Build a [0, 1] grayscale-RGB version of the spectrogram for overlay ---
    # The spectrogram was per-sample standardized, so its min/max vary across
    # samples. A per-sample min-max makes the contrast pop the same way for
    # every figure we produce.
    spec_min = spec_2d.min()
    spec_max = spec_2d.max()
    # Guard against the (extremely unlikely) flat-spectrogram edge case
    # where min == max — division by zero would produce NaNs and a black plot.
    spec_range = spec_max - spec_min if (spec_max - spec_min) > 1e-9 else 1.0
    spec_norm = (spec_2d - spec_min) / spec_range  # now in [0, 1]

    # Stack the 2D grayscale into a 3-channel RGB image. axis=-1 puts the
    # channel dim LAST, matching show_cam_on_image's (H, W, 3) expectation.
    spec_rgb = np.stack([spec_norm] * 3, axis=-1).astype(np.float32)

    # show_cam_on_image overlays the heatmap on the RGB background using
    # the jet colormap. use_rgb=True tells it our image is already in RGB
    # order (matplotlib convention) rather than BGR (OpenCV convention).
    overlay = show_cam_on_image(spec_rgb, heatmap, use_rgb=True)

    # --- Build the figure ---
    # Two side-by-side subplots. We do NOT share the y-axis here because
    # librosa.display.specshow draws its own log-Mel-scaled y-axis labels
    # on each subplot, and sharing would suppress the right-hand axis.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Left subplot: raw spectrogram (with physical-unit axes) ===
    # librosa.display.specshow handles three things imshow cannot:
    #   1. x-axis in SECONDS: it computes time = column_index * hop_length / sr
    #      from the sr and hop_length args, so what was "frames 0..313" now
    #      reads as "0.0 s … ~10.0 s".
    #   2. y-axis in HERTZ on the Mel scale: low frequencies get more pixels
    #      per Hz than high ones, exactly matching the perceptual Mel warping
    #      that preprocess.py applied. y_axis='mel' triggers this behavior.
    #   3. Origin/orientation: specshow puts low frequencies at the bottom
    #      automatically — no origin='lower' needed.
    # cmap='viridis' matches the convention used elsewhere in the project.
    img0 = librosa.display.specshow(
        spec_2d,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[0],
        cmap="viridis",
    )
    axes[0].set_title(f"Original Mel-Spectrogram ({class_name})")
    # Colorbar so the reader can read off dB values if they care. We attach
    # to the QuadMesh returned by specshow (img0) so the colors match the
    # subplot exactly.
    fig.colorbar(img0, ax=axes[0], format="%.1f")

    # === Right subplot: Grad-CAM overlay (also with physical-unit axes) ===
    # 1. Use specshow with the 2D data to calculate the physical time/Hz axes, 
    #    but set alpha=0 so the plot itself is invisible.
    librosa.display.specshow(
        spec_2d,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        alpha=0.0
    )
    
    # 2. Grab the physical axis limits that librosa just calculated
    x_min, x_max = axes[1].get_xlim()
    y_min, y_max = axes[1].get_ylim()

    # 3. Draw the RGB overlay using standard imshow, but stretch it to fit the physical limits
    axes[1].imshow(
        overlay, 
        aspect="auto", 
        origin="lower",
        extent=[x_min, x_max, y_min, y_max]
    )
    axes[1].set_title(f"Grad-CAM: regions driving '{class_name}' prediction")

    # Overall figure title + tight layout so subplot titles don't collide
    # with the suptitle.
    fig.suptitle(f"Model Explanation — {class_name} Test Sample",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    # --- Save to disk ---
    # Make sure the destination folder exists. exist_ok=True means we
    # don't crash if it's already there.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 150 DPI is the same setting evaluate.py uses for the confusion matrix
    # and ROC curve — keeps the report's visuals consistent.
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    # Close the figure to release matplotlib's internal references —
    # leaving figures open in a script-style run causes a memory warning
    # after enough iterations.
    plt.close(fig)

    print(f"  Saved: {output_path}")


# ============================================================================
# STEP 5 — ORCHESTRATOR
# ============================================================================

def main():
    """
    Top-level entry point. Loads the run-specific checkpoint, picks one
    sample of each class from the test set, and writes per-run Grad-CAM
    figures whose filenames include run_name so HPO results stay distinct.
    """

    # ---- 0. Parse CLI args ----
    args = parse_args()

    # Per-run paths — checkpoint to load, plus the two heatmap PNGs to write.
    checkpoint_path       = os.path.join(CHECKPOINT_DIR, f"best_model_{args.run_name}.pth")
    gradcam_normal_path   = os.path.join(VISUALIZATION_DIR, f"gradcam_normal_{args.run_name}.png")
    gradcam_abnormal_path = os.path.join(VISUALIZATION_DIR, f"gradcam_abnormal_{args.run_name}.png")

    print("=" * 60)
    print("GRAD-CAM EXPLAINABILITY")
    print(f"  Run name:    {args.run_name}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print("=" * 60)

    # --- 1. Device selection (reuses train.py's helper for consistency) ---
    print("\n[1/5] Selecting device...")
    device = get_device()

    # --- 2. Load the trained model and grab the target layer ---
    # target_layers MUST be a list — the library is built to support
    # explaining multiple layers at once even though we only use one.
    print("\n[2/5] Loading trained model...")
    model = load_model_for_explain(
        device,
        checkpoint_path=checkpoint_path,
        base_filters=args.base_filters,
    )
    target_layers = [model.get_final_conv_layer()]

    # --- 3. Build the test DataLoader and pick one sample per class ---
    # We deliberately throw away the train/val loaders and class_weights —
    # explanation only ever looks at the held-out test set, and using
    # train/val data here would defeat the purpose of an unbiased
    # qualitative check.
    print("\n[3/5] Building test DataLoader and selecting samples...")
    _, _, test_loader, _ = get_dataloaders(batch_size=args.batch_size)
    normal_spec, abnormal_spec = find_one_per_class(test_loader)
    print(f"  Found one Normal and one Abnormal sample.")

    # --- 4 & 5. Generate and save Grad-CAM for the Normal sample ---
    print("\n[4/5] Generating Grad-CAM for the NORMAL sample...")
    # target_class=0 → back-prop from the "Normal" logit. This visualizes
    # the spectrogram regions that pushed the model toward the (correct)
    # Normal prediction.
    spec_2d, heatmap = generate_gradcam(
        model=model,
        target_layers=target_layers,
        spectrogram=normal_spec,
        target_class=0,
        device=device,
    )
    plot_and_save_explanation(
        spec_2d=spec_2d,
        heatmap=heatmap,
        class_name=CLASS_NAMES[0],
        output_path=gradcam_normal_path,
    )

    # --- 4 & 5. Generate and save Grad-CAM for the Abnormal sample ---
    print("\n[5/5] Generating Grad-CAM for the ABNORMAL sample...")
    # target_class=1 → back-prop from the "Abnormal" logit. This is the
    # heatmap a domain expert would scrutinize most carefully — does the
    # model latch onto frequencies consistent with real bearing/pump
    # faults, or onto recording artifacts?
    spec_2d, heatmap = generate_gradcam(
        model=model,
        target_layers=target_layers,
        spectrogram=abnormal_spec,
        target_class=1,
        device=device,
    )
    plot_and_save_explanation(
        spec_2d=spec_2d,
        heatmap=heatmap,
        class_name=CLASS_NAMES[1],
        output_path=gradcam_abnormal_path,
    )

    print("\n" + "=" * 60)
    print("EXPLAINABILITY COMPLETE")
    print(f"  Heatmaps written to: {VISUALIZATION_DIR}")
    print("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================
# Standard `if __name__ == "__main__":` guard. Required on Windows because
# DataLoader workers (if num_workers > 0) use 'spawn' multiprocessing, which
# re-imports this file in each subprocess — without the guard, every worker
# would re-run main() and we'd get an infinite fork bomb of training jobs.

if __name__ == "__main__":
    main()
