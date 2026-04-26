"""
model.py — CNN Architecture for Audio Anomaly Classification
============================================================
This script is the THIRD stage of our Predictive Maintenance Audio CNN project.
After preprocess.py produces .npy spectrograms and dataset.py serves them via
DataLoaders, this script defines the neural network that learns to classify
those spectrograms as Normal (label 0) or Abnormal (label 1).

WHAT THIS FILE PROVIDES:
    - AudioClassifier: a flexible PyTorch nn.Module implementing a 4-block
      Convolutional Neural Network with batch normalization, ReLU activations,
      max-pooling, dropout regularization, and a 2-class linear output head.
    - Built-in support for Grad-CAM via get_final_conv_layer() — exposes the
      last Conv2d so the explainability script (src/explain.py) can attach
      forward/backward hooks without inspecting model internals.

WHY A CNN FOR AUDIO?
    Even though audio is fundamentally 1D (a time series of amplitudes), once
    we convert it to a 2D Log-Mel-Spectrogram it becomes spatially structured
    just like an image. CNNs excel at finding LOCAL spatial patterns
    (e.g. a horizontal stripe = a sustained tone, a diagonal smear = a
    sweeping frequency, a sudden vertical bar = a click). Anomalies in
    industrial machinery often manifest as exactly these kinds of localized
    spectral signatures, so 2D convolutions are a natural fit.

ARCHITECTURE OVERVIEW (with default base_filters=16):
    Input:                [B,   1, 128, 313]   (1 channel, 128 mels, ~313 time frames)
      Conv Block 1:       [B,  16, 128, 313] → MaxPool → [B,  16, 64, 156]
      Conv Block 2:       [B,  32,  64, 156] → MaxPool → [B,  32, 32,  78]
      Conv Block 3:       [B,  64,  32,  78] → MaxPool → [B,  64, 16,  39]
      Conv Block 4:       [B, 128,  16,  39] → MaxPool → [B, 128,  8,  19]
    Global Avg Pool 2D:   [B, 128,   1,   1]   ← collapses every spatial map to a single mean
    Flatten:              [B, 128]
    Dropout(0.5)
    Linear:               [B, 2]                ← raw logits (no softmax — CrossEntropyLoss applies it)

WHY GLOBAL AVERAGE POOLING (NOT FLATTEN)?
    A standard Flatten + Linear(128) head on the (B, 128, 8, 19) feature map
    produces 19456 * 128 ≈ 2.5 MILLION parameters in a single layer — wildly
    excessive for our small MIMII dataset and a textbook overfitting trap.

    Global Average Pooling collapses every channel's spatial map to its mean,
    going straight from (B, 128, 8, 19) to (B, 128). This:
      - Slashes the head's parameter count from ~2.5M down to ~258 (10,000x).
      - Acts as a strong structural regularizer: the model can no longer
        memorize WHERE features fire, only HOW MUCH each fires overall.
      - Makes the model agnostic to input length — the same trained weights
        accept spectrograms of any duration without shape errors.
    GAP is the standard choice in modern CNNs (ResNet, Inception, etc.) for
    exactly these reasons.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import torch                # PyTorch core (tensor library)
import torch.nn as nn       # Neural network building blocks (layers, modules, losses)
# torch.nn.functional is imported as F by convention but we don't need it here:
# all our operations are stateful layers (Conv2d, BatchNorm2d, etc.) so we use
# the class-based API throughout, which auto-registers parameters with the module.


# ============================================================================
# DEFAULT INPUT SHAPE
# ============================================================================
# These constants describe the EXPECTED shape of inputs from our DataLoader.
# They MUST match the output of src/dataset.py / src/preprocess.py. If you
# change n_mels or hop_length in preprocess.py, update these too.

DEFAULT_N_MELS = 128
# Number of Mel bands — the height of the input spectrogram. Set to match
# the N_MELS constant in preprocess.py.

DEFAULT_TIME_FRAMES = 313
# Number of time frames — the width of the input spectrogram. Set to match
# the (TARGET_LENGTH_SAMPLES // HOP_LENGTH + 1) value from preprocess.py.

NUM_CLASSES = 2
# Binary classification: 0 = normal, 1 = abnormal. Output layer produces
# 2 raw logits per sample. We use CrossEntropyLoss which applies softmax
# internally — DO NOT add softmax to the model output.


# ============================================================================
# THE AUDIO CLASSIFIER MODEL
# ============================================================================

class AudioClassifier(nn.Module):
    """
    A flexible 4-block Convolutional Neural Network for binary classification
    of Log-Mel-Spectrograms.

    DESIGN PRINCIPLES:
        - DOUBLING FEATURE MAPS: Each conv block doubles the number of output
          channels (base_filters → 2x → 4x → 8x). This is a classic CNN pattern
          (used in VGG, ResNet, etc.): as spatial dimensions shrink, we
          compensate by increasing channel depth, keeping total information
          capacity roughly constant while building up more abstract features.

        - PADDING=1 with KERNEL=3 preserves spatial dimensions inside each conv
          layer. The pooling step is the ONLY operation that downsamples,
          which makes the dimension math clean and predictable.

        - BATCHNORM after every Conv stabilizes training by re-centering
          activations to zero-mean unit-variance per channel. It also acts as
          a mild regularizer and lets us use higher learning rates.

        - DROPOUT before the linear layers fights overfitting by randomly
          zeroing 50% of activations during training. This forces the network
          to learn redundant, robust representations rather than relying on
          any single neuron — particularly important here because audio data
          has inherent noise that we don't want the model to memorize.

    Parameters
    ----------
    base_filters : int
        Number of output channels in the FIRST conv layer. Subsequent layers
        scale this up: 2x, 4x, 8x. This is our "model size" knob — increase
        for capacity, decrease for speed.
            base_filters=16 → final block has 128 channels  (small, fast)
            base_filters=32 → final block has 256 channels  (medium)
            base_filters=64 → final block has 512 channels  (large, slow)
    num_classes : int
        Number of output classes. Default: 2 (binary normal vs. abnormal).
    n_mels : int
        Height of the input spectrogram. Default: 128.
    time_frames : int
        Width of the input spectrogram. Default: 313.
    dropout_rate : float
        Probability of zeroing each activation in the dropout layers. 0.5 is
        the classic Hinton et al. value and a strong default. Default: 0.5.
    """

    def __init__(self, base_filters=16, num_classes=NUM_CLASSES,
                 n_mels=DEFAULT_N_MELS, time_frames=DEFAULT_TIME_FRAMES,
                 dropout_rate=0.5):

        # nn.Module's __init__ MUST be called first — it sets up internal
        # tracking for parameters, buffers, and submodules. Skipping this
        # would silently break gradient computation.
        super().__init__()

        # Stash hyperparameters as attributes so they're queryable later
        # (e.g., for logging, for Grad-CAM, for saving model metadata).
        self.base_filters = base_filters
        self.num_classes = num_classes
        self.n_mels = n_mels
        self.time_frames = time_frames

        # ====================================================================
        # CONVOLUTIONAL FEATURE EXTRACTOR
        # ====================================================================
        # Each block follows the canonical pattern:
        #     Conv2d → BatchNorm2d → ReLU → MaxPool2d
        #
        # DIMENSION MATH FOR MAXPOOL (kernel_size=2, stride=2):
        #   output_dim = floor(input_dim / 2)
        # Each pool roughly halves both spatial dimensions, which means after
        # 4 blocks the spatial size is reduced by a factor of 2^4 = 16.
        #
        # WHY MAXPOOL (vs. AveragePool or strided convs)?
        #   MaxPool keeps the strongest activation in each 2x2 window, which
        #   gives a small amount of TRANSLATION INVARIANCE: the network doesn't
        #   care if a feature is at position (5,5) or (5,6) — it'll pool to
        #   the same value. For audio, this means the model is more robust to
        #   small timing jitter or pitch shifts.

        # ----- Block 1 -----
        # Input:  [B, 1, 128, 313]
        # Conv:   in_channels=1, out_channels=base_filters
        #   kernel_size=3 with padding=1 keeps spatial dims unchanged.
        #   Formula: out = floor((in + 2*pad - kernel)/stride) + 1
        #            = floor((128 + 2 - 3)/1) + 1 = 128.
        # After MaxPool(2): floor(128/2)=64,  floor(313/2)=156
        # Output: [B, base_filters, 64, 156]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=base_filters,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        # BatchNorm2d normalizes across the (B, H, W) dimensions for each of
        # the `base_filters` channels independently. It has 2 learnable
        # parameters per channel (gamma scale + beta shift) so the network
        # can undo the normalization if it's harmful.

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 2 -----
        # Input:  [B, base_filters, 64, 156]
        # Conv preserves spatial dims; pool halves them.
        # After pool: floor(64/2)=32, floor(156/2)=78
        # Output: [B, base_filters*2, 32, 78]
        self.conv2 = nn.Conv2d(in_channels=base_filters,
                               out_channels=base_filters * 2,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 3 -----
        # Input:  [B, base_filters*2, 32, 78]
        # After pool: floor(32/2)=16, floor(78/2)=39
        # Output: [B, base_filters*4, 16, 39]
        self.conv3 = nn.Conv2d(in_channels=base_filters * 2,
                               out_channels=base_filters * 4,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Block 4 (FINAL conv block — Grad-CAM target) -----
        # Input:  [B, base_filters*4, 16, 39]
        # After pool: floor(16/2)=8, floor(39/2)=19
        # Output: [B, base_filters*8, 8, 19]
        #
        # NOTE: We deliberately keep `self.conv4` as a NAMED attribute (not
        # buried inside nn.Sequential) so that get_final_conv_layer() can
        # return a stable handle to it. Grad-CAM needs to attach forward and
        # backward hooks to THIS specific layer to capture activations and
        # gradients used in the CAM computation.
        self.conv4 = nn.Conv2d(in_channels=base_filters * 4,
                               out_channels=base_filters * 8,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Shared activation function. ReLU has no learnable parameters, so we
        # can reuse a single instance across all blocks instead of creating
        # four identical copies.
        # ReLU: f(x) = max(0, x). Cheap, sparse, and avoids vanishing
        # gradients better than tanh/sigmoid.
        self.relu = nn.ReLU(inplace=True)
        # inplace=True saves a small amount of memory by overwriting the input
        # tensor instead of allocating a new one. Safe here because we don't
        # need the pre-ReLU values for the backward pass (BatchNorm has
        # already saved them).

        # ====================================================================
        # CLASSIFIER HEAD — Global Average Pooling architecture
        # ====================================================================
        # Pipeline: GlobalAvgPool → Flatten → Dropout → Linear → logits
        #
        # WHY GLOBAL AVERAGE POOLING (GAP) INSTEAD OF FLATTEN + DENSE?
        #
        #   The naive approach is to Flatten the (B, 128, 8, 19) feature map
        #   into a (B, 19456) vector and feed it into a large Linear layer.
        #   With a 128-unit hidden layer, that's 19456 * 128 ≈ 2.49 MILLION
        #   parameters in a single layer — more than the entire conv stack!
        #
        #   For our small MIMII dataset this is a textbook overfitting trap:
        #   the model has more than enough capacity to memorize every training
        #   spectrogram, and validation accuracy will collapse.
        #
        #   GAP solves this by taking the SPATIAL AVERAGE of each channel's
        #   feature map, collapsing (B, 128, 8, 19) directly to (B, 128, 1, 1).
        #   Each channel becomes a single number summarizing "how strongly is
        #   THIS feature present anywhere in the input?"
        #
        #   Three concrete wins:
        #
        #     1. PARAMETER REDUCTION — drastic.
        #        Flatten + FC(128 hidden):  ≈ 2,500,000 trainable params
        #        GAP + FC(2 logits):                 258 trainable params
        #        That's a ~10,000x reduction in head parameters, removing the
        #        biggest overfitting risk in the model by far.
        #
        #     2. STRONG REGULARIZATION.
        #        Averaging is a structural prior — the model can't memorize
        #        which spatial cells light up, only how much each feature
        #        type fires overall. Anomaly signatures should appear as
        #        elevated channel responses regardless of WHERE in the
        #        spectrogram they occur, so averaging reinforces translation
        #        invariance and forces the conv layers to learn semantically
        #        meaningful features (not pixel-position memorization).
        #
        #     3. INPUT-LENGTH FLEXIBILITY.
        #        The Flatten approach hard-codes the spatial output size into
        #        the FC layer's weight matrix — feed a 12-second clip instead
        #        of 10s and you get a shape mismatch error. GAP averages over
        #        whatever spatial size the conv stack produces, so the same
        #        trained model accepts spectrograms of any length without
        #        re-training. This will matter when we experiment with
        #        TARGET_DURATION_SEC in preprocess.py.
        #
        #   GAP is the standard choice in modern CNN architectures (ResNet,
        #   Inception, MobileNet, EfficientNet) for exactly these reasons.

        # AdaptiveAvgPool2d((1, 1)) tells PyTorch: "however big the spatial
        # dimensions are, average-pool them down to a 1x1 grid." This works
        # for any input size — that's the "adaptive" part. Output shape:
        #   (B, base_filters*8, H_final, W_final) → (B, base_filters*8, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # After GAP, our feature vector length is exactly the channel count
        # of the final conv layer — no spatial multiplication needed. With
        # base_filters=16 this is 128; with base_filters=32 it's 256, etc.
        flattened_size = base_filters * 8

        # nn.Flatten strips the trailing 1x1 spatial dims:
        #   (B, base_filters*8, 1, 1) → (B, base_filters*8)
        # It's parameter-free; we keep it for clarity over a manual .view()/.squeeze().
        self.flatten = nn.Flatten()

        # WHY KEEP DROPOUT EVEN AFTER GAP?
        #   GAP slashes the parameter count but doesn't eliminate overfitting
        #   risk in the conv layers themselves. Dropout on the pooled feature
        #   vector regularizes the final logit computation, randomly zeroing
        #   50% of channel summaries each training step. This forces the
        #   model not to over-rely on any single feature channel — useful
        #   when background noise might cause one channel to spuriously
        #   correlate with class labels.
        #
        #   At eval time Dropout is automatically disabled and all neurons
        #   fire (their outputs implicitly scaled to match training-time
        #   expected magnitudes).
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Single Linear layer: pooled features → raw class logits.
        # With base_filters=16 and num_classes=2, this is just 128*2 + 2 = 258
        # parameters — a tiny, easily-trainable head.
        #
        # We output RAW logits (no softmax) because nn.CrossEntropyLoss
        # combines log-softmax + NLL in one numerically stable operation.
        # Adding our own softmax would double-apply it and silently break
        # gradients.
        self.fc_final = nn.Linear(in_features=flattened_size,
                                  out_features=num_classes)

    # ------------------------------------------------------------------------
    # INTERNAL HELPER: forward pass through ONLY the conv stack.
    # Used both by forward() and by __init__ (to measure output shape).
    # ------------------------------------------------------------------------
    def _forward_conv(self, x):
        """
        Run the input through all four convolutional blocks and return the
        feature map BEFORE flattening. Kept as a separate method so we can
        reuse it for shape-inference at construction time.

        Parameters
        ----------
        x : torch.Tensor, shape [B, 1, n_mels, time_frames]
            A batch of single-channel spectrograms.

        Returns
        -------
        x : torch.Tensor, shape [B, base_filters*8, H_final, W_final]
            The deepest convolutional feature map (after pool4).
        """
        # Block 1: Conv → BN → ReLU → Pool
        # Each line transforms the tensor; comments show the post-op shape
        # for the default config (n_mels=128, time_frames=313, base_filters=16).
        x = self.conv1(x)   # [B, 16, 128, 313]
        x = self.bn1(x)     # [B, 16, 128, 313]  (shape unchanged by BN)
        x = self.relu(x)    # [B, 16, 128, 313]  (shape unchanged by ReLU)
        x = self.pool1(x)   # [B, 16,  64, 156]

        # Block 2
        x = self.conv2(x)   # [B, 32,  64, 156]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)   # [B, 32,  32,  78]

        # Block 3
        x = self.conv3(x)   # [B, 64,  32,  78]
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)   # [B, 64,  16,  39]

        # Block 4 (final conv — Grad-CAM target)
        x = self.conv4(x)   # [B, 128, 16,  39]
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)   # [B, 128,  8,  19]

        return x

    # ------------------------------------------------------------------------
    # PUBLIC FORWARD: full pipeline conv stack → classifier head.
    # PyTorch calls this when you do `logits = model(spectrograms)`.
    # ------------------------------------------------------------------------
    def forward(self, x):
        """
        Full forward pass: spectrogram batch → class logits.

        Parameters
        ----------
        x : torch.Tensor, shape [B, 1, n_mels, time_frames]
            A batch of single-channel Log-Mel-Spectrograms.

        Returns
        -------
        logits : torch.Tensor, shape [B, num_classes]
            Raw (un-softmaxed) class scores. Pass directly into
            nn.CrossEntropyLoss for training, or apply torch.softmax at
            inference time if you want probabilities.
        """
        # Step 1: extract spatial features through the conv stack.
        x = self._forward_conv(x)
        # Shape now: [B, base_filters*8, H_final, W_final]
        # With defaults: [B, 128, 8, 19]

        # Step 2: collapse spatial dimensions to 1x1 via Global Average Pool.
        # Each channel becomes its mean over the entire spatial map — a single
        # number summarizing how strongly that feature fires anywhere in the
        # input. This is the regularization step that prevents overfitting.
        x = self.global_pool(x)
        # Shape now: [B, base_filters*8, 1, 1]

        # Step 3: drop the trailing 1x1 dims to get a clean feature vector.
        x = self.flatten(x)
        # Shape now: [B, base_filters*8]

        # Step 4: dropout regularization on the pooled feature vector.
        x = self.dropout1(x)

        # Step 5: linear projection to class logits. No activation afterwards —
        # CrossEntropyLoss applies log-softmax internally during training.
        logits = self.fc_final(x)
        # Shape: [B, num_classes]

        return logits

    # ------------------------------------------------------------------------
    # GRAD-CAM SUPPORT
    # ------------------------------------------------------------------------
    def get_final_conv_layer(self):
        """
        Return a reference to the LAST convolutional layer (self.conv4).

        WHY THE LAST CONV LAYER?
            Grad-CAM works by combining (a) the activations of the chosen
            layer on a forward pass with (b) the gradients of the predicted
            class with respect to those same activations. The deepest conv
            layer is preferred because it captures the most semantically
            meaningful, high-level features while still preserving spatial
            structure (linear/FC layers destroy spatial layout). Using
            conv4's output gives heatmaps localized to specific regions of
            the original spectrogram.

        USAGE (from src/explain.py):
            target_layer = model.get_final_conv_layer()
            cam = GradCAM(model=model, target_layers=[target_layer])

        Returns
        -------
        nn.Conv2d
            A direct reference to self.conv4 (NOT a copy — the caller can
            register hooks on it and they'll fire during real forward passes).
        """
        return self.conv4


# ============================================================================
# ENTRY POINT — Smoke test the architecture
# ============================================================================
# Running `python src/model.py` builds the model, runs a dummy batch through
# it, and prints the layer-by-layer shape transformations. This is a fast
# sanity check that the architecture wires up correctly before integrating
# with the training loop.

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL SMOKE TEST — AudioClassifier")
    print("=" * 60)

    # Instantiate with default hyperparameters.
    model = AudioClassifier(base_filters=16)

    # Build a dummy batch matching the DataLoader's output shape:
    # batch_size=4 (small for a quick test), 1 channel, 128 mels, 313 frames.
    dummy_batch = torch.randn(4, 1, DEFAULT_N_MELS, DEFAULT_TIME_FRAMES)
    print(f"\n  Input shape:   {tuple(dummy_batch.shape)}")

    # Run the dummy batch through the model (eval mode disables dropout for
    # a clean shape readout — we'll see deterministic output sizes).
    model.eval()
    with torch.no_grad():
        logits = model(dummy_batch)
    print(f"  Output shape:  {tuple(logits.shape)}  (expected: (4, 2))")

    # Print parameter count as a sanity check on model size.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Verify the Grad-CAM hook target exists and is the right type.
    final_conv = model.get_final_conv_layer()
    print(f"\n  Final conv layer (Grad-CAM target): {final_conv}")
    print(f"    type:         {type(final_conv).__name__}")
    print(f"    in_channels:  {final_conv.in_channels}")
    print(f"    out_channels: {final_conv.out_channels}")

    print("\n" + "=" * 60)
    print("MODEL OK — ready for training")
    print("=" * 60)
