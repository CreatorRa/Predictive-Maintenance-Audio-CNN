"""
test_model.py — Unit tests for src/model.py
============================================================
Two contracts in model.py are sacred:

    1. The standard input shape [B, 1, 128, 313] must produce output
       shape [B, num_classes]. This is the integration point with the
       DataLoader (which always emits 128x313) and with CrossEntropyLoss
       (which always expects [B, num_classes] logits).

    2. The model must accept variable-length spectrograms WITHOUT
       crashing. This is the entire reason we chose Global Average
       Pooling over Flatten + Dense in the head — GAP collapses any
       spatial size to (1, 1), so a 12-second clip should run through
       the same trained weights as a 10-second clip.

WHAT FAILURES THESE TESTS PROTECT AGAINST:
    - A "cleanup" refactor that replaces self.global_pool with
      nn.Flatten + a hard-coded Linear(19456, 128). Forward pass on
      [2, 1, 128, 313] would still work, but [2, 1, 128, 800] would
      raise a RuntimeError about shape mismatch. The variable-length
      test is the canary that catches this.
    - Someone accidentally hard-codes time_frames into the Conv stack
      via a non-padding kernel — the standard-shape test catches that.
    - Forgetting to call super().__init__() in a model subclass — the
      forward pass would error on missing parameter registration.
    - A regression that adds softmax inside forward() — output shape
      would still be [B, 2] but values would be in [0, 1] instead of
      raw logits, silently double-applying softmax with CrossEntropyLoss.

DESIGN PRINCIPLE — NO DISK I/O, NO CHECKPOINTS:
    Every test instantiates a fresh AudioClassifier with random weights
    and runs a forward pass on a `torch.randn` dummy tensor. We never
    touch the filesystem, never load a checkpoint, never use a real
    DataLoader. Tests run in milliseconds on CPU.
"""

import torch

# Imports rely on tests/conftest.py inserting <repo>/src into sys.path.
from model import AudioClassifier


# ============================================================================
# SECTION 1 — STANDARD INPUT SHAPE CONTRACT
# ============================================================================

def test_forward_pass_standard_shape_produces_correct_logits():
    """
    PROTECTS AGAINST:
        - A regression that breaks the [B, 1, 128, 313] → [B, 2]
          contract — the most fundamental integration assumption in
          the entire pipeline.
        - A refactor that accidentally adds softmax to forward() — the
          output shape would still pass, but the assertion below on
          values being unbounded raw logits would fail.
        - Forgetting to register conv layers as named attributes,
          which would silently produce a model with no learnable
          parameters and forward() would not crash.
    """
    # eval() to disable Dropout. Dropout doesn't change output shape, but
    # disabling it makes the test 100% deterministic (no randomness in
    # the forward pass).
    model = AudioClassifier(base_filters=16)
    model.eval()

    # Build the canonical DataLoader-style dummy batch:
    #   batch=2, channels=1, mels=128, time_frames=313
    # batch=2 (not 1) catches a class of bugs where the model accidentally
    # squeezes the batch dim — those bugs are invisible with batch=1.
    dummy_input = torch.randn(2, 1, 128, 313)

    # torch.no_grad() because this is a shape test, not a training test.
    # Skips the autograd graph construction and runs faster.
    with torch.no_grad():
        logits = model(dummy_input)

    # --- The headline assertion ---
    # Shape MUST be exactly [2, 2]: 2 samples × 2 class logits.
    # We use tuple comparison (not torch.Size equality) for a clearer
    # failure message when this breaks.
    assert tuple(logits.shape) == (2, 2), (
        f"Expected output shape (2, 2), got {tuple(logits.shape)}"
    )

    # --- Sanity: outputs are RAW LOGITS, not probabilities ---
    # If a regression adds softmax inside forward(), every value falls
    # in [0, 1] and rows sum to 1.0. We probe that by checking the row
    # sum is NOT close to 1.0 (with random weights and inputs, the raw
    # logits should sum to something far from unity — about 99.99% of
    # the time on a healthy network).
    row_sums = logits.sum(dim=1)
    # Each row sum is a 1-element tensor. Probabilities would force
    # row_sum == 1.0 exactly. Random logits are essentially never that.
    # We assert that AT LEAST ONE row is meaningfully far from 1.0,
    # which is a robust signal that softmax was NOT applied.
    assert torch.any(torch.abs(row_sums - 1.0) > 0.1), (
        "Output rows sum suspiciously close to 1.0 — has softmax been "
        "added inside forward()? CrossEntropyLoss expects raw logits."
    )


def test_forward_pass_dtype_is_float32():
    """
    PROTECTS AGAINST:
        Someone changes a layer to .double() or returns a float64
        tensor. Mixing dtypes between model output and loss target is
        a classic source of silent CPU<->GPU transfer slowdowns and
        cryptic dtype errors during backward().
    """
    model = AudioClassifier(base_filters=16).eval()
    dummy_input = torch.randn(2, 1, 128, 313)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.dtype == torch.float32, (
        f"Expected float32 logits, got {logits.dtype}"
    )


def test_grad_cam_target_layer_is_a_conv2d():
    """
    PROTECTS AGAINST:
        A refactor that buries conv4 inside an nn.Sequential and
        forgets to update get_final_conv_layer to point at the new
        location. Grad-CAM would then hook into a Sequential, which
        does not have the activation tensor we need, and the heatmap
        would silently be all zeros.
    """
    model = AudioClassifier(base_filters=16)
    target = model.get_final_conv_layer()

    # The target MUST be an actual Conv2d — not a Sequential, not a
    # ModuleList, not the Linear head. Hooking the wrong layer type is
    # one of the most common Grad-CAM failure modes.
    assert isinstance(target, torch.nn.Conv2d), (
        f"get_final_conv_layer should return nn.Conv2d, got {type(target).__name__}"
    )

    # And it should be the LAST conv layer specifically: out_channels
    # equals base_filters * 8. If someone returns conv1 by mistake the
    # heatmap will technically render but show low-level edge features
    # instead of class-discriminative ones.
    assert target.out_channels == 16 * 8, (
        f"Final conv should have base_filters*8={16*8} out_channels, "
        f"got {target.out_channels} — get_final_conv_layer may be "
        f"returning the wrong layer."
    )


# ============================================================================
# SECTION 2 — VARIABLE-LENGTH INPUT (the GAP guarantee)
# ============================================================================

def test_forward_pass_accepts_longer_time_axis():
    """
    PROTECTS AGAINST:
        The most damaging refactor we could make to model.py: replacing
        Global Average Pooling with `nn.Flatten() → nn.Linear(19456, ...)`.
        Such a change would:
          - Not crash on the standard 313-frame input (so other tests
            wouldn't catch it).
          - Silently fail on any spectrogram with a different time-frame
            count, raising a RuntimeError about shape mismatch.
          - Add ~2.5 MILLION parameters to the head, blowing up the
            model's overfitting risk.

        This test is the SOLE guarantee in the project that the GAP
        head stays in place. If the test passes, the model is still
        spatial-size agnostic. If it fails, the GAP guarantee has been
        broken and every downstream piece of "this model accepts any
        clip length" reasoning is invalid.

    SETUP:
        Feed an 800-frame input (about 25 seconds of audio at our
        default hop length) instead of the standard 313 frames.
    """
    model = AudioClassifier(base_filters=16).eval()

    # Same batch and mel dims as the standard test, but the time axis
    # is 800 instead of 313. With base_filters=16 and 4 pooling stages,
    # the conv stack reduces the time dim to floor(800/16) = 50 — a
    # totally non-default size that hard-coded shapes would reject.
    dummy_input = torch.randn(2, 1, 128, 800)

    with torch.no_grad():
        # If GAP has been removed, this line raises a RuntimeError about
        # mat1/mat2 shape mismatch in the Linear head. With GAP intact,
        # the spatial dims collapse to (1, 1) regardless of input size
        # and the Linear receives the same (B, 128) feature vector it
        # always sees.
        logits = model(dummy_input)

    # Output shape must STILL be (B, num_classes), independent of input
    # time-axis length. This is the whole point of GAP.
    assert tuple(logits.shape) == (2, 2), (
        f"Variable-length input should still yield (2, 2) logits; "
        f"got {tuple(logits.shape)}. The Global Average Pooling layer "
        f"may have been replaced with a hard-coded dense head."
    )


def test_forward_pass_accepts_shorter_time_axis():
    """
    PROTECTS AGAINST:
        The same regression as the longer-input test, but in the
        opposite direction. A naive Flatten head sized for 313 frames
        would also reject anything SHORTER. We test both directions
        because some bugs only manifest in one direction (e.g.,
        slicing operations that silently truncate but raise on
        upsampling).

    SETUP:
        160 time frames — about half the default duration. After 4
        rounds of MaxPool2d(2): floor(160/16) = 10, still a valid
        non-default spatial size.
    """
    model = AudioClassifier(base_filters=16).eval()
    dummy_input = torch.randn(2, 1, 128, 160)

    with torch.no_grad():
        logits = model(dummy_input)

    assert tuple(logits.shape) == (2, 2), (
        f"Shorter-than-default input should still yield (2, 2) logits; "
        f"got {tuple(logits.shape)}."
    )
