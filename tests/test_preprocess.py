"""
test_preprocess.py — Unit tests for src/preprocess.py
============================================================
This module pins down the two PROPERTIES that every downstream stage
silently relies on:

    1. After load_and_normalize_audio runs, the waveform is EXACTLY
       TARGET_LENGTH_SAMPLES long — no off-by-one, no librosa-resample
       drift, regardless of whether the original audio was longer,
       shorter, or already perfect length.

    2. After generate_log_mel_spectrogram runs, the output is a 2D
       float array of shape (N_MELS, ~313). The CNN's first Conv2d is
       hard-coded to in_channels=1 with the 128-mel input assumed, and
       SpectrogramDataset.__getitem__ does an unsqueeze(0) — both will
       silently produce wrong-shape tensors if N_MELS or the time-frame
       count drifts.

WHAT FAILURES THESE TESTS PROTECT AGAINST:
    - A future "small refactor" of load_and_normalize_audio that uses
      np.resize (which mod-wraps) instead of slice/pad → would silently
      produce non-zero-padded clips, training would converge to a model
      that has memorized the wrap-around artifact.
    - Someone changes N_MELS in preprocess.py without updating
      DEFAULT_N_MELS in model.py → the test catches the mismatch
      immediately; without the test, the bug only surfaces 30 min into
      training as a cryptic Conv shape error.
    - A librosa upgrade that changes the default centering / framing
      logic → the time-frame count drifts from 313 to 312 or 314,
      breaking saved checkpoints and any hard-coded shape assumptions.
    - The off-by-one classic: load_and_normalize_audio returns
      target_length - 1 samples → DataLoader.collate would fail
      mysteriously on batching.

DESIGN PRINCIPLE — NO DISK I/O:
    Every test fabricates a numpy array in memory and feeds it directly
    to the public functions. We deliberately do NOT write a temp .wav
    file and call load_and_normalize_audio on a real path, because:
      - Disk I/O is slow (ms vs μs) and flaky (permissions, antivirus).
      - librosa.load needs ffmpeg/soundfile decoders which may differ
        across CI environments — a test that passes locally but fails
        on Colab is worse than no test.
      - The padding/truncation logic is what we actually care about, and
        it operates on a numpy array AFTER librosa.load returns. Mocking
        librosa.load lets us test the logic that's ours, not librosa's.
"""

import numpy as np
from unittest.mock import patch

# These imports work because tests/conftest.py inserts <repo>/src into
# sys.path before any test runs.
from preprocess import (
    load_and_normalize_audio,
    generate_log_mel_spectrogram,
    SAMPLE_RATE,
    TARGET_LENGTH_SAMPLES,
    N_MELS,
    HOP_LENGTH,
)


# ============================================================================
# SECTION 1 — PADDING / TRUNCATION INVARIANT
# ============================================================================
# Every test in this section mocks librosa.load to return a fake waveform of
# a chosen length. We then assert load_and_normalize_audio produced exactly
# TARGET_LENGTH_SAMPLES samples.
#
# We use unittest.mock.patch (not a real file) so the test runs in microseconds
# and can't be broken by librosa version drift.

def _fake_audio(n_samples, seed=0):
    """
    Build a deterministic fake waveform of the requested length.

    Why deterministic (seeded) noise?
        Reproducible test output makes failures debuggable — a flaky test
        that occasionally passes is worse than no test at all.
    """
    rng = np.random.default_rng(seed)
    # float32 matches what librosa.load returns; mismatched dtypes have
    # caused real bugs in this project before (silent precision loss).
    return rng.standard_normal(n_samples).astype(np.float32)


@patch("preprocess.librosa.load")
def test_truncation_produces_exact_target_length(mock_load):
    """
    PROTECTS AGAINST:
        A regression where truncation slicing uses the wrong index
        (e.g. audio[:target_length-1] off-by-one) or where someone
        replaces the slice with np.resize (which would wrap, not truncate).
    """
    # Build a fake waveform that is 50% LONGER than the target. This forces
    # the truncation branch of load_and_normalize_audio to fire.
    too_long = _fake_audio(int(TARGET_LENGTH_SAMPLES * 1.5))
    mock_load.return_value = (too_long, SAMPLE_RATE)

    # The path argument is irrelevant — librosa.load is mocked, so it never
    # touches the filesystem. We pass a sentinel string for clarity.
    audio = load_and_normalize_audio("FAKE_PATH.wav")

    # The single property under test: exact length. NOT "at most" or
    # "approximately" — the CNN's first Conv layer would silently behave
    # differently on a 159999-sample input vs a 160000-sample input.
    assert audio.shape == (TARGET_LENGTH_SAMPLES,), (
        f"Truncation should yield exactly {TARGET_LENGTH_SAMPLES} samples, "
        f"got shape {audio.shape}"
    )


@patch("preprocess.librosa.load")
def test_padding_produces_exact_target_length(mock_load):
    """
    PROTECTS AGAINST:
        A regression where np.pad is called with the wrong tuple
        (e.g. (deficit, 0) padding the START instead of the end), or
        where the deficit calculation rounds incorrectly.
    """
    # Half-length input forces the zero-padding branch.
    too_short = _fake_audio(TARGET_LENGTH_SAMPLES // 2)
    mock_load.return_value = (too_short, SAMPLE_RATE)

    audio = load_and_normalize_audio("FAKE_PATH.wav")

    assert audio.shape == (TARGET_LENGTH_SAMPLES,), (
        f"Padding should yield exactly {TARGET_LENGTH_SAMPLES} samples, "
        f"got shape {audio.shape}"
    )

    # Verify the padding went on the END (not the start). A regression where
    # the pad tuple is reversed is silent and corrupts every batch — it would
    # turn every clip into "silence then audio," teaching the CNN that
    # "abnormal" means "audio early in the clip." This assertion is the
    # canary for that exact failure mode.
    second_half_is_silence = np.all(audio[TARGET_LENGTH_SAMPLES // 2:] == 0.0)
    assert second_half_is_silence, (
        "Zero-padding should be appended to the END of the waveform; "
        "trailing samples were not all zero."
    )


@patch("preprocess.librosa.load")
def test_exact_length_passes_through_untouched(mock_load):
    """
    PROTECTS AGAINST:
        A regression that always pads or always truncates by 1 sample
        regardless of input length. The "no-op" branch is the easiest
        one to break in a refactor because it has no visible side effects.
    """
    perfect = _fake_audio(TARGET_LENGTH_SAMPLES)
    mock_load.return_value = (perfect, SAMPLE_RATE)

    audio = load_and_normalize_audio("FAKE_PATH.wav")

    # Length must still match exactly.
    assert audio.shape == (TARGET_LENGTH_SAMPLES,)
    # Content must be unchanged — if a refactor accidentally always
    # ran a padding / truncation branch, the array would still have
    # the right LENGTH but wrong VALUES. Catch that by comparing to
    # the input verbatim.
    np.testing.assert_array_equal(audio, perfect)


# ============================================================================
# SECTION 2 — SPECTROGRAM SHAPE INVARIANT
# ============================================================================
# The CNN expects (1, N_MELS, n_time_frames) inputs. If preprocess.py ever
# emits a spectrogram with the wrong number of mel bands or wildly different
# time-frame count, the entire pipeline breaks. These tests pin the contract.

def test_spectrogram_shape_matches_n_mels_and_expected_time_frames():
    """
    PROTECTS AGAINST:
        - N_MELS being changed in preprocess.py without updating
          DEFAULT_N_MELS in model.py.
        - A librosa version bump that changes the default centering
          convention and shifts the time-frame count by 1.
        - generate_log_mel_spectrogram accidentally returning the
          POWER spectrogram instead of the dB-scaled one (different
          shape would NOT catch this — the dtype/range test below does).
    """
    # Generate the spectrogram from a target-length fake waveform —
    # this is exactly what load_and_normalize_audio would have produced.
    fake_audio = _fake_audio(TARGET_LENGTH_SAMPLES)
    spec = generate_log_mel_spectrogram(fake_audio)

    # --- Output must be a 2D array. ---
    # If someone refactors and forgets to drop a singleton dim, the CNN
    # will reject the resulting (1, 128, 313) array later.
    assert spec.ndim == 2, f"Spectrogram should be 2D, got {spec.ndim}D"

    # --- Row count must equal N_MELS exactly. ---
    # This is the strictest invariant in the file — every layer in the
    # CNN was sized assuming 128 mel bands.
    assert spec.shape[0] == N_MELS, (
        f"First axis should be N_MELS={N_MELS}, got {spec.shape[0]}"
    )

    # --- Column count must equal the EXACT formula from preprocess.py. ---
    # librosa with default center=True produces:
    #     n_frames = floor(n_samples / hop_length) + 1
    # We assert against this formula (not the literal 313) so the test
    # remains correct if TARGET_DURATION_SEC or HOP_LENGTH are tuned
    # later — only an actual contract violation triggers the failure.
    expected_time_frames = TARGET_LENGTH_SAMPLES // HOP_LENGTH + 1
    assert spec.shape[1] == expected_time_frames, (
        f"Second axis should be {expected_time_frames} time frames "
        f"(N_MELS={N_MELS}, HOP_LENGTH={HOP_LENGTH}), got {spec.shape[1]}"
    )

    # With our defaults this works out to (128, 313). Make that explicit
    # so a developer reading the assertion knows what "correct" looks like.
    assert spec.shape == (128, 313), (
        f"Default config should produce (128, 313); got {spec.shape}. "
        f"If you intentionally changed preprocess constants, update this test."
    )


def test_spectrogram_is_dB_scaled_not_raw_power():
    """
    PROTECTS AGAINST:
        Someone deletes the librosa.power_to_db call — the shape would
        still be correct, but the values would span 1e-10 to 1e+3 instead
        of roughly [-80, 0] dB. Training would fail to converge with
        no obvious shape error, and the bug would take hours to find.
    """
    fake_audio = _fake_audio(TARGET_LENGTH_SAMPLES)
    spec = generate_log_mel_spectrogram(fake_audio)

    # power_to_db with ref=np.max produces values <= 0. A raw power
    # spectrogram would have all-positive values. This is the cleanest
    # one-line proof that the dB conversion ran.
    assert spec.max() <= 0.0 + 1e-6, (
        f"Log-Mel-Spectrogram should be in dB (max <= 0), got max={spec.max()}. "
        f"Did someone remove the power_to_db call?"
    )

    # And a sanity floor — dB values shouldn't go below ~-100 in practice
    # for our normalized fake audio. If they do, ref= was likely changed
    # to a tiny constant by mistake.
    assert spec.min() > -200.0, (
        f"dB floor unexpectedly low ({spec.min()}); ref= argument may "
        f"have been changed."
    )
