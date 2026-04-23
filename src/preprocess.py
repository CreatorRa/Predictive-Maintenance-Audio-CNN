"""
preprocess.py — Audio-to-Spectrogram Preprocessing Pipeline
============================================================
This script is the first stage of our Predictive Maintenance Audio CNN project.
Its job is to convert raw .wav audio files into Log-Mel-Spectrograms and save them
as NumPy arrays (.npy files). These arrays will later be fed into our CNN as if they
were single-channel "images" of sound.

WHY MEL-SPECTROGRAMS?
    Raw audio is a 1D waveform (amplitude over time). CNNs, however, excel at
    finding spatial patterns in 2D data (like images). A Mel-spectrogram bridges
    this gap: it transforms audio into a 2D matrix where one axis is time, the other
    is frequency (in Mel scale), and the cell values represent energy/loudness.
    The Mel scale warps frequency to match human hearing — we perceive the difference
    between 200 Hz and 400 Hz as much larger than between 8000 Hz and 8200 Hz. By
    using the Mel scale, we give the CNN a representation that emphasizes the
    frequency differences that matter most for distinguishing normal vs. abnormal
    machine sounds.

WHY LOG (DECIBEL) SCALING?
    Raw spectrogram values span a huge dynamic range (e.g. 0.0001 to 1000). Taking
    the log (converting to decibels) compresses this range, making quiet details
    visible alongside loud ones, which is much like how our ears perceive loudness on a
    logarithmic scale. Without this step, quiet but important anomaly signatures
    could be drowned out by dominant frequency bands.

WHY FIXED-LENGTH PADDING/TRUNCATION?
    Neural networks require fixed-size inputs. Real-world audio clips may vary in
    duration. We enforce a consistent length by either:
      - Truncating clips that are too long (cutting from the end), or
      - Zero-padding clips that are too short (appending silence to the end).
    This guarantees every spectrogram matrix has the exact same shape (n_mels x n_time_frames),
    which is required for batching tensors in PyTorch.

USAGE:
    Run this script directly from the command line:
        python src/preprocess.py

    Or import individual functions into another script/notebook:
        from src.preprocess import generate_log_mel_spectrogram
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os          # For filesystem operations (building paths, listing directories)
import glob        # For pattern-matching file paths (e.g., find all .wav files)
import numpy as np # NumPy: the foundation for numerical arrays in Python
import librosa     # Librosa: the go-to library for audio analysis in Python


# ============================================================================
# CONFIGURATION — All tunable hyperparameters live here in one place.
# ============================================================================
# We define these as module-level constants so they're easy to find and adjust.
# When we move to Colab for hyperparameter tuning, we can override these values
# without hunting through function bodies.

SAMPLE_RATE = 16000
# The number of audio samples per second. 16 kHz is standard for industrial/
# environmental audio — it captures frequencies up to 8 kHz (by the Nyquist
# theorem: max_freq = sample_rate / 2), which covers the range where most
# machine anomalies manifest. Higher rates (e.g. 44.1 kHz for music) would
# add unnecessary data and slow training without improving anomaly detection.

TARGET_DURATION_SEC = 10
# The fixed duration (in seconds) that every audio clip will be normalized to.
# Clips shorter than this get zero-padded; clips longer get truncated.
# 10 seconds is a reasonable window to capture repetitive machine cycles. This value should be tuned based on the specific MIMII machine type you're working with.

TARGET_LENGTH_SAMPLES = SAMPLE_RATE * TARGET_DURATION_SEC
# The fixed length in raw samples. At 16 kHz for 10 seconds, this equals 160,000 samples. 
# Every waveform will be exactly this many samples long before we compute the spectrogram.

N_FFT = 2048
# The size of the FFT (Fast Fourier Transform) window in samples.
# This controls the frequency resolution of our spectrogram.
#   - Larger N_FFT → better frequency resolution (can distinguish closely
#     spaced frequencies) but worse time resolution (events blur together).
#   - Smaller N_FFT → better time resolution but worse frequency resolution.
# 2048 is a widely-used default that balances both. At 16 kHz sample rate,
# each FFT window spans 2048/16000 = 0.128 seconds of audio.

HOP_LENGTH = 512
# The number of samples between successive FFT windows (i.e., the "stride").
# This determines the time resolution of the spectrogram's horizontal axis.
#   - HOP_LENGTH = N_FFT / 4 is a common choice (75% overlap between windows),
#     which provides smooth time resolution without excessive redundancy.
#   - Smaller hop → more time frames → larger spectrogram → more computation.
# With hop_length=512 at 16 kHz, we get ~31.25 frames per second of audio.

N_MELS = 128
# The number of Mel-frequency bands (i.e., the height of our spectrogram).
# This is the vertical resolution of our "image."
#   - 128 is a standard choice that provides fine-grained frequency detail.
#   - 64 is a common alternative if you want smaller inputs / faster training.
#   - More bands capture subtler frequency distinctions but increase input size.
# Each of these 128 bands covers a different slice of the Mel-scaled frequency
# axis, with narrower bands at low frequencies (where our ears are sensitive)
# and wider bands at high frequencies.

# --- Filesystem Paths ---
# These paths define where to read raw audio and where to write processed data.
# They are relative to the project root (the directory you run the script from).

RAW_DATA_DIR = os.path.join("data", "raw")
# Parent directory containing the class subfolders (normal/, abnormal/).

PROCESSED_DATA_DIR = os.path.join("data", "processed")
# Parent directory where we'll write the output .npy files, mirroring the
# same subfolder structure (processed/normal/, processed/abnormal/).

CLASS_LABELS = ["normal", "abnormal"]
# The two class directories we expect inside data/raw/. These also become
# the subfolder names inside data/processed/. Using a list makes it trivial
# to add more classes later if the project scope expands.


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_and_normalize_audio(file_path, sample_rate=SAMPLE_RATE,
                             target_length=TARGET_LENGTH_SAMPLES):
    """
    Load a .wav file from disk, resample it to a consistent sample rate,
    and pad or truncate it to a fixed number of samples.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a .wav audio file.
    sample_rate : int
        The target sample rate in Hz. Librosa will resample the audio to this
        rate regardless of the file's original rate. Default: 16000.
    target_length : int
        The exact number of samples the output waveform must have.
        Default: 160000 (10 seconds at 16 kHz).

    Returns
    -------
    audio : np.ndarray, shape (target_length,)
        A 1D NumPy array of float32 audio samples, guaranteed to be exactly
        `target_length` samples long.
    """

    # --- Step 1: Load the audio file ---
    # librosa.load() reads the .wav file and returns:
    #   - audio: a 1D NumPy array of float32 samples (amplitude values between ~-1 and 1)
    #   - sr: the sample rate of the returned audio (will match our requested `sr`)
    #
    # Key behaviors:
    #   - sr=sample_rate forces resampling to our target rate. If the original file
    #     was recorded at 44.1 kHz, librosa automatically downsamples to 16 kHz.
    #   - mono=True mixes multi-channel audio down to a single channel. Industrial
    #     recordings are often mono anyway, but this ensures consistency.
    #   - The returned dtype is float32 with amplitudes normalized to roughly [-1, 1].
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)

    # --- Step 2: Pad or truncate to the fixed target length ---
    # We need every single audio clip to produce a waveform of exactly `target_length`
    # samples. This is non-negotiable because:
    #   1. The spectrogram dimensions are determined by the waveform length.
    #   2. PyTorch's DataLoader requires all tensors in a batch to have the same shape.
    #   3. Inconsistent shapes would cause a runtime error during training.

    current_length = len(audio)
    # current_length is the number of samples librosa actually decoded from the file.

    if current_length > target_length:
        # TRUNCATION: The clip is longer than our target duration.
        # We simply slice off everything after `target_length` samples.
        # We cut from the END (not the beginning) to preserve the start of the
        # recording, which typically contains the machine's startup or steady-state
        # behavior — both relevant for anomaly detection.
        audio = audio[:target_length]

    elif current_length < target_length:
        # ZERO-PADDING: The clip is shorter than our target duration.
        # We append zeros (silence) to the end until we reach `target_length`.
        # np.pad with mode='constant' and default constant_values=0 does this.
        #
        # Why pad with zeros (silence) rather than, say, repeating the audio?
        #   - Zero-padding is the simplest and most common approach.
        #   - It doesn't fabricate false patterns that could confuse the CNN.
        #   - The CNN can learn to ignore the silent tail region.
        #
        # The padding tuple (0, deficit) means: add 0 samples before, `deficit` after.
        deficit = target_length - current_length
        audio = np.pad(audio, (0, deficit), mode='constant')

    # If current_length == target_length, no action needed — the clip is already
    # the perfect length. We fall through to the return statement.

    return audio


def generate_log_mel_spectrogram(audio, sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                  hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Convert a 1D audio waveform into a 2D Log-Mel-Spectrogram (in decibels).

    This is the heart of our preprocessing pipeline. It transforms raw audio
    into the visual representation that our CNN will learn from.

    Parameters
    ----------
    audio : np.ndarray, shape (n_samples,)
        A 1D array of audio samples (output of load_and_normalize_audio).
    sample_rate : int
        Sample rate of the audio in Hz. Default: 16000.
    n_fft : int
        FFT window size in samples. Controls frequency resolution. Default: 2048.
    hop_length : int
        Stride between successive FFT windows. Controls time resolution. Default: 512.
    n_mels : int
        Number of Mel-frequency bands. Controls vertical resolution. Default: 128.

    Returns
    -------
    log_mel_spec : np.ndarray, shape (n_mels, n_time_frames)
        A 2D matrix representing the Log-Mel-Spectrogram in decibels.
        With default settings and 10s audio at 16 kHz:
            n_mels = 128 (rows, frequency axis)
            n_time_frames ≈ ceil(160000 / 512) + 1 = 313 (columns, time axis)
        So the output shape is approximately (128, 313).
    """

    # --- Step 1: Compute the Mel-scaled spectrogram ---
    # librosa.feature.melspectrogram() performs several operations under the hood:
    #   1. Applies a Short-Time Fourier Transform (STFT) to the audio, breaking
    #      it into overlapping windows and computing the FFT of each window.
    #      This produces a complex-valued spectrogram of shape (1 + n_fft/2, n_time_frames).
    #   2. Takes the magnitude squared (power spectrum) of each FFT bin.
    #   3. Multiplies by a Mel filterbank matrix that maps the linear frequency bins
    #      onto `n_mels` Mel-scaled bands. The filterbank has overlapping triangular
    #      filters spaced according to the Mel scale.
    #
    # The result is a real-valued matrix of shape (n_mels, n_time_frames) where
    # each cell represents the energy in a particular Mel-frequency band at a
    # particular time frame. Higher values = more energy at that frequency/time.
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,            # y: the raw audio waveform (1D array)
        sr=sample_rate,     # sr: sample rate, needed to correctly map Hz → Mel
        n_fft=n_fft,        # n_fft: FFT window size
        hop_length=hop_length,  # hop_length: stride between FFT windows
        n_mels=n_mels       # n_mels: number of output Mel bands
    )

    # --- Step 2: Convert power spectrogram to decibel (log) scale ---
    # librosa.power_to_db() applies the formula:
    #     log_mel_spec = 10 * log10(mel_spectrogram / ref)
    #
    # Where ref=np.max means we use the maximum value in the spectrogram as the
    # reference point. This makes the loudest point = 0 dB and everything else
    # is negative dB (quieter). This is a form of normalization that:
    #   - Compresses the enormous dynamic range of raw power values.
    #   - Makes the spectrogram invariant to overall recording volume.
    #   - Ensures the CNN sees relative loudness patterns, not absolute amplitudes.
    #
    # Without this step, the raw power values might range from 1e-10 to 1e+3,
    # which would make gradient-based learning extremely unstable. The log
    # transform brings values into a manageable range (typically -80 dB to 0 dB).
    log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spec


def process_class_directory(class_label, raw_dir=RAW_DATA_DIR,
                            processed_dir=PROCESSED_DATA_DIR):
    """
    Process all .wav files for a single class (e.g., "normal" or "abnormal").

    This function orchestrates the full pipeline for one class:
    reads every .wav file from the class's raw directory, converts each to a
    Log-Mel-Spectrogram, and saves the result as a .npy file in the
    corresponding processed directory.

    Parameters
    ----------
    class_label : str
        The name of the class subfolder (e.g., "normal" or "abnormal").
        This determines both the input folder (raw_dir/class_label/) and
        the output folder (processed_dir/class_label/).
    raw_dir : str
        Path to the parent raw data directory. Default: "data/raw".
    processed_dir : str
        Path to the parent processed data directory. Default: "data/processed".

    Returns
    -------
    processed_count : int
        The number of files that were successfully processed and saved.
    """

    # --- Build input and output directory paths ---
    # os.path.join handles cross-platform path separators (/ on Linux, \\ on Windows).
    input_dir = os.path.join(raw_dir, class_label)
    output_dir = os.path.join(processed_dir, class_label)

    # --- Create the output directory if it doesn't already exist ---
    # os.makedirs with exist_ok=True is safe to call even if the directory already
    # exists — it won't raise an error or delete existing files.
    os.makedirs(output_dir, exist_ok=True)

    # --- Find all .wav files in the input directory ---
    # glob.glob returns a list of file paths matching the pattern.
    # os.path.join(input_dir, "*.wav") creates a pattern like "data/raw/normal/*.wav".
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    # We sort the file list for deterministic processing order. This makes
    # debugging easier — if something fails on file #47, you can reliably
    # reproduce which file that was.

    # --- Inform the user how many files were found ---
    print(f"[{class_label.upper()}] Found {len(wav_files)} .wav files in '{input_dir}'")

    if len(wav_files) == 0:
        # Warn (don't crash) if the directory is empty or missing .wav files.
        # This is common during local development when you're testing with
        # a micro-batch before downloading the full MIMII dataset.
        print(f"  WARNING: No .wav files found. Skipping '{class_label}' class.")
        return 0

    processed_count = 0  # Track how many files we successfully process.

    # --- Process each .wav file one at a time ---
    for i, wav_path in enumerate(wav_files):

        # Extract just the filename (e.g., "00000000.wav") from the full path.
        # os.path.basename strips the directory prefix.
        filename = os.path.basename(wav_path)

        # Build the output filename by replacing .wav with .npy.
        # os.path.splitext splits "00000000.wav" into ("00000000", ".wav"),
        # and we take the first part (the stem) and append ".npy".
        output_filename = os.path.splitext(filename)[0] + ".npy"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Step A: Load and normalize the audio to a fixed-length waveform.
            audio = load_and_normalize_audio(wav_path)

            # Step B: Convert the fixed-length waveform into a Log-Mel-Spectrogram.
            log_mel_spec = generate_log_mel_spectrogram(audio)

            # Step C: Save the spectrogram as a .npy file.
            # np.save writes a NumPy array to disk in a compact binary format.
            # This is far more efficient than saving as an image (.png) because:
            #   1. No lossy compression — we preserve exact float32 values.
            #   2. No image encoding/decoding overhead.
            #   3. Loading with np.load() is extremely fast.
            #   4. The CNN needs numerical arrays, not images — so saving as
            #      .npy avoids an unnecessary image→array conversion step later.
            np.save(output_path, log_mel_spec)

            processed_count += 1

            # Print progress every 50 files to avoid flooding the console
            # but still give a sense of progress during long runs.
            if (i + 1) % 50 == 0 or (i + 1) == len(wav_files):
                print(f"  Processed {i + 1}/{len(wav_files)}: "
                      f"{filename} → {output_filename} "
                      f"(shape: {log_mel_spec.shape})")

        except Exception as e:
            # If a single file fails (corrupted audio, wrong format, etc.),
            # log the error and continue processing the remaining files.
            # We don't want one bad file to crash the entire pipeline.
            print(f"  ERROR processing '{filename}': {e}")

    print(f"[{class_label.upper()}] Done. {processed_count}/{len(wav_files)} files saved "
          f"to '{output_dir}'\n")

    return processed_count


def run_preprocessing_pipeline():
    """
    Top-level function that orchestrates the full preprocessing pipeline.

    Iterates over each class label (normal, abnormal), processes all .wav files
    in each class directory, and prints a summary of the results.

    This function takes no parameters — it reads from the module-level
    configuration constants defined at the top of this file.

    Returns
    -------
    None
        Results are saved to disk as .npy files. A summary is printed to stdout.
    """

    print("=" * 60)
    print("PREPROCESSING PIPELINE — Audio to Log-Mel-Spectrogram")
    print("=" * 60)

    # --- Print the active configuration so the user can verify settings ---
    # This is especially useful when running on Colab where you might override
    # the defaults. Printing config upfront makes debugging much easier.
    print(f"\n  Configuration:")
    print(f"    Sample Rate:     {SAMPLE_RATE} Hz")
    print(f"    Target Duration: {TARGET_DURATION_SEC} sec ({TARGET_LENGTH_SAMPLES} samples)")
    print(f"    N_FFT:           {N_FFT}")
    print(f"    Hop Length:      {HOP_LENGTH}")
    print(f"    N_Mels:          {N_MELS}")
    print(f"    Raw Data Dir:    {RAW_DATA_DIR}")
    print(f"    Output Dir:      {PROCESSED_DATA_DIR}")
    print()

    total_processed = 0  # Running total across all classes.

    # --- Process each class directory ---
    for label in CLASS_LABELS:
        count = process_class_directory(label)
        total_processed += count

    # --- Print final summary ---
    print("=" * 60)
    print(f"PIPELINE COMPLETE — {total_processed} total spectrograms saved.")
    print(f"Output shape per spectrogram: ({N_MELS}, ~{TARGET_LENGTH_SAMPLES // HOP_LENGTH + 1})")
    print("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================
# This block runs only when the script is executed directly (e.g., `python preprocess.py`).
# It does NOT run when the script is imported as a module (e.g., `from preprocess import ...`).
# This is standard Python practice for making scripts that are both runnable and importable.

if __name__ == "__main__":
    run_preprocessing_pipeline()
