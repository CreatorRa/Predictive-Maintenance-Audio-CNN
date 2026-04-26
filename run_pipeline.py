"""
run_pipeline.py — Master Orchestrator for the Predictive Maintenance Pipeline
============================================================================
This script is the SINGLE COMMAND that runs the entire project end-to-end:

    pytest  →  preprocess  →  dataset smoke test  →  model smoke test  →
    train   →  evaluate    →  explain (Grad-CAM)

It exists so a reviewer (or our future selves) can reproduce every artifact
in the report — processed spectrograms, trained checkpoint, confusion
matrix, ROC curve, Grad-CAM heatmaps — by typing one command:

    python run_pipeline.py

WHY subprocess.run INSTEAD OF DIRECT IMPORTS?
    The intuitive alternative is to import each script's main() function
    and call them sequentially in this same Python process. We deliberately
    do NOT do that, because each stage is HEAVY and stateful:

      1. MEMORY HYGIENE BETWEEN STAGES.
         preprocess.py loads librosa + audio buffers. dataset.py builds
         large tensor caches. train.py allocates the model on the GPU,
         optimizer state, gradient buffers, and a backward graph that
         can hold hundreds of MB of activations. Once a stage finishes,
         we want ALL of that memory returned to the OS, not lingering
         in our process while the next stage runs.

         When we call subprocess.run, the child Python interpreter exits
         when the stage finishes — Python's reference counts get torn
         down, the CUDA context is destroyed, and every byte is reclaimed.
         A long import-and-call chain in one process, by contrast, leaves
         partially-freed PyTorch caches and lingering tensor references
         that can OOM the next stage on a small GPU (e.g. Colab T4).

      2. ISOLATED FAILURE MODES.
         If train.py crashes with a CUDA OOM, we don't want the crash to
         take down the orchestrator's own state. A subprocess crash is
         localized — its non-zero exit code propagates back to us as a
         clean signal we can react to without our own process being in
         an undefined state.

      3. RESET GLOBAL STATE EVERY STAGE.
         PyTorch caches device kernels. Matplotlib accumulates open
         figures. NumPy's RNG state changes silently. Restarting the
         interpreter for each stage gives every stage a guaranteed
         pristine starting environment, which removes a whole class of
         "works alone but not in sequence" bugs.

      4. EXACT PARITY WITH MANUAL RUNS.
         The commands this script issues are the same commands a developer
         would type by hand: `pytest`, `python src/preprocess.py`, etc.
         Whatever works here works one-by-one in the terminal too —
         no hidden orchestrator-only behavior to debug later.

ERROR HANDLING:
    If any stage exits with a non-zero return code, we print a loud,
    bordered error block naming the failing stage and HALT IMMEDIATELY.
    Subsequent stages are skipped because they would only fail in
    confusing ways (e.g. evaluate.py needs the checkpoint that train.py
    didn't produce). Halting fast surfaces the real problem clearly.

USAGE:
    From the project root:
        python run_pipeline.py

"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys             # sys.executable for the current Python interpreter,
                       # sys.exit for halting with a meaningful return code.
import time            # Wall-clock timing for per-stage progress reporting.
import subprocess      # Process spawning — see "WHY subprocess.run" above.


# ============================================================================
# PIPELINE DEFINITION
# ============================================================================
# Each stage is a (label, command) tuple. The label is what we print in the
# banner; the command is the actual argv list passed to subprocess.run.
#
# Why a list of argv tokens (not a single shell string)?
#   - shell=False is the secure, cross-platform default. It bypasses the
#     OS shell entirely, so spaces in paths and quoting differences between
#     Windows cmd / bash / PowerShell are non-issues.
#   - sys.executable expands to the EXACT Python interpreter currently
#     running this orchestrator. Critical when the project lives inside a
#     virtualenv (.venv) — bare `python` might resolve to a different
#     interpreter on PATH and miss our installed packages.
#
# Why pytest is FIRST:
#   The whole point of the test suite is to catch bugs CHEAPLY before we
#   spend 30+ minutes preprocessing audio and training a model. If the
#   tests fail, we learn within seconds and fix the bug; we never burn
#   GPU time on broken code.

PIPELINE_STAGES = [
    # --- Stage 0: Fast safety net. Runs in milliseconds. ---
    ("Test Suite (pytest)",            [sys.executable, "-m", "pytest", "tests/", "-v"]),

    # --- Stage 1: Audio → spectrograms. Heavy disk I/O, librosa-bound. ---
    ("Preprocessing (audio → .npy)",   [sys.executable, "src/preprocess.py"]),

    # --- Stage 2: Verify the DataLoader sees the new spectrograms. ---
    # dataset.py's __main__ block does a single-batch smoke fetch.
    ("Dataset Smoke Test",             [sys.executable, "src/dataset.py"]),

    # --- Stage 3: Verify the model wires up. Cheap forward pass on dummy data. ---
    ("Model Smoke Test",               [sys.executable, "src/model.py"]),

    # --- Stage 4: The expensive one. Trains the CNN and writes the checkpoint. ---
    ("Training",                       [sys.executable, "src/train.py"]),

    # --- Stage 5: Final test-set metrics + confusion matrix + ROC curve PNGs. ---
    ("Evaluation (test metrics)",      [sys.executable, "src/evaluate.py"]),

    # --- Stage 6: Grad-CAM heatmaps for the report. ---
    ("Explainability (Grad-CAM)",      [sys.executable, "src/explain.py"]),
]


# ============================================================================
# RUNNER
# ============================================================================

def run_stage(label, command):
    """
    Execute one pipeline stage as a subprocess and return its exit code.

    Parameters
    ----------
    label : str
        Human-readable name shown in the banner (e.g., "Training").
    command : list of str
        The argv list to pass to subprocess.run. First element is the
        executable; subsequent elements are command-line arguments.

    Returns
    -------
    returncode : int
        The child process's exit code. 0 means success; anything else
        is treated as failure by the orchestrator.
    elapsed : float
        Wall-clock seconds the stage took. Useful for the per-stage
        timing summary at the end.
    """

    # Loud banner so the stage boundary is unmissable in a long log.
    print("\n" + "=" * 70)
    print(f"  STAGE: {label}")
    print(f"  CMD:   {' '.join(command)}")
    print("=" * 70 + "\n")

    start = time.perf_counter()

    # subprocess.run with check=False means we receive the exit code as
    # a normal return value rather than as a CalledProcessError exception.
    # We deliberately want manual control over the failure path (custom
    # error block + halt) instead of letting an exception bubble up.
    #
    # We do NOT capture stdout/stderr — we let the child stream directly
    # to our terminal. This gives the user real-time progress (Tqdm bars,
    # per-epoch logs from train.py) instead of a wall of text dumped at
    # the end. The trade-off is that we can't post-process the output
    # here, which we don't need to do anyway.
    completed = subprocess.run(command, check=False)

    elapsed = time.perf_counter() - start
    return completed.returncode, elapsed


def report_failure(label, returncode):
    """
    Print a loud, bordered error block when a stage fails. Used right
    before sys.exit so the failure is impossible to miss in a long log.

    Why so visually heavy?
        The output of train.py alone can be hundreds of lines. A subtle
        "stage failed" line gets lost. A bordered, all-caps block stays
        visible even when scrolling through epoch-by-epoch metrics.
    """
    border = "!" * 70
    print("\n" + border)
    print(border)
    print(f"!!  PIPELINE HALTED — STAGE FAILED: {label}")
    print(f"!!  Exit code: {returncode}")
    print(f"!!  Subsequent stages were skipped because they depend on this one.")
    print(border)
    print(border + "\n")


def main():
    """
    Walk the PIPELINE_STAGES list in order, halting on the first non-zero
    exit code. Prints a per-stage timing summary at the end on success.
    """

    print("=" * 70)
    print("  PREDICTIVE MAINTENANCE AUDIO CNN — FULL PIPELINE")
    print(f"  Stages to run: {len(PIPELINE_STAGES)}")
    print("=" * 70)

    # Track timings so we can print a summary table at the end. Helpful
    # for spotting unexpectedly slow stages over time.
    timings = []

    overall_start = time.perf_counter()

    for label, command in PIPELINE_STAGES:
        returncode, elapsed = run_stage(label, command)
        timings.append((label, elapsed, returncode))

        if returncode != 0:
            # Halt immediately. We do NOT continue to the next stage —
            # downstream stages depend on this one's outputs and would
            # only fail in more confusing ways.
            report_failure(label, returncode)
            # sys.exit propagates the child's non-zero code so a caller
            # (CI, a shell `&&` chain, etc.) can react correctly.
            sys.exit(returncode)

    overall_elapsed = time.perf_counter() - overall_start

    # --- Success summary ---
    # We only reach this block if EVERY stage returned 0.
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — every stage succeeded")
    print("=" * 70)
    print(f"\n  Total elapsed: {overall_elapsed:6.1f} s\n")
    print(f"  {'Stage':<35} {'Time (s)':>10}")
    print(f"  {'-' * 35} {'-' * 10}")
    for label, elapsed, _ in timings:
        print(f"  {label:<35} {elapsed:>10.1f}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
