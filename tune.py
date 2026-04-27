"""
tune.py — HPO Master Orchestrator (Grid Search)
============================================================
Runs a full Cartesian-product grid search over learning rate, batch size,
and base filters, calling src/train.py → src/evaluate.py → src/explain.py
for every combination. Each run's checkpoint and visualizations are
namespaced by an auto-generated --run_name so nothing overwrites anything.
At the end, src/analyze_tuning.py is invoked to render a summary line graph
from the rows evaluate.py has been appending to docs/experiment_tracking.csv.

------------------------------------------------------------
WHY subprocess.run INSTEAD OF importing main() in a Python loop?
------------------------------------------------------------
This is the single most important design decision in the file. There are
three reasons we shell out to a fresh Python process for every experiment
rather than calling train.main() directly in a `for` loop:

  1. GUARANTEED GPU VRAM RELEASE.
     PyTorch tensors live on the GPU until the Python process that
     allocated them exits. Even del-ing models, calling
     torch.cuda.empty_cache(), and running gc.collect() leaves residual
     allocations behind — caching allocator fragments, autograd state,
     pinned host buffers, and CUDA contexts are NOT fully reclaimed
     within a single process. After 3-4 large runs in the same process,
     a Colab T4 (16 GB) reliably OOMs partway through training even
     though each individual run fits comfortably.

     subprocess.run spawns a brand-new Python interpreter. When that
     child process exits, the OS reclaims its entire memory footprint
     — VRAM and host RAM — atomically and completely. Run N+1 starts
     from a clean slate, every time. This is the ONLY reliable way to
     do back-to-back PyTorch experiments in one session without manual
     intervention.

  2. CRASH ISOLATION.
     If one combination diverges, hits an unrelated CUDA error, or the
     kernel is killed by Colab for transient reasons, only that single
     subprocess dies. The orchestrator catches the non-zero exit code
     and decides what to do; the other 11 experiments are unaffected.
     A bare-loop import-and-call would propagate the exception and kill
     the entire grid search.

  3. EXACT REPRODUCIBILITY.
     Each subprocess receives identical CLI args every time. Module-level
     side effects (CUDA initialization, RNG seeding, lazy imports) all
     happen in a fresh interpreter. No "the second run behaves differently
     because some global state stuck around" — a well-known headache when
     tuning in notebooks.

The cost is small: ~1-2 seconds of Python startup per subprocess, a
negligible fraction of any real training run.

------------------------------------------------------------
USAGE
------------------------------------------------------------
From the project root:
    python tune.py

The grid below produces 2 × 2 × 2 = 8 experiments by default. Edit the
HYPERPARAMETER_GRID dict to expand or shrink the search.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os                # Path joins for the cross-platform script paths
import sys               # sys.executable → use the SAME Python interpreter
                         # that launched tune.py for every subprocess
import subprocess        # The whole point of this script — see file header
import itertools         # itertools.product gives us the Cartesian grid
import time              # Per-run timing so we can budget Colab credits


# ============================================================================
# HYPERPARAMETER GRID — Edit this dict to change the search space.
# ============================================================================
# itertools.product over the dict's value lists produces every combination.
# 2 × 2 × 2 = 8 runs is a reasonable starter sweep that finishes in one
# Colab session. To expand, just add more values to the lists — e.g.,
# adding a fourth LR turns this into 4 × 2 × 2 = 16 runs.
#
# WHY THESE RANGES?
#   - lr ∈ {1e-3, 5e-4}: Adam's canonical default vs. a 2× smaller step.
#     Lower learning rates train more slowly but converge to a tighter
#     minimum; this range probes whether the default is already optimal.
#   - batch_size ∈ {16, 32}: 16 = noisier gradients, often better
#     generalization on small datasets; 32 = the architecture's
#     comfort zone. Both fit on a Colab T4.
#   - base_filters ∈ {16, 32}: small vs. medium model capacity. We avoid
#     64 in the initial sweep because it doubles params and roughly
#     doubles per-epoch training time.
HYPERPARAMETER_GRID = {
    "lr":           [1e-3, 5e-4],
    "batch_size":   [16, 32],
    "base_filters": [16, 32],
}


# ============================================================================
# PATHS — Resolved relative to this file so tune.py works from any CWD.
# ============================================================================
PROJECT_ROOT     = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT     = os.path.join(PROJECT_ROOT, "src", "train.py")
EVALUATE_SCRIPT  = os.path.join(PROJECT_ROOT, "src", "evaluate.py")
EXPLAIN_SCRIPT   = os.path.join(PROJECT_ROOT, "src", "explain.py")
ANALYZE_SCRIPT   = os.path.join(PROJECT_ROOT, "src", "analyze_tuning.py")


# ============================================================================
# HELPERS
# ============================================================================

def make_run_name(lr, batch_size, base_filters):
    """
    Build a deterministic, filesystem-safe identifier for one HPO run.

    The pattern is:
        run_lr<LR>_bs<BATCH>_bf<FILTERS>

    Where the LR is rendered with no decimal point or scientific notation
    that could confuse a filesystem (we replace '.' and '-' with 'p' and
    'm'). Example: lr=5e-4, batch_size=16, base_filters=32
        → "run_lr5em4_bs16_bf32"

    DETERMINISTIC: same args always produce the same name, so re-running
    tune.py is idempotent — it will overwrite the old run's outputs in
    place rather than producing parallel "_v2" copies.
    """
    # Format LR with up to four significant digits in scientific notation
    # ("5.0e-04"), then strip filesystem-hostile chars.
    lr_tag = f"{lr:.0e}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"run_lr{lr_tag}_bs{batch_size}_bf{base_filters}"


def run_step(label, cmd):
    """
    Run one subprocess step (train / evaluate / explain) for a single HPO run.

    Returns True on success (exit code 0), False on any non-zero exit. We
    do NOT use check=True here because we want CONTROLLED error handling:
    the caller decides whether a failure halts the entire grid or just
    skips ahead to the next combination.

    capture_output=False means stdout/stderr stream directly to the parent
    terminal in real time — essential for watching a 30-minute training run
    without blind-flying. (capture_output=True would buffer everything in
    memory until the child exits, which is both wasteful and misleading.)
    """
    print(f"\n  ── [{label}] ", end="")
    print(" ".join(map(str, cmd)))
    print("  " + "─" * 56)

    # We deliberately do NOT pass check=True. A non-zero exit becomes a
    # False return value, letting the orchestrator log+halt cleanly.
    result = subprocess.run(cmd)
    success = (result.returncode == 0)

    if not success:
        print(f"  ── [{label}] FAILED with exit code {result.returncode}")
    return success


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    # itertools.product expands the dict-of-lists into a list of tuples
    # — one tuple per Cartesian combination. We pull the keys once so we
    # know the tuple ordering matches.
    keys = list(HYPERPARAMETER_GRID.keys())
    combinations = list(itertools.product(*HYPERPARAMETER_GRID.values()))

    total = len(combinations)
    print("=" * 64)
    print(f"HPO GRID SEARCH — {total} combinations")
    for k, vals in HYPERPARAMETER_GRID.items():
        print(f"  {k:>14}: {vals}")
    print("=" * 64)

    sweep_start = time.time()
    completed = 0
    failed_runs = []

    for idx, values in enumerate(combinations, start=1):
        # Build a {param: value} dict for this specific combination.
        params = dict(zip(keys, values))
        run_name = make_run_name(**params)

        print("\n" + "=" * 64)
        print(f"[{idx}/{total}] {run_name}")
        for k, v in params.items():
            print(f"    {k} = {v}")
        print("=" * 64)

        # Common CLI tail used by all three sub-scripts. sys.executable is
        # the SAME interpreter that launched tune.py — using it (rather
        # than a bare "python") guarantees the subprocess sees the same
        # virtualenv / Conda env / Colab kernel as the parent.
        common_args = [
            "--lr",           str(params["lr"]),
            "--batch_size",   str(params["batch_size"]),
            "--base_filters", str(params["base_filters"]),
            "--run_name",     run_name,
        ]

        run_start = time.time()

        # ---- Step 1: TRAIN ----
        # Produces models/best_model_{run_name}.pth.
        if not run_step("TRAIN", [sys.executable, TRAIN_SCRIPT, *common_args]):
            failed_runs.append((run_name, "train"))
            print(f"\n  Halting grid search at run {idx}/{total} due to "
                  f"train failure.")
            break

        # ---- Step 2: EVALUATE ----
        # Loads that checkpoint, writes confusion_matrix_{run_name}.png,
        # roc_curve_{run_name}.png, and APPENDS a row to
        # docs/experiment_tracking.csv (Task 2).
        if not run_step("EVALUATE", [sys.executable, EVALUATE_SCRIPT, *common_args]):
            failed_runs.append((run_name, "evaluate"))
            print(f"\n  Halting grid search at run {idx}/{total} due to "
                  f"evaluate failure.")
            break

        # ---- Step 3: EXPLAIN ----
        # Writes gradcam_normal_{run_name}.png, gradcam_abnormal_{run_name}.png.
        if not run_step("EXPLAIN", [sys.executable, EXPLAIN_SCRIPT, *common_args]):
            failed_runs.append((run_name, "explain"))
            print(f"\n  Halting grid search at run {idx}/{total} due to "
                  f"explain failure.")
            break

        run_secs = time.time() - run_start
        completed += 1
        print(f"\n  ✓ {run_name} finished in {run_secs/60:.1f} min "
              f"({completed}/{total} done)")

    # ---- Summary ----
    sweep_secs = time.time() - sweep_start
    print("\n" + "=" * 64)
    print(f"GRID SEARCH COMPLETE — {completed}/{total} runs succeeded "
          f"in {sweep_secs/60:.1f} min")
    if failed_runs:
        print(f"  Failed: {failed_runs}")
    print("=" * 64)

    # ---- Final step: render the summary graph ----
    # Per the spec, we automatically invoke analyze_tuning.py here so the
    # user gets the graph for free without remembering a second command.
    # We only run it if at least one experiment produced a CSV row;
    # otherwise the graph would be empty and the script would crash on an
    # empty DataFrame.
    if completed > 0:
        print("\nGenerating tuning summary graph...")
        analyze_ok = run_step(
            "ANALYZE",
            [sys.executable, ANALYZE_SCRIPT],
        )
        if not analyze_ok:
            print("  Summary graph generation FAILED — see error above.")
    else:
        print("\nSkipping summary graph: no successful runs to plot.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
