"""
analyze_tuning.py — Render the HPO Summary Line Graph
============================================================
After tune.py finishes its grid search, docs/experiment_tracking.csv
contains one row per (lr, batch_size, base_filters) combination, with
the test-set accuracy / precision / recall / F1 / ROC-AUC for each. This
script reads that CSV with pandas and produces a single line graph
comparing F1-Score and ROC-AUC across runs, saved to
docs/visualizations/tuning_summary_graph.png.

WHY F1 AND ROC-AUC TOGETHER ON ONE GRAPH?
    Both metrics are imbalance-aware and address the "Accuracy Paradox"
    we discussed in evaluate.py: on MIMII, plain accuracy can look great
    while the model misses every real anomaly.

      - F1 (harmonic mean of precision + recall) summarizes performance
        at the FIXED 0.5 decision threshold. It tells us "given the
        default operating point, how good is this model?".
      - ROC-AUC summarizes performance across EVERY possible threshold.
        It tells us "how well does this model RANK abnormal vs. normal
        samples?" — independent of where we ultimately set the alarm.

    Plotting both lets us spot the case where one metric improves while
    the other degrades — usually a sign that a configuration is gaining
    on ranking quality but losing on calibration (or vice-versa). When
    both metrics rise together, the run is unambiguously better.

WHY A LINE PLOT (not a bar chart)?
    Lines emphasize TRENDS across the sweep. Reading left-to-right, you
    can see at a glance whether F1 climbs as base_filters grows, whether
    ROC-AUC peaks at a particular LR, etc. A bar chart would hide that
    trend information behind 8 separate bars per metric.

USAGE:
    Auto-invoked by tune.py at the end of a sweep. Can also be run
    manually any time docs/experiment_tracking.csv has new rows:
        python src/analyze_tuning.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os                          # Path joins + makedirs for the output dir
import sys                         # Clean exit code on missing CSV

import pandas as pd                # CSV → DataFrame
import matplotlib.pyplot as plt    # Figure container
import seaborn as sns              # Pretty defaults; lineplot for the curves


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_CSV    = os.path.join(PROJECT_ROOT, "docs", "experiment_tracking.csv")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "docs", "visualizations")
OUTPUT_PATH       = os.path.join(VISUALIZATION_DIR, "tuning_summary_graph.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("HPO ANALYSIS — Building tuning summary graph")
    print("=" * 60)

    # ---- 1. Read the experiment-tracking CSV ----
    # We fail loudly if the CSV is missing or empty: there's nothing to plot
    # and a confusing pandas EmptyDataError downstream would be worse than a
    # clear message here.
    if not os.path.isfile(EXPERIMENT_CSV):
        print(f"ERROR: {EXPERIMENT_CSV} not found. "
              f"Run tune.py first to populate it.")
        sys.exit(1)

    df = pd.read_csv(EXPERIMENT_CSV)
    if df.empty:
        print(f"ERROR: {EXPERIMENT_CSV} is empty — no runs to plot.")
        sys.exit(1)

    print(f"  Loaded {len(df)} run(s) from {EXPERIMENT_CSV}")

    # ---- 2. Establish a stable plotting order ----
    # We sort by F1 ascending so the graph reads naturally left-to-right
    # from worst to best. This makes the rightmost runs the headline winners
    # and turns the chart into a quick "which configs to keep?" reference.
    df_sorted = df.sort_values(by="f1", ascending=True).reset_index(drop=True)

    # ---- 3. Build the figure ----
    # Wide aspect ratio: with 8+ runs the run_name labels along the x-axis
    # need horizontal room or they'll collide.
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # X positions: integer indices 0..N-1. We'll relabel the ticks with
    # run_name strings below.
    x = range(len(df_sorted))

    # Plot F1 and ROC-AUC as two lines on a shared y-axis. Both metrics
    # live in [0, 1] so a shared axis is honest — no risk of misleading
    # dual-axis foreshortening.
    sns.lineplot(x=x, y=df_sorted["f1"], ax=ax, marker="o", linewidth=2,
                 label="F1-Score (threshold=0.5)")
    sns.lineplot(x=x, y=df_sorted["roc_auc"], ax=ax, marker="s", linewidth=2,
                 label="ROC-AUC (threshold-independent)")

    # ---- 4. Labels + ticks ----
    ax.set_xticks(list(x))
    # Rotate run_name labels — they're long ("run_lr1em3_bs32_bf16") and
    # would overlap horizontally otherwise.
    ax.set_xticklabels(df_sorted["run_name"], rotation=30, ha="right")
    ax.set_xlabel("Run name (sorted ascending by F1)")
    ax.set_ylabel("Score")
    ax.set_title("Hyperparameter Tuning Summary — F1 and ROC-AUC across runs")
    # Both metrics live in [0, 1]. Pinning the y-axis prevents matplotlib's
    # auto-scaling from making small differences look dramatic.
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")

    # ---- 5. Save ----
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also print the leaderboard so the terminal user sees the answer
    # without opening the PNG.
    print("\n  Leaderboard (sorted by F1, descending):")
    leaderboard = df.sort_values(by="f1", ascending=False)[
        ["run_name", "lr", "batch_size", "base_filters", "f1", "roc_auc"]
    ]
    print(leaderboard.to_string(index=False))
    print(f"\n  Summary graph saved: {OUTPUT_PATH}")
    print("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
