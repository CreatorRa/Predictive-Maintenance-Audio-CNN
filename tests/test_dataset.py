"""
test_dataset.py — Unit tests for src/dataset.py
============================================================
Two contracts in dataset.py are easy to break in a refactor and
catastrophic to break in production:

    1. compute_class_weights MUST heavily up-weight the minority class.
       This is the single line of code that prevents the model from
       collapsing to "always predict normal" — a 90%+-accurate model
       that catches zero faults. If the math drifts, the model silently
       degrades to useless without throwing any error.

    2. stratified_split MUST preserve class ratios in every subset.
       Without stratification, a random split could put zero abnormal
       samples in the test set, which makes F1 / Recall / ROC-AUC
       undefined or trivially zero — invalidating the entire final
       evaluation.

WHAT FAILURES THESE TESTS PROTECT AGAINST:
    - A "small math cleanup" that swaps the inverse-frequency formula
      from N_total / (num_classes * N_c) to something like 1 / N_c
      → the relative weight ratio between classes changes, the model
      retrains, and we don't notice the regression until production.
    - A refactor that switches stratified_split to use train_test_split
      WITHOUT the stratify= argument → splits look fine on a balanced
      toy dataset but degenerate on the real (heavily imbalanced) data.
    - Off-by-one in the two-stage split arithmetic that drops or
      duplicates samples.

DESIGN PRINCIPLE — NO DISK I/O, NO REAL .npy FILES:
    We test the math/logic of compute_class_weights and stratified_split
    by passing them in-memory Python lists of fake labels and string
    paths. Neither function actually opens a file — they just shuffle
    metadata around — so we get full coverage in microseconds.
"""

import numpy as np

# Imports rely on tests/conftest.py inserting <repo>/src into sys.path.
from dataset import (
    compute_class_weights,
    stratified_split,
)


# ============================================================================
# SECTION 1 — CLASS WEIGHT MATH
# ============================================================================
# We feed compute_class_weights a hand-crafted label list with a known
# imbalance and assert the returned weights match the inverse-frequency
# formula exactly.

def test_class_weights_heavily_penalize_minority_class():
    """
    PROTECTS AGAINST:
        A regression that silently flattens the weights toward 1.0,
        which would let the model coast to high accuracy by predicting
        "normal" every time — exactly the failure mode class weighting
        is supposed to prevent.

    SETUP:
        labels = [0, 0, 0, 1]   →   3 normal, 1 abnormal (75/25 split)

    EXPECTED MATH (inverse-frequency, num_classes=2, N_total=4):
        weight[0] = 4 / (2 * 3) = 0.6667   (normal — common, low weight)
        weight[1] = 4 / (2 * 1) = 2.0      (abnormal — rare, high weight)

    INVARIANTS WE ASSERT:
        - weight[1] > weight[0]               (minority gets higher penalty)
        - weight[1] / weight[0] == 3.0        (ratio matches class imbalance)
        - Numeric values match the closed-form formula exactly.
    """
    # Dummy labels chosen to make the math simple and the imbalance visible.
    # 3:1 ratio is realistic for MIMII (heavily skewed toward normal).
    labels = [0, 0, 0, 1]

    weights = compute_class_weights(labels, num_classes=2)

    # --- Closed-form expected values ---
    # We compute these from first principles instead of hard-coding the
    # numbers, so a future change to the formula in dataset.py would
    # have to ALSO update this assertion's logic — preventing silent
    # drift between code and test.
    n_total = len(labels)
    expected_w0 = n_total / (2 * labels.count(0))   # 4 / (2*3) = 0.6667
    expected_w1 = n_total / (2 * labels.count(1))   # 4 / (2*1) = 2.0

    # numpy's allclose handles float rounding (the dataset code casts
    # to float32, so exact equality is unsafe across architectures).
    np.testing.assert_allclose(weights[0], expected_w0, rtol=1e-5)
    np.testing.assert_allclose(weights[1], expected_w1, rtol=1e-5)

    # --- The behavioral assertion the user actually cares about ---
    # The minority class MUST receive a strictly larger weight than the
    # majority class. If this assertion ever fails, class weighting is
    # broken and the model is at risk of collapsing to "always normal."
    assert weights[1] > weights[0], (
        f"Minority class (label 1) should have a larger weight than the "
        f"majority class (label 0). Got w0={weights[0]}, w1={weights[1]}."
    )

    # The exact ratio of weights should equal the inverse ratio of class
    # frequencies — that's the whole point of inverse-frequency weighting.
    # 3 normal : 1 abnormal → w1/w0 should be 3.0 exactly.
    np.testing.assert_allclose(weights[1] / weights[0], 3.0, rtol=1e-5)


def test_class_weights_balanced_input_produces_unit_weights():
    """
    PROTECTS AGAINST:
        A regression where compute_class_weights inserts a spurious
        scale factor (e.g. multiplying by num_classes again). On the
        production imbalanced dataset such a bug would still produce
        DIFFERENT weights for each class, masking the issue. Only a
        perfectly-balanced input reveals "should be 1.0 but isn't."
    """
    # 50/50 split — every class is exactly N_total/num_classes samples,
    # so every weight should equal 1.0 exactly.
    labels = [0, 0, 1, 1]

    weights = compute_class_weights(labels, num_classes=2)

    np.testing.assert_allclose(weights, [1.0, 1.0], rtol=1e-5)


def test_class_weights_dtype_is_float32():
    """
    PROTECTS AGAINST:
        A change that returns float64 weights. CrossEntropyLoss accepts
        either, but mixing float32 inputs with a float64 weight tensor
        triggers an implicit upcast that wastes GPU memory and (more
        importantly) HIDES dtype bugs elsewhere in the pipeline by
        silently fixing them.
    """
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2)
    assert weights.dtype == np.float32, (
        f"Expected float32 weights, got {weights.dtype}"
    )


# ============================================================================
# SECTION 2 — STRATIFIED TRAIN / VAL / TEST SPLIT
# ============================================================================
# We build a fake dataset of dummy "paths" (just unique strings — they
# never get opened) with a known class imbalance, run stratified_split,
# and verify the output ratios.

def _build_fake_dataset(n_normal, n_abnormal):
    """
    Generate parallel lists of fake paths and labels with a chosen imbalance.

    Why fake string paths instead of real files?
        stratified_split treats `file_paths` as opaque tokens — it never
        reads them. Using strings keeps the test entirely in memory and
        means we can scale to thousands of samples without slowing down.
    """
    paths = (
        [f"normal_{i}.npy"   for i in range(n_normal)] +
        [f"abnormal_{i}.npy" for i in range(n_abnormal)]
    )
    labels = [0] * n_normal + [1] * n_abnormal
    return paths, labels


def test_stratified_split_preserves_class_ratio_in_every_subset():
    """
    PROTECTS AGAINST:
        - A refactor that drops the stratify= argument from sklearn's
          train_test_split — splits would still LOOK reasonable on a
          balanced toy set but break on real imbalanced data.
        - Off-by-one in the relative_test_size calculation that biases
          val and test toward different class distributions.

    SETUP:
        1000 total samples, 80% normal / 20% abnormal — realistic MIMII
        imbalance with enough samples that the 70/15/15 split produces
        non-trivial subset sizes.

    INVARIANT:
        Within ~3 percentage points, every subset's abnormal fraction
        should match the global 20%. The tolerance accounts for the
        fact that integer counts can't divide evenly into exact
        percentages.
    """
    n_normal, n_abnormal = 800, 200
    paths, labels = _build_fake_dataset(n_normal, n_abnormal)
    expected_abnormal_fraction = n_abnormal / (n_normal + n_abnormal)  # 0.20

    splits = stratified_split(paths, labels, random_seed=42)

    # Tolerance: with 150 val samples and 30 expected abnormals, ±1 sample
    # shifts the fraction by ~0.7 pp. ±3 pp is generous but tight enough
    # that an unstratified random split (which can deviate 5–10 pp) fails
    # this test reliably.
    tol = 0.03

    for subset_name in ("train_labels", "val_labels", "test_labels"):
        subset = splits[subset_name]
        n_total = len(subset)
        # Defensive: make sure no subset is empty. Some refactors of the
        # split arithmetic accidentally produce a 0-size test set.
        assert n_total > 0, f"Subset {subset_name} is empty after split."

        observed_fraction = subset.count(1) / n_total

        assert abs(observed_fraction - expected_abnormal_fraction) < tol, (
            f"Subset '{subset_name}' has abnormal fraction "
            f"{observed_fraction:.3f}, expected ~{expected_abnormal_fraction:.3f} "
            f"(±{tol:.2f}). Stratification is broken."
        )


def test_stratified_split_ratios_are_70_15_15():
    """
    PROTECTS AGAINST:
        A regression in the two-stage split arithmetic — particularly
        the formula `relative_test_size = test_ratio / (val_ratio + test_ratio)`
        which is easy to mess up. If someone "simplifies" it to just
        test_ratio, val and test sizes drift dramatically.

    INVARIANT:
        Subset sizes match the requested 70/15/15 ratios within ±1
        sample (sklearn's split rounds to integer counts, so ±1 is
        unavoidable but anything larger indicates a logic bug).
    """
    paths, labels = _build_fake_dataset(800, 200)
    n_total = len(paths)

    splits = stratified_split(paths, labels, random_seed=42)

    # ±1 tolerance accounts for sklearn's integer rounding behavior.
    assert abs(len(splits["train_paths"]) - int(n_total * 0.70)) <= 1
    assert abs(len(splits["val_paths"])   - int(n_total * 0.15)) <= 1
    assert abs(len(splits["test_paths"])  - int(n_total * 0.15)) <= 1

    # Every sample must end up in EXACTLY ONE subset — no duplication,
    # no losses. This catches a whole class of bugs where slicing
    # accidentally drops or repeats elements.
    total_split = (
        len(splits["train_paths"])
        + len(splits["val_paths"])
        + len(splits["test_paths"])
    )
    assert total_split == n_total, (
        f"Splits do not cover all samples: got {total_split}, expected {n_total}"
    )


def test_stratified_split_has_no_overlap_between_subsets():
    """
    PROTECTS AGAINST:
        A refactor that accidentally feeds the FULL list (instead of the
        leftover `temp_paths`) into the second train_test_split call.
        That bug would silently leak training samples into the test set,
        producing an inflated test-set F1 that the team would happily
        report. This test is the only line of defense against that
        specific failure mode.
    """
    paths, labels = _build_fake_dataset(800, 200)
    splits = stratified_split(paths, labels, random_seed=42)

    train_set = set(splits["train_paths"])
    val_set   = set(splits["val_paths"])
    test_set  = set(splits["test_paths"])

    # Pairwise intersections must all be empty.
    assert train_set.isdisjoint(val_set),  "train and val subsets overlap!"
    assert train_set.isdisjoint(test_set), "train and test subsets overlap — DATA LEAK!"
    assert val_set.isdisjoint(test_set),   "val and test subsets overlap!"
