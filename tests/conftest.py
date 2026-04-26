"""
conftest.py — pytest shared fixtures and import-path setup.
============================================================
This file is auto-loaded by pytest before any test in tests/ runs. We use it
for ONE job: make `import preprocess`, `import dataset`, `import model`
work without requiring an editable install or PYTHONPATH gymnastics.

WHY THIS IS NECESSARY:
    Our source code lives in src/ (no __init__.py — it's a flat module
    folder, not a package). Pytest discovers tests in tests/, but its
    default sys.path doesn't include src/, so a bare `import preprocess`
    inside a test would fail with ModuleNotFoundError.

    Inserting src/ at the front of sys.path here means every test file
    can use the same `from preprocess import ...` style that the real
    pipeline modules use (e.g. dataset.py imports model.py the same way).
    Keeping the import style identical between production code and tests
    avoids subtle bugs where a test passes against a slightly-different
    import path than what runs in production.
"""

import os
import sys

# Compute the absolute path to <repo_root>/src/ regardless of where pytest
# was invoked from. __file__ is this conftest.py; its parent is tests/;
# its grandparent is the repo root; src/ sits beside tests/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "src"))

# Insert at index 0 so our local modules win over any same-named packages
# that may exist on the system path (defensive — prevents a stray
# `pip install model` package from shadowing src/model.py).
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
