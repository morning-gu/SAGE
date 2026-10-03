"""Taxonomy single-source-of-truth checks.

After the 2026-09 refactor, BEHAVIOR_NAMES / BEHAVIOR_CODES / CODE_TO_NAME
must exist only in sage/taxonomy.py (the domain core); every other module
imports from there.  sage_check/constants.py re-exports it for backward
compatibility plus the VLM prompt/parsing concerns.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
from sage_check.constants import (  # noqa: E402
    BEHAVIOR_CODES,
    BEHAVIOR_NAMES,
    CODE_TO_NAME,
)

NUM_CLASSES = len(BEHAVIOR_NAMES)


def test_fifteen_classes():
    assert NUM_CLASSES == 15
    assert len(BEHAVIOR_CODES) == 15
    assert len(CODE_TO_NAME) == 15
    assert set(CODE_TO_NAME.values()) == set(BEHAVIOR_NAMES)


def test_metrics_and_data_share_taxonomy():
    from sage.data import BEHAVIOR_NAMES as DATA_NAMES, NUM_CLASSES as DATA_N
    from sage.metrics import BEHAVIOR_NAMES as METRIC_NAMES, NUM_CLASSES as METRIC_N
    assert DATA_NAMES == BEHAVIOR_NAMES and DATA_N == NUM_CLASSES
    assert METRIC_NAMES is BEHAVIOR_NAMES  # re-export, same object
    assert METRIC_N == NUM_CLASSES


def test_train_cnn_mlb_uses_shared_taxonomy():
    import scripts.train_cnn_mlb as m
    assert m.BEHAVIOR_NAMES is BEHAVIOR_NAMES
    assert m.NUM_CLASSES == NUM_CLASSES


def test_taxonomy_lives_in_sage_package():
    """The taxonomy source of truth must be sage/taxonomy.py."""
    src = (REPO / "sage/taxonomy.py").read_text(encoding="utf-8")
    for pattern in ("BEHAVIOR_CODES = [", "CODE_TO_NAME = {", "NAME_TO_CODE = "):
        assert pattern in src, f"taxonomy definition missing: {pattern}"
    # No other package module may re-define the taxonomy lists.
    for mod in ("sage_check/constants.py", "sage/metrics.py", "sage/data.py"):
        other = (REPO / mod).read_text(encoding="utf-8")
        for pattern in ("BEHAVIOR_CODES = [", "CODE_TO_NAME = {"):
            assert pattern not in other, f"{mod} re-defines {pattern}"


def test_server_has_no_local_taxonomy_copy():
    """servers/vlm_detect_server.py must import, not re-declare, the taxonomy."""
    src = (REPO / "servers/vlm_detect_server.py").read_text(encoding="utf-8")
    for pattern in ("BEHAVIOR_CODES = [", "BEHAVIOR_NAMES = [", "CODE_TO_NAME = {"):
        assert pattern not in src, f"local duplicate found: {pattern}"
    assert "from sage_check.constants import" in src


def test_no_sys_path_hacks():
    """Packages are installed (pip install -e .); runtime sys.path.insert is banned.

    Scripts importing *other scripts* under scripts/ is also banned — shared
    code belongs in the sage package (see sage/eval_harness.py).
    """
    roots = ("sage", "sage_check", "sage_recheck", "scripts",
             "servers", "tests", "tools")
    offenders = []
    for root in roots:
        for p in (REPO / root).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if p.name == "test_constants_sync.py":
                continue  # this tripwire contains the literal it searches for
            if "sys.path.insert" in p.read_text(encoding="utf-8"):
                offenders.append(str(p.relative_to(REPO)))
    assert offenders == [], f"sys.path.insert crept back in: {offenders}"


def test_eval_scripts_import_shared_metrics():
    """eval_* scripts must use the shared harness, not inline metric code."""
    for rel in ("scripts/eval_veto.py", "scripts/eval_vlm_check.py",
                "scripts/eval_yolo_cls.py", "scripts/eval_yolo_bce.py",
                "scripts/eval_cnn_mlb.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "from sage.eval_harness import" in src, f"{rel} not refactored"
        # no local def of the shared functions
        for fn in ("def _prf(", "def average_precision(", "def compute_ece(",
                   "def latency_stats("):
            assert fn not in src, f"{rel} still defines {fn}"
