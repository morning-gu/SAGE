"""get_label_token_ids three-branch behavior (sage_check/constants.py).

Uses a stub tokenizer so no model download is required:
  branch 1: code encodes to a single token directly
  branch 2: code needs the prefix-space form (" nr") to be single-token
  branch 3: neither form is single-token -> code is skipped (BCE excludes it)
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
from sage_check.constants import BEHAVIOR_CODES, get_label_token_ids  # noqa: E402


class StubTokenizer:
    """encode('x') -> [hash(x)]; encode(' x') -> [hash(x) + 10000];
    codes longer than 2 chars encode to two tokens (forced multi-token)."""

    def encode(self, text: str, add_special_tokens=False):
        base = abs(hash(text)) % 50000
        if text.startswith(" ") and len(text.strip()) <= 2:
            return [base]
        if len(text) <= 2:
            return [base]
        return [base, base + 1]


def test_all_branches():
    tok = StubTokenizer()
    ids = get_label_token_ids(tok)
    assert set(ids) == set(BEHAVIOR_CODES)  # all 2-letter codes resolve
    assert len(ids) == len(BEHAVIOR_CODES)
    assert all(isinstance(v, int) for v in ids.values())


class DirectOnlyTokenizer(StubTokenizer):
    """Only the bare form is single-token; ' x' becomes two tokens."""

    def encode(self, text: str, add_special_tokens=False):
        if text.startswith(" "):
            return [abs(hash(text)) % 50000, 99999]
        if len(text) <= 2:
            return [abs(hash(text)) % 50000]
        return [1, 2]


def test_direct_branch():
    tok = DirectOnlyTokenizer()
    ids = get_label_token_ids(tok)
    assert set(ids) == set(BEHAVIOR_CODES)


class SpaceFallbackTokenizer(StubTokenizer):
    """Bare form is always multi-token; only the space-prefixed form works."""

    def encode(self, text: str, add_special_tokens=False):
        if text.startswith(" ") and len(text.strip()) <= 2:
            return [abs(hash(text.strip())) % 50000]
        if len(text) <= 2:
            return [abs(hash(text)) % 50000, 88888]
        return [1, 2]


def test_space_fallback_branch():
    tok = SpaceFallbackTokenizer()
    ids = get_label_token_ids(tok)
    assert set(ids) == set(BEHAVIOR_CODES)


class BrokenTokenizer:
    """Nothing ever encodes to a single token."""

    def encode(self, text: str, add_special_tokens=False):
        return [1, 2]


def test_skip_branch_reports_empty():
    ids = get_label_token_ids(BrokenTokenizer())
    assert ids == {}
