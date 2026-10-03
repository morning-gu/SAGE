"""Loss-mechanism tests: MP correction identity and BCE label-position
off-by-one semantics (scripts/train_sage_vlm.py).

torch is imported lazily inside the training module, so these tests run on
CPU without ms-swift installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
from scripts.train_sage_vlm import (  # noqa: E402
    _label_position,
    _label_position_bce,
    _label_position_mp_correction,
)

# 15 label token ids (synthetic vocab: codes map to ids 100..114)
LABEL_IDS = {c: 100 + i for i, c in enumerate(
    ["nr", "aw", "bl", "ty", "ph", "sn", "ec", "pr", "bw", "cr",
     "tl", "tn", "sl", "rc", "lu"])}
ID_TENSOR = torch.tensor(list(LABEL_IDS.values()), dtype=torch.long)
CODE_TO_IDX = {c: i for i, c in enumerate(LABEL_IDS)}


def _labels_with_response(seq_len=10, resp_start=6, first_label_code="nr",
                          second_code=None):
    """labels tensor: -100 padding, response tokens starting at resp_start.

    Response = [first_label_code, second_code?, ...].  Label token ids are
    drawn from LABEL_IDS values; filler tokens are 7 (non-label).
    """
    labels = torch.full((1, seq_len), -100, dtype=torch.long)
    toks = [LABEL_IDS[first_label_code]] + ([7] * (seq_len - resp_start - 1))
    if second_code is not None:
        toks[1] = LABEL_IDS[second_code]
    for j, t in enumerate(toks[:seq_len - resp_start]):
        labels[0, resp_start + j] = t
    return labels


def test_label_position_finds_first():
    labels = _labels_with_response(first_label_code="sn")
    resp_tokens = labels[0][labels[0] != -100].tolist()
    idx = _label_position(resp_tokens, set(LABEL_IDS.values()))
    assert idx == 0


def test_bce_uses_offbyone_logit_position():
    """The logit that produced the label token at absolute position
    resp_start+idx lives at resp_start+idx-1 — pin the off-by-one."""
    resp_start, seq_len = 6, 10
    labels = _labels_with_response(seq_len, resp_start, "nr")
    V = 200
    logits = torch.zeros(1, seq_len, V)
    # put a distinctive value at the logit position of code "nr"
    logits[0, resp_start - 1, LABEL_IDS["nr"]] = 5.0
    label_vector = torch.zeros(1, 15)
    label_vector[0, CODE_TO_IDX["nr"]] = 1.0
    loss = _label_position_bce(logits, labels, label_vector, ID_TENSOR)
    # BCE with target 1 and logit 5 (vs 0 for the other 14 codes)
    import torch.nn.functional as F
    other = torch.tensor([LABEL_IDS[c] for c in LABEL_IDS if c != "nr"])
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([[5.0] + [0.0] * 14]), label_vector)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_mp_single_label_is_noop():
    """valid == {primary} -> delta == 0 for every sample."""
    labels = _labels_with_response(first_label_code="nr")
    V = 200
    logits = torch.randn(1, 10, V, generator=torch.Generator().manual_seed(0))
    lv = torch.zeros(1, 15)
    lv[0, CODE_TO_IDX["nr"]] = 1.0
    corr = _label_position_mp_correction(logits, labels, lv, ID_TENSOR, denom=1.0)
    assert corr.item() == pytest.approx(0.0, abs=1e-6)


def test_mp_delta_identity():
    """delta = logit[primary] - logsumexp(logit[valid]) <= 0, and the loss
    reduction equals the hand-computed soft-set correction."""
    resp_start = 6
    labels = _labels_with_response(10, resp_start, "nr", second_code="aw")
    V = 200
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(1, 10, V, generator=g)
    lv = torch.zeros(1, 15)
    lv[0, CODE_TO_IDX["nr"]] = 1.0
    lv[0, CODE_TO_IDX["aw"]] = 1.0
    denom = 42.0
    corr = _label_position_mp_correction(logits, labels, lv, ID_TENSOR, denom=denom)
    primary_logit = logits[0, resp_start - 1, LABEL_IDS["nr"]]
    valid = torch.tensor([LABEL_IDS["nr"], LABEL_IDS["aw"]])
    expected = (primary_logit - torch.logsumexp(
        logits[0, resp_start - 1, valid], dim=0)) / denom
    assert corr.item() == pytest.approx(expected.item(), rel=1e-5)
    assert corr.item() <= 0.0  # primary ∈ valid ⇒ delta ≤ 0


def test_mp_skips_when_no_response():
    labels = torch.full((1, 5), -100, dtype=torch.long)
    logits = torch.randn(1, 5, 200)
    lv = torch.zeros(1, 15)
    corr = _label_position_mp_correction(logits, labels, lv, ID_TENSOR, denom=1.0)
    assert corr.item() == 0.0
    bce = _label_position_bce(logits, labels, lv, ID_TENSOR)
    assert bce.item() == 0.0
