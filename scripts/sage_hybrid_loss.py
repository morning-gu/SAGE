"""
Hybrid loss for SAGE swift training: CE (generation) + BCE (multi-label logits).

This module adds a BCE loss that trains the 15 label-token logits against the
per-sample `label_vector` (15-dim, produced by convert_to_swift.py), so
co-occurring behaviors keep high logits and the sigmoid extraction at inference
yields correct multi-label probabilities.

The label-position logic mirrors sage_infer._extract_probs: scan the response
for label-token positions; take the FIRST position; read the 15 label logits
at (pos-1) [causal shift] and BCE them vs label_vector[B, 15].

Usage
-----
  # 1. self-test (no model / no swift needed). Verifies position logic always;
  #    if torch is installed, also verifies the BCE math.
  python scripts/sage_hybrid_loss.py --selftest

  # 2. launch a swift run with the hybrid loss
  python scripts/sage_hybrid_loss.py --model Qwen/Qwen3.5-4B \
      --dataset data/swift/train.jsonl --output-dir checkpoints/swift_lora

Dataset records MUST carry a `label_vector` field (see convert_to_swift.py).
SAGEDataCollator injects it into the batch; SAGETrainer.compute_loss uses it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sage_infer import BEHAVIOR_CODES, get_label_token_ids  # noqa: E402


import logging


def configure_logging(verbose: int = 0):
    """Silence swift/datasets INFO decoded-prompt chatter that buries the tqdm
    progress bar and per-logging_steps loss. --verbose 1 restores INFO when
    debugging data formatting.
    """
    level = logging.INFO if verbose else logging.WARNING
    for name in ("swift", "ms_swift", "datasets"):
        logging.getLogger(name).setLevel(level)


def _label_position(resp_tokens, id_set):
    """Find the first label-token position in the response (mirrors
    sage_infer._extract_probs which always takes the first label position).

    Args:
        resp_tokens: list[int] of response token ids (labels != -100).
        id_set: set[int] of the 15 label token ids.
    Returns:
        idx (int) relative index into resp_tokens of the label token whose
        predicting-logit we should read; None if no label token is present.
    """
    positions = [i for i, t in enumerate(resp_tokens) if t in id_set]
    if not positions:
        return None
    return positions[0]


def _label_position_bce(logits, labels, label_vector, label_ids):
    """BCE on the 15 label-token logits at the first label-code position.

    logits: [B, T, V], labels: [B, T] (-100 on non-response), label_vector: [B, 15].
    """
    import torch
    dev = logits.device
    B = logits.shape[0]
    id_list = label_ids.tolist() if torch.is_tensor(label_ids) else list(label_ids)
    id_set = set(id_list)
    id_tensor = torch.tensor(id_list, device=dev, dtype=torch.long)  # [15]

    sel_b, sel_pos, sel_tgt = [], [], []
    for b in range(B):
        lab = labels[b]
        resp_mask = lab != -100
        if not resp_mask.any():
            continue
        resp_start = int(resp_mask.nonzero(as_tuple=False)[0].item())
        resp_tokens = lab[resp_mask].tolist()
        idx = _label_position(resp_tokens, id_set)
        if idx is None:
            continue
        # logits[t] predicts token t+1 -> the logit that produced the label token
        # at absolute position (resp_start + idx) lives at (resp_start + idx - 1).
        logit_pos = resp_start + idx - 1
        if logit_pos < 0:
            continue
        sel_b.append(b)
        sel_pos.append(logit_pos)
        sel_tgt.append(label_vector[b])

    if not sel_b:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    b_idx = torch.tensor(sel_b, device=dev, dtype=torch.long)
    pos_idx = torch.tensor(sel_pos, device=dev, dtype=torch.long)
    tgt = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in sel_tgt]
                       ).to(dev).float()  # [n, 15]

    sel_logits = logits[b_idx, pos_idx, :]  # [n, V]
    label_logits = sel_logits[:, id_tensor]  # [n, 15]
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(label_logits, tgt)


def compute_hybrid_loss(model, inputs, label_ids, bce_weight=0.3):
    """CE (model SFT loss) + bce_weight * BCE (label-position multi-label)."""
    outputs = model(**inputs)
    ce = outputs.loss
    if label_ids is None or "label_vector" not in inputs or "labels" not in inputs:
        return ce, (ce, _zero(ce.device))
    bce = _label_position_bce(outputs.logits, inputs["labels"],
                              inputs["label_vector"], label_ids)
    return ce + bce_weight * bce, (ce, bce)


def _zero(dev):
    import torch
    return torch.tensor(0.0, device=dev)


class SAGEDataCollator:
    """Wraps a base collator and injects `label_vector` into the batch.

    swift's default collator pads input_ids/labels/pixel_values but drops custom
    fields. This re-adds label_vector so SAGETrainer.compute_loss can read it.
    """

    def __init__(self, base_collator):
        self.base = base_collator

    def __call__(self, features):
        batch = self.base(features)
        lvs = [f["label_vector"] for f in features if f.get("label_vector") is not None]
        if lvs:
            import torch
            batch["label_vector"] = torch.tensor(lvs, dtype=torch.float32)
        return batch


def make_trainer(label_ids, bce_weight=0.3):
    """Build a swift Seq2SeqTrainer subclass that adds BCE on the 15 label-token
    logits at the first label-code position, on top of swift's native CE.

    Reuses swift's compute_loss (forward + CE + per-token scaling) and layers the
    multi-label BCE on its outputs, so there is no double forward pass. Falls back
    to CE-only when label_ids is empty (no single-token codes resolved).
    """
    import torch
    from swift.trainers import Seq2SeqTrainer

    # get_label_token_ids iterates BEHAVIOR_CODES in order and keeps only the
    # single-token codes; align the BCE target to exactly that subset so the
    # [n, k] logits and [n, k] targets always match (k <= 15).
    id_tensor = (torch.tensor(list(label_ids.values()), dtype=torch.long)
                 if label_ids else None)
    valid_idx = (torch.tensor([i for i, c in enumerate(BEHAVIOR_CODES) if c in label_ids],
                               dtype=torch.long) if label_ids else None)

    class SAGETrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Pop label_vector so swift/HF's forward never sees the key.
            label_vector = inputs.pop("label_vector", None)
            labels = inputs.get("labels")
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            logits = getattr(outputs, "logits", None)
            if (id_tensor is not None and label_vector is not None and labels is not None
                    and logits is not None):
                lv = label_vector.to(logits.device)
                if valid_idx is not None:
                    lv = lv.index_select(-1, valid_idx.to(lv.device))
                bce = _label_position_bce(logits, labels, lv, id_tensor)
                loss = loss + bce_weight * bce
            return (loss, outputs) if return_outputs else loss

    return SAGETrainer


def _patch_sage_template(template):
    """Thread `label_vector` through swift's lazy-encode + collation.

    Idempotent.
    """
    import torch
    if getattr(template, "_sage_patched", False):
        return
    template._sage_patched = True

    orig_encode = template.encode

    def sage_encode(inputs, return_template_inputs=False, return_length=False, **kw):
        lv = inputs.get("label_vector") if isinstance(inputs, dict) else None
        enc = orig_encode(inputs, return_template_inputs=return_template_inputs,
                          return_length=return_length, **kw)
        if lv is not None and isinstance(enc, dict):
            enc["label_vector"] = lv
        return enc

    template.encode = sage_encode

    orig_collator = template.data_collator

    def sage_collator(batch, *, padding_to=None, **kw):
        res = orig_collator(batch, padding_to=padding_to, **kw)
        flat = sum(batch, start=[]) if (batch and isinstance(batch[0], list)) else batch
        lvs = [b.get("label_vector") for b in flat if b.get("label_vector") is not None]
        if lvs:
            res["label_vector"] = torch.tensor(lvs, dtype=torch.float32)
        return res

    template.data_collator = sage_collator


def _customize_chat_template(tokenizer):
    """Default enable_thinking to False in the tokenizer's chat_template.

    Qwen3 models ship a chat_template that defaults enable_thinking=True,
    causing the model to emit a chain-of-thought preamble before the actual
    response. SAGE training data has NO think block in the assistant target.
    """
    if not getattr(tokenizer, "chat_template", None):
        return
    line = "{% set enable_thinking = enable_thinking | default(false) %}"
    if line in tokenizer.chat_template.splitlines()[0]:
        return
    tokenizer.chat_template = line + "\n" + tokenizer.chat_template
    print("[sage] chat_template: enable_thinking defaults to False")


def selftest_positions():
    """Verify _label_position picks the right response index (always first)."""
    id_set = set(range(10, 25))

    # Case A: response = [nr, EOS], label first.
    resp = [10, 0]
    idx = _label_position(resp, id_set)
    assert idx == 0, f"single-code: expected idx 0, got {idx}"

    # Case B: multi-code response [prone=17, space=40, bowed=18, EOS]
    resp = [17, 40, 18, 0]
    assert _label_position(resp, id_set) == 0  # first -> prone

    # Case C: no label token present -> None (BCE skipped for this sample)
    assert _label_position([99, 0, 88], id_set) is None

    # Case D: think-block-prefixed response [think..., nr] -- label NOT first.
    # Must STILL find it via dynamic scan.
    resp = [77, 77, 77, 10]
    assert _label_position(resp, id_set) == 3

    print("  position logic: OK (always first label position, none->skip)")
    return True


def selftest_bce():
    """Verify the BCE math + causal-shift position. Requires torch."""
    import torch
    V = 100
    label_ids = torch.tensor([10 + i for i in range(15)])
    id_set = set(label_ids.tolist())

    # 3 prompt tokens (-100) then response [nr=10, EOS=0].
    # resp_start=3, idx=0, logit_pos = 3+0-1 = 2 -> read logits[0,2,:]
    labels = torch.full((1, 6), -100, dtype=torch.long)
    labels[0, 3] = 10
    labels[0, 4] = 0
    label_vector = torch.zeros(1, 15)
    label_vector[0, 0] = 1  # nr
    label_vector[0, 1] = 1  # aw  (co-occurring -- the case CE-only breaks)

    logits = torch.zeros(1, 6, V)
    logits[0, 2, 10] = 6.0   # nr high, target 1
    logits[0, 2, 11] = 5.0  # aw high, target 1  (BCE protects this)
    logits[0, 2, 12] = -6.0 # bl low, target 0
    bce = _label_position_bce(logits, labels, label_vector, label_ids)
    print(f"  matched  bce = {bce.item():.4f}  (expect ~0)")
    assert bce.item() < 0.05, "BCE ~0 when logits match the multi-label target"

    # Suppress aw's logit (what CE-only does to co-occurring labels) -> BCE must rise.
    logits[0, 2, 11] = -6.0
    bce2 = _label_position_bce(logits, labels, label_vector, label_ids)
    print(f"  mismatch bce = {bce2.item():.4f}  (expect large)")
    assert bce2.item() > 1.0, "BCE large when a co-occurring label logit is suppressed"

    # Noise at a non-label-predicting position must not affect BCE.
    logits[0, 2, 11] = 5.0
    logits[0, 0, 11] = 99.0
    bce3 = _label_position_bce(logits, labels, label_vector, label_ids)
    assert abs(bce3.item() - bce.item()) < 1e-6, "BCE must ignore other positions"
    print("  position isolation: OK")
    return True


def selftest():
    print("[sage_hybrid_loss self-test]")
    selftest_positions()
    try:
        import torch  # noqa: F401
        selftest_bce()
        print("selftest PASSED (positions + BCE)")
    except ModuleNotFoundError:
        print("selftest PASSED (positions only; install torch to also verify BCE)")


def run_training(a):
    """Launch an ms-swift (>=4.0) SFT run with the hybrid loss."""
    import os
    try:
        from swift.pipelines.train.sft import SwiftSft
    except Exception as e:
        print(f"[err] ms-swift not importable: {e}")
        print("      pip install 'ms-swift[llm]'  (this script targets ms-swift >=4.0)")
        return

    configure_logging(a.verbose)

    swift_args = [
        "--model", a.model,
        "--dataset", os.path.abspath(a.dataset),
        "--output_dir", a.output_dir,
        "--num_train_epochs", str(a.num_train_epochs),
        "--per_device_train_batch_size", str(a.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(a.gradient_accumulation_steps),
        "--learning_rate", str(a.learning_rate),
        "--warmup_ratio", str(a.warmup_ratio),
        "--logging_steps", str(a.logging_steps),
        "--save_strategy", a.save_strategy,
        "--bf16", "true",
        "--report_to", "none",
        "--remove_unused_columns", "false",
        "--tuner_type", "lora",
    ]
    if a.val_dataset:
        swift_args += ["--val_dataset", os.path.abspath(a.val_dataset)]

    class SAGESft(SwiftSft):
        def run(self):
            configure_logging(a.verbose)
            _patch_sage_template(self.template)
            _customize_chat_template(self.template.tokenizer)
            label_token_ids = get_label_token_ids(self.template.tokenizer)
            print(f"[sage] label token ids: {len(label_token_ids)}/{len(BEHAVIOR_CODES)} "
                  f"single-token")
            if len(label_token_ids) < len(BEHAVIOR_CODES):
                missing = [c for c in BEHAVIOR_CODES if c not in label_token_ids]
                print(f"[sage] WARN non-single-token codes (BCE skips them): {missing}")

            args = self.args
            train_dataset, val_dataset = self._prepare_dataset()
            if args.task_type == "seq_cls":
                args.problem_type = (args.problem_type
                                     or getattr(self.model.config, "problem_type", None))
            args.save_args()
            self.model = self.prepare_model(
                args, self.model, template=self.template, train_dataset=train_dataset)

            TrainerCls = make_trainer(label_token_ids, bce_weight=a.bce_weight)
            trainer = TrainerCls(
                model=self.model, args=args.training_args, template=self.template,
                train_dataset=train_dataset, eval_dataset=val_dataset)
            return self.train(trainer)

    SAGESft(swift_args).main()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the position/target self-test (no model/swift needed)")
    ap.add_argument("--model", default="", help="base model id/path")
    ap.add_argument("--dataset", default="data/swift/train.jsonl",
                    help="training dataset jsonl path")
    ap.add_argument("--val-dataset", default="data/swift/val.jsonl",
                    help="validation dataset jsonl path (empty string = skip val)")
    ap.add_argument("--output-dir", default="checkpoints/swift_lora",
                    help="LoRA adapter output directory")
    ap.add_argument("--bce-weight", type=float, default=0.3)
    ap.add_argument("--num-train-epochs", type=float, default=5)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--verbose", type=int, default=0,
                    help="0=silent, 1=INFO (debug data formatting)")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.model:
        ap.error("pass --model, or --selftest")
    run_training(a)


if __name__ == "__main__":
    main()
