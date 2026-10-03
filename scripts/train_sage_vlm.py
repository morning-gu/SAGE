"""Train a SAGE VLM with a CE + BCE hybrid loss via ms-swift.

The BCE loss trains the 15 label-token logits against the per-sample
`label_vector` (15-dim, produced by scripts/convert_to_swift.py), so
co-occurring behaviors keep high logits and the sigmoid extraction at
inference yields correct multi-label probabilities.

Usage
-----
  python scripts/train_sage_vlm.py --model Qwen/Qwen3.5-4B \
      --dataset data/no_reason/train.jsonl --output-dir checkpoints/swift_lora

Dataset records MUST carry a `label_vector` field (see scripts/convert_to_swift.py).
_patch_sage_template injects it into the batch; SAGETrainer.compute_loss uses it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from transformers import TrainerCallback

REPO = Path(__file__).resolve().parent.parent
from sage.taxonomy import BEHAVIOR_CODES  # noqa: E402
from sage_check.constants import get_label_token_ids  # noqa: E402

# Subclassing (not duck-typing) matters: HF Trainer's callback_handler invokes
# every lifecycle event on every callback, so on_log-only classes crash on
# on_train_begin. TrainerCallback supplies no-op defaults for the rest.
from transformers import TrainerCallback  # noqa: E402


import logging


def configure_logging(verbose: int = 0):
    """Silence swift/datasets INFO decoded-prompt chatter that buries the tqdm
    progress bar and per-logging_steps loss. --verbose 1 restores INFO.
    """
    level = logging.INFO if verbose else logging.WARNING
    for name in ("swift", "ms_swift", "datasets"):
        logging.getLogger(name).setLevel(level)


def _label_position(resp_tokens, id_set):
    """Find the first label-token position in the response.

    Mirrors the inference-time sigmoid extraction which always takes
    the first label position.
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



def _label_position_mp_correction(logits, labels, label_vector, label_ids, denom):
    """Multi-positive correction that turns the hard single-label CE at the
    first label-code position into a soft-set CE (any valid label accepted).

    For each sample the per-token CE at the label position is
        L_hard = -log p(primary)            (what swift CE includes)
        L_soft = -log sum_{c in valid} p(c)  (what we want instead)
    so the replacement correction is
        delta = L_soft - L_hard = logit[primary] - logsumexp(logit[valid])  (<= 0)
    because primary is one of the valid tokens.  The log-partition cancels,
    so this is numerically stable and needs no full-vocab softmax.

    Single-label samples have valid == {primary} -> delta == 0 (no-op), so they
    keep the standard hard CE.  The total correction is summed over the batch and
    divided by denom (the CE reduction denominator) so it lives on the same
    scale as the token-averaged CE loss.
    """
    import torch
    dev = logits.device
    B = logits.shape[0]
    id_list = label_ids.tolist() if torch.is_tensor(label_ids) else list(label_ids)
    id_set = set(id_list)
    id_tensor = torch.tensor(id_list, device=dev, dtype=torch.long)  # [k]

    deltas = []
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
        logit_pos = resp_start + idx - 1
        if logit_pos < 0:
            continue
        primary_tok = resp_tokens[idx]
        if primary_tok not in id_set:
            continue  # multi-token primary code; correction cannot apply
        lv = label_vector[b]
        valid_mask = lv == 1
        if int(valid_mask.sum().item()) <= 1:
            continue  # single-label: delta == 0, skip
        valid_ids = id_tensor[valid_mask]            # [m] valid label token ids
        pos_logits = logits[b, logit_pos, :]          # [V]
        log_sum_valid = torch.logsumexp(pos_logits[valid_ids], dim=0)
        delta = pos_logits[primary_tok] - log_sum_valid  # <= 0
        deltas.append(delta)

    if not deltas:
        return torch.tensor(0.0, device=dev, requires_grad=True)
    return torch.stack(deltas).sum() / float(denom)

def make_trainer(label_ids, bce_weight=1.0, mp_weight=1.0, audit=None):
    """Build a swift Seq2SeqTrainer subclass with a hybrid loss:

        total = CE(full sequence)
              + mp_weight * multi_positive_correction(label position)
              + bce_weight * BCE(15 label-token logits)

    The multi-positive correction replaces the hard single-label CE at the
    first label-code position with a soft-set CE so that predicting *any* of
    a sample valid labels is accepted (no penalty for co-occurring labels).
    Single-label samples are unaffected (correction == 0).  mp_weight=1.0
    is an exact replacement; 0.0 disables it (reverts to hard-CE behaviour).

    Reuses swift's compute_loss (forward + CE) and layers the BCE on its
    outputs, so there is no double forward pass.  Falls back to CE-only
    when label_ids is empty (no single-token codes resolved).

    `audit` (optional dict) receives the latest raw loss components per
    micro-batch: keys 'ce' (CE returned by super().compute_loss before the
    corrections), 'mp' and 'bce' (raw correction / BCE values, before their
    weights).  Consumed by LossComponentsCallback for the per-log
    loss_components.jsonl audit trail.
    """
    import torch
    from swift.trainers import Seq2SeqTrainer

    if audit is None:
        audit = {}

    id_tensor = (torch.tensor(list(label_ids.values()), dtype=torch.long)
                 if label_ids else None)
    valid_idx = (torch.tensor([i for i, c in enumerate(BEHAVIOR_CODES) if c in label_ids],
                               dtype=torch.long) if label_ids else None)

    class SAGETrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
                # CE reduction denominator: num_items_in_batch (grad-accum) else
                # the count of non-ignored tokens in this micro-batch.
                if num_items_in_batch is not None:
                    denom = float(num_items_in_batch)
                else:
                    denom = float((labels != -100).sum().item()) or 1.0
                audit["ce"] = float(loss.detach())
                if mp_weight:
                    corr = _label_position_mp_correction(
                        logits, labels, lv, id_tensor, denom)
                    audit["mp"] = float(corr.detach())
                    loss = loss + mp_weight * corr
                bce = _label_position_bce(logits, labels, lv, id_tensor)
                audit["bce"] = float(bce.detach())
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


class LossComponentsCallback(TrainerCallback):
    """Lightweight TrainerCallback writing per-log loss components to
    <output_dir>/loss_components.jsonl.

    on_log fires every logging_steps with the trainer's logs dict (verified
    against ms-swift 4.4.2, which passes HF TrainingArguments through to the
    underlying HF Trainer).  Each line:
        {step, loss_total, loss_ce, loss_bce, loss_mp, mp_weight, bce_weight}
    Components that are unavailable (e.g. BCE skipped on a micro-batch) are
    written as null — this must never crash training.
    """

    FIELDS = ("loss_total", "loss_ce", "loss_bce", "loss_mp")

    def __init__(self, audit, mp_weight, bce_weight):
        self.audit = audit
        self.mp_weight = mp_weight
        self.bce_weight = bce_weight

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            if state is not None and not getattr(state, "is_world_process_zero", True):
                return
            logs = logs or {}
            rec = {
                "step": getattr(state, "global_step", None),
                "loss_total": logs.get("loss", logs.get("train_loss")),
                "loss_ce": self.audit.get("ce"),
                "loss_bce": self.audit.get("bce"),
                "loss_mp": self.audit.get("mp"),
                "mp_weight": self.mp_weight,
                "bce_weight": self.bce_weight,
            }
            out_dir = Path(getattr(args, "output_dir", ".") or ".")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "loss_components.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # never let the audit break training
            print(f"[sage] loss_components audit write failed (ignored): {e}")

    def on_train_end(self, args, state, control, **kwargs):
        self.audit.clear()


def run_training(a):
    """Launch an ms-swift (>=4.0) SFT run with the hybrid loss."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[sage] provenance: bce_weight={a.bce_weight} mp_weight={a.mp_weight} "
          f"seed={a.seed} epochs={a.num_train_epochs} dtype={a.dtype} model={a.model}")
    try:
        from swift.pipelines.train.sft import SwiftSft
    except Exception as e:
        print(f"[err] ms-swift not importable: {e}")
        print("      pip install 'ms-swift[llm]'  (this script targets ms-swift >=4.0)")
        return

    # Disable swift's Qwen3.5 linear attention SP patch (needs flash-linear-attention).
    # On single-GPU T4, the original transformers forward works without it.
    try:
        import swift.model.models.qwen as _swift_qwen
        _swift_qwen._patch_qwen3_5_linear_attention_sequence_parallel = lambda: None
    except Exception:
        pass

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
        "--report_to", "none",
        "--remove_unused_columns", "false",
        "--tuner_type", "lora",
        # ms-swift 4.4.2 verified: save_total_limit and seed are standard
        # TrainingArguments fields passed through swift's arg parsing.
        "--save_total_limit", str(a.save_total_limit),
        "--seed", str(a.seed),
        "--max_length", str(a.max_length),
        "--max_pixels", str(a.max_pixels),
    ]
    if a.val_dataset:
        swift_args += ["--val_dataset", os.path.abspath(a.val_dataset)]
    if a.adapters:
        swift_args += ["--adapters", os.path.abspath(a.adapters)]
    if a.resume_from_checkpoint:
        swift_args += ["--resume_from_checkpoint", a.resume_from_checkpoint]

    # Precision flag — mutually exclusive. Both are always set explicitly:
    # ms-swift's SftArguments defaults bf16=True, so --fp16 alone leaves
    # bf16=True and transformers rejects "At most one of fp16 and bf16".
    #   fp16: default; Turing GPUs without native bf16 (e.g. T4 16GB).
    #         Validate with a short smoke run (T4 16GB vs ~13GB historical peak).
    #   bf16: Ampere+ GPUs (historical A10 24GB protocol), opt-in.
    if a.dtype == "bf16":
        swift_args += ["--bf16", "true", "--fp16", "false",
                         "--torch_dtype", "bfloat16"]
    elif a.dtype == "fp16":
        swift_args += ["--fp16", "true", "--bf16", "false",
                         "--torch_dtype", "float16"]
    else:
        raise ValueError(f"unsupported --dtype: {a.dtype} (choose bf16 or fp16)")

    # Activation checkpointing: trades ~20-30% step time for a large
    # activation-memory drop — the margin that fits 5-epoch LoRA SFT into
    # T4 16GB when fp16 alone still peaks too high.
    if a.gradient_checkpointing:
        swift_args += ["--gradient_checkpointing", "true"]

    # Visual token cap (--max_pixels is already in the base swift_args above,
    # default 200704 = 448*448 = sage_check.constants.MAX_PIXELS, the T4
    # fleet protocol; the registry passes the same value explicitly).
    # PROTOCOL: every eval VLM server must apply the same cap
    # (SAGE_VLM_MAX_PIXELS, which the runner propagates from the experiment
    # config).

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

            TrainerCls = make_trainer(label_token_ids, bce_weight=a.bce_weight,
                              mp_weight=a.mp_weight, audit=audit)
            trainer = TrainerCls(
                model=self.model, args=args.training_args, template=self.template,
                train_dataset=train_dataset, eval_dataset=val_dataset)
            try:
                trainer.add_callback(
                    LossComponentsCallback(audit, mp_weight=a.mp_weight,
                                           bce_weight=a.bce_weight))
            except Exception as e:
                print(f"[sage] WARN loss-components callback not attached: {e}")
            return self.train(trainer)

    audit = {}  # latest raw loss components, drained by LossComponentsCallback
    SAGESft(swift_args).main()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model id/path")
    ap.add_argument("--dataset", default="data/no_reason/train.jsonl",
                    help="training dataset jsonl path")
    ap.add_argument("--val-dataset", default="data/no_reason/val.jsonl",
                    help="validation dataset jsonl path (empty string = skip val)")
    ap.add_argument("--output-dir", default="checkpoints/swift_lora",
                    help="LoRA adapter output directory")
    ap.add_argument("--adapters", default="",
                    help="path to an existing LoRA adapter dir to continue "
                         "training from (empty = fresh adapter on the base model)")
    ap.add_argument("--resume-from-checkpoint", default="",
                    help="checkpoint dir to resume from (restores optimizer/"
                         "scheduler state); empty = no resume")
    ap.add_argument("--bce-weight", type=float, default=1.0)
    # Preregistered MP keep/drop decision (docs/ablation_protocol.md §4,
    # frozen 2026-09-21): the val-split paired bootstrap returned DROP
    # (0/3 seeds' CI excl 0, pooled dF1 +0.44pp < +1pp, seed 43 negative),
    # so the default disables the correction — see
    # results/mp_keep_drop_decision.json.  The mp ablation configs set
    # --mp-weight explicitly, so E1 reproducibility is unaffected.
    ap.add_argument("--mp-weight", type=float, default=0.0,
                    help="multi-positive correction weight (1.0=exact soft-set "
                         "replacement of hard CE at label pos; 0.0=disable; "
                         "default 0 per the DROP decision)")
    ap.add_argument("--num-train-epochs", type=float, default=5)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16",
                    help="training precision: fp16 (default, T4/Turing — "
                         "no native bf16) or bf16 (Ampere+ A10 historical "
                         "protocol, opt-in)")
    ap.add_argument("--gradient-checkpointing", type=int, default=0,
                    help="1 enables activation checkpointing (T4 16GB "
                         "memory headroom; ~20-30%% slower steps)")
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="max checkpoints kept on disk (passed to swift args)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed (ms-swift 4.4.2 verified: passed through "
                         "to TrainingArguments.seed)")
    ap.add_argument("--verbose", type=int, default=0,
                    help="0=silent, 1=INFO (debug data formatting)")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="max sequence length for swift template (default 2048)")
    ap.add_argument("--max-pixels", type=int, default=200704,
                    help="max image pixels for vision encoder (default "
                         "448*448=200704 = sage_check.constants.MAX_PIXELS, "
                         "the T4 fleet protocol; must match the eval "
                         "servers' SAGE_VLM_MAX_PIXELS)")
    a = ap.parse_args()

    run_training(a)


if __name__ == "__main__":
    main()
