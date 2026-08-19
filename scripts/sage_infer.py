"""
SAGE Student Behavior Inference.

Supports base model and LoRA fine-tuned model inference, with no_reason
(code only) and reason (code + explanation) modes.

Usage:
  python scripts/sage_infer.py --model-path Qwen/Qwen3.5-4B --image photo.jpg
  python scripts/sage_infer.py --model-path Qwen/Qwen3.5-4B --image photo.jpg --mode reason
  python scripts/sage_infer.py --model-path Qwen/Qwen3.5-4B --lora-path ./lora --image photo.jpg
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

BEHAVIOR_CODES = [
    "nr", "aw", "bl", "ty", "ph", "sn", "ec", "pr", "bw", "cr",
    "tl", "tn", "sl", "rc", "lu",
]

CODE_TO_NAME = {
    "nr": "normal", "aw": "away", "bl": "blocked", "ty": "toy",
    "ph": "phone", "sn": "snack", "ec": "eyesclosed", "pr": "prone",
    "bw": "bowed", "cr": "chinrest", "tl": "tilt", "tn": "turn",
    "sl": "slope", "rc": "recline", "lu": "lookup",
}

# SINGLE SOURCE OF TRUTH: The category descriptions in SYSTEM_TEMPLATE below
# define the canonical criteria for all 15 behavior classes. convert_to_swift.py
# (LABEL_EXPLANATION) and generate_explanations.py (LABEL_DEFS) must stay in sync.
# When you change a definition here, regenerate training data:
#   python scripts/convert_to_swift.py --mode reason
SYSTEM_TEMPLATE = """You are an expert in student behavior evaluation, possessing strong visual comprehension and behavior identification skills.
Your task is to classify the provided student image into the most appropriate category from the list below as part of a classroom learning quality audit.

# Category List
- nr: normal study state (upright posture, gaze on study materials)
- aw: not in seat (nobody in frame, or student not seated: standing or away from seat)
- bl: occluded (body >50% or face occluded such that behavior cannot be determined)
- ty: playing with toys (hands manipulating non-study toys, excluding study-dependent items)
- ph: using an electronic device (holding the device, face clearly facing the screen)
- sn: eating snacks (holding food, or handling snack packaging, or eating)
- ec: eyes closed
- pr: lying on the desk (upper body slumped onto the desk, head resting on the desk or arms)
- bw: head down (head clearly lowered or face down, not touching the desk)
- cr: chin resting (both hands with elbows on the desk supporting the chin or cheek; a single hand does not count)
- tl: head tilted (head noticeably leaning left or right)
- tn: head turned (head noticeably turned to one side)
- sl: uneven shoulders (shoulders noticeably not level)
- rc: reclining (body markedly leaning back against the chair)
- lu: looking up (face clearly upward, chin raised)

# Dominant Category
When several behaviors are visible at once, use your own visual judgment to identify the single most dominant behavior -- the one that most characterizes the student's current state and most directly determines their engagement with learning.
# Instructions
{% if no_reason %}
- Identify the single most dominant category ID for the input image. Output ONLY the category ID, nothing else.
{% else %}
- Identify the single most dominant category ID for the input image.
- On the next line, provide a concise justification for your choice, placing it between <explanation> and </explanation> tags.
{% endif %}

---"""


@dataclass
class InferenceResult:
    primary_code: str = ""
    explanation: str = ""
    probs: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in BEHAVIOR_CODES})
    raw_output: str = ""
    latency_ms: float = 0.0

    @property
    def primary_name(self) -> str:
        return CODE_TO_NAME.get(self.primary_code, self.primary_code)


def render_prompt(no_reason: bool = False) -> str:
    from jinja2 import Template
    return Template(SYSTEM_TEMPLATE).render(no_reason=no_reason)


def parse_output(text: str) -> InferenceResult:
    """Strip think block, extract <explanation> + bare category code."""
    r = InferenceResult(raw_output=text)
    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    clean = text[think.end():].strip() if think else text
    m = re.search(r"<explanation>(.*?)</explanation>", clean, re.DOTALL)
    r.explanation = m.group(1).strip() if m else ""
    no_expl = re.sub(r"<explanation>.*?</explanation>", "", clean, flags=re.DOTALL).strip()
    for code in BEHAVIOR_CODES:
        if re.search(rf"\b{code}\b", no_expl.lower()):
            r.primary_code = code
            break
    if not r.primary_code:
        r.primary_code = "nr"
    return r


def get_label_token_ids(tok) -> dict[str, int]:
    """Map 2-letter codes to single-token IDs."""
    m = {}
    for code in BEHAVIOR_CODES:
        ids = tok.encode(code, add_special_tokens=False)
        if len(ids) == 1:
            m[code] = ids[0]
        else:
            ids2 = tok.encode(" " + code, add_special_tokens=False)
            if len(ids2) == 1:
                m[code] = ids2[0]
    return m


def download_model(model_id: str, source: str = "auto") -> str:
    if Path(model_id).is_dir():
        return model_id
    print(f"[download] {model_id} | {source}")
    if source != "huggingface":
        try:
            from modelscope import snapshot_download
            return snapshot_download(model_id, cache_dir="./models")
        except Exception as e:
            if source == "modelscope":
                raise
            print(f"[download] ModelScope failed: {e}")
    from huggingface_hub import snapshot_download as hf
    return hf(model_id, cache_dir="./models")


class LocalModel:
    """Local VLM wrapper supporting base model and LoRA adapter inference."""

    def __init__(self, model_path, device="cuda", dtype="bf16",
                 lora_path=None, download_source="auto"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.lora_path = lora_path
        self.download_source = download_source
        self._model = None
        self._processor = None
        self._label_token_ids = None

    def _ensure_loaded(self):
        if self._model:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_path = download_model(self.model_path, self.download_source)
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        print(f"[local] Loading {self.model_path} ...")
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        load_kwargs = dict(device_map=self.device, trust_remote_code=True)
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, dtype=dt.get(self.dtype, torch.bfloat16), **load_kwargs)
        except TypeError:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, torch_dtype=dt.get(self.dtype, torch.bfloat16), **load_kwargs)

        tok = getattr(self._processor, "tokenizer", self._processor)

        if self.lora_path:
            lora_dir = Path(self.lora_path)
            if any((lora_dir / f).exists() for f in
                   ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")):
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(lora_dir), trust_remote_code=True)
                if hasattr(self._processor, "tokenizer"):
                    self._processor.tokenizer = tok
            cfg = self._model.config
            vocab_size = getattr(cfg, "vocab_size", None)
            if vocab_size is None:
                text_cfg = getattr(cfg, "text_config", None)
                vocab_size = getattr(text_cfg, "vocab_size", None)
            if vocab_size is not None and len(tok) > vocab_size:
                self._model.resize_token_embeddings(len(tok))
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            print(f"[local] LoRA: {self.lora_path}")

        self._label_token_ids = get_label_token_ids(tok)
        print(f"[local] Token mapping: {len(self._label_token_ids)}/{len(BEHAVIOR_CODES)}")
        print("[local] Ready.")

    def infer(self, image_bytes, system_prompt, max_tokens=1024):
        import torch
        from PIL import Image
        import io

        self._ensure_loaded()

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Analyze the student's behavior and provide your classification."},
            ]},
        ]
        # Qwen3 models enable thinking by default; suppress it so the model
        # outputs only the classification response. Fall back to manual
        # think-block injection for templates without enable_thinking support.
        try:
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=False).to(self._model.device)
        except (TypeError, KeyError):
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self._model.device)
            tk = self._processor.tokenizer.encode(
                "<think>\n\n</think>\n\n", add_special_tokens=False)
            tids = torch.tensor([tk], dtype=inputs["input_ids"].dtype,
                                device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], tids], dim=-1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(tids)], dim=-1)
            if "mm_token_type_ids" in inputs:
                inputs["mm_token_type_ids"] = torch.cat(
                    [inputs["mm_token_type_ids"], torch.zeros_like(tids)], dim=-1)

        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True)

        ilen = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, ilen:]
        gen_text = self._processor.decode(gen_ids, skip_special_tokens=True)
        probs = self._extract_probs(out, gen_ids)
        return gen_text, probs

    def _extract_probs(self, output, gen_ids) -> dict[str, float] | None:
        """Extract sigmoid probabilities at the first label-token position."""
        import torch
        if not self._label_token_ids:
            return None
        try:
            scores = torch.stack(output.scores, 1).sigmoid()
            id_set = set(self._label_token_ids.values())
            positions = [i for i, t in enumerate(gen_ids.tolist()) if t in id_set]
            if positions:
                idx = positions[0]
            else:
                pad = self._processor.tokenizer.pad_token_id
                non_pad = [i for i, t in enumerate(gen_ids.tolist()) if t != pad]
                if not non_pad:
                    return None
                idx = non_pad[0]
            return {c: round(float(scores[0, idx, t].item()), 4)
                    for c, t in self._label_token_ids.items()}
        except (RuntimeError, ValueError, IndexError) as e:
            print(f"  [warn] Prob extraction: {e}")
            return None


def main():
    p = argparse.ArgumentParser(description="SAGE inference (base model + LoRA)")
    p.add_argument("--model-path", required=True,
                   help="Base model path or HuggingFace/ModelScope ID")
    p.add_argument("--image", required=True, help="Path to the input image")
    p.add_argument("--lora-path", default="", help="LoRA adapter path (optional)")
    p.add_argument("--mode", choices=["no_reason", "reason"], default="no_reason",
                   help="no_reason: output code only; reason: output code + explanation")
    p.add_argument("--download-source", default="auto",
                   choices=["auto", "modelscope", "huggingface"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = p.parse_args()

    no_reason = (a.mode == "no_reason")
    prompt = render_prompt(no_reason=no_reason)
    print(f"{'=' * 60}\nModel: {a.model_path} | mode={a.mode}"
          + (f" | LoRA: {a.lora_path}" if a.lora_path else "")
          + f"\n{'=' * 60}\n{prompt}\n{'=' * 60}\n")

    from PIL import Image
    import io
    img = Image.open(a.image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    model = LocalModel(a.model_path, a.device, a.dtype,
                       a.lora_path or None, a.download_source)

    t0 = time.perf_counter()
    raw, probs = model.infer(
        image_bytes=buf.getvalue(), system_prompt=prompt,
        max_tokens=1 if no_reason else 1024,
        )
    ms = (time.perf_counter() - t0) * 1000

    r = parse_output(raw)
    r.latency_ms = round(ms, 1)
    if probs:
        r.probs = probs
    else:
        r.probs[r.primary_code] = 1.0

    print(f"\n{'=' * 60}")
    print(f"Primary: {r.primary_name} ({r.primary_code}) "
          f"(p={r.probs.get(r.primary_code, 0):.4f})")
    print(f"Latency: {r.latency_ms:.0f} ms")
    if r.explanation:
        print(f"Explanation: {r.explanation}")
    print("\nAll probabilities:")
    for code in BEHAVIOR_CODES:
        prob = r.probs.get(code, 0.0)
        print(f"  {CODE_TO_NAME[code]:<12s} ({code}) {prob:.4f} {'#' * int(prob * 40)}")
    print(f"\n{'=' * 60}\nRaw output:\n{raw}\n{'=' * 60}")


if __name__ == "__main__":
    main()
