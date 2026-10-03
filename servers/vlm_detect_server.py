"""VLM (large model) detection service (port 8010).

Independent model loading via transformers + optional LoRA (Plan B).
Model loading and inference logic are self-contained; the 15-class
taxonomy, system prompt, and output parsing are imported from
sage_check.constants (single source of truth).

Env vars:
   SAGE_VLM_MODEL            base model id or path (default: Qwen/Qwen3.5-4B)
   SAGE_VLM_LORA_PATH        LoRA adapter path (optional, empty = base only)
   SAGE_VLM_PORT             listen port (default: 8010)
   SAGE_SERVER_DEVICE        inference device (default: auto)
   SAGE_VLM_DTYPE            bf16 / fp16 / fp32 (default: fp16 — T4 fleet;
                             set bf16 explicitly on Ampere+ hosts)
   SAGE_VLM_MAX_PIXELS       image pixel cap (default: MAX_PIXELS from
                             sage_check.constants — MUST match training's
                             --max-pixels or train/eval resolutions diverge)
   SAGE_MODELS_CACHE_DIR     model cache dir (default: ./models)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI

from servers.common.device import resolve_device
from servers.common.image_utils import decode_base64_image
from servers.common.model_loader import resolve_model_path
from servers.common.protocol import VLMDetectRequest

# ---- Taxonomy / prompt / parsing: single source of truth ----
# Taxonomy: sage/taxonomy.py.  VLM prompt/parsing: sage_check/constants.py.
# Re-training is required after any taxonomy change.
from sage.taxonomy import BEHAVIOR_CODES  # noqa: E402
from sage_check.constants import (  # noqa: E402
    MAX_PIXELS,
    get_label_token_ids,
    parse_output,
    render_prompt,
)


class VLMDetector:
    """Independent VLM wrapper for the detection server (Plan B).

    Model loading and inference logic are self-contained; the taxonomy,
    prompt template, and output parsing come from sage_check.constants.
    """

    def __init__(self, model_path, device="cuda", dtype="fp16",
                 lora_path=None, max_pixels=MAX_PIXELS):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.lora_path = lora_path
        self.max_pixels = max_pixels
        self._model = None
        self._processor = None
        self._label_token_ids = None

    def _ensure_loaded(self):
        if self._model:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_path = resolve_model_path(self.model_path)
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        print(f"[vlm_detect] Loading {self.model_path} ...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        self._apply_max_pixels()
        load_kwargs = dict(device_map=self.device, trust_remote_code=True)
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, dtype=dt.get(self.dtype, torch.bfloat16),
                **load_kwargs)
        except TypeError:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=dt.get(self.dtype, torch.bfloat16),
                **load_kwargs)

        tok = getattr(self._processor, "tokenizer", self._processor)

        if self.lora_path:
            lora_dir = Path(self.lora_path)
            if any((lora_dir / f).exists() for f in
                   ("tokenizer.json", "tokenizer_config.json",
                    "tokenizer.model")):
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(
                    str(lora_dir), trust_remote_code=True)
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
            print(f"[vlm_detect] LoRA: {self.lora_path}")

        self._label_token_ids = get_label_token_ids(tok)
        print(f"[vlm_detect] Token mapping: "
              f"{len(self._label_token_ids)}/{len(BEHAVIOR_CODES)}")
        print("[vlm_detect] Ready.")

    def _apply_max_pixels(self):
        """Cap the processor's visual token budget.

        Must mirror training's --max-pixels (T4 protocol: MAX_PIXELS) —
        a mismatch means train and eval see different image resolutions.
        Qwen2VL-family image processors resize so that
        min_pixels <= pixels <= max_pixels; we only lower the cap and clamp
        min_pixels to stay consistent.
        """
        ip = getattr(self._processor, "image_processor", None)
        if ip is None or not self.max_pixels:
            return
        cur = getattr(ip, "max_pixels", None)
        if cur is None:
            return  # not a Qwen2VL-style processor; nothing to cap
        if cur > self.max_pixels:
            ip.max_pixels = self.max_pixels
        if getattr(ip, "min_pixels", 0) > ip.max_pixels:
            ip.min_pixels = ip.max_pixels
        print(f"[vlm_detect] max_pixels: {ip.max_pixels} "
              f"(protocol cap {self.max_pixels})")

    def detect(self, image_bytes, mode="no_reason", max_tokens=1024) -> dict:
        """Run VLM detection on a single image.

        Returns a dict matching VLMDetectResponse fields.
        """
        self._ensure_loaded()
        import io
        import torch
        from PIL import Image

        no_reason = (mode == "no_reason")
        effective_max_tokens = 1 if no_reason else max_tokens

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if image exceeds SAGE_VLM_MAX_PIXELS (matches training max_pixels).
        max_px = int(os.environ.get("SAGE_VLM_MAX_PIXELS", "0"))
        if max_px > 0:
            w, h = img.size
            if w * h > max_px:
                scale = (max_px / (w * h)) ** 0.5
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS)

        system_prompt = render_prompt(no_reason=no_reason)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Analyze the student's behavior and provide your classification."},
            ]},
        ]

        # Qwen3 models enable thinking by default; suppress it so the
        # model outputs only the classification response.  Fall back to
        # manual think-block injection for templates without enable_thinking.
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
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], tids], dim=-1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(tids)],
                    dim=-1)
            if "mm_token_type_ids" in inputs:
                inputs["mm_token_type_ids"] = torch.cat(
                    [inputs["mm_token_type_ids"], torch.zeros_like(tids)],
                    dim=-1)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=effective_max_tokens,
                do_sample=False, output_scores=True,
                return_dict_in_generate=True)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        ilen = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, ilen:]
        gen_text = self._processor.decode(gen_ids, skip_special_tokens=True)
        primary_code, explanation = parse_output(gen_text)
        probs = self._extract_probs(out, gen_ids)

        if not probs:
            probs = {code: 0.0 for code in BEHAVIOR_CODES}
            probs[primary_code] = 1.0

        return {
            "primary_code": primary_code,
            "explanation": explanation,
            "probabilities": probs,
            "raw_output": gen_text,
            "latency_ms": latency_ms,
            "error": "",
        }

    def _extract_probs(self, output, gen_ids) -> dict[str, float] | None:
        """Extract sigmoid probabilities at the first label-token position."""
        import torch
        if not self._label_token_ids:
            return None
        try:
            scores = torch.stack(output.scores, 1).sigmoid()
            id_set = set(self._label_token_ids.values())
            positions = [i for i, t in enumerate(gen_ids.tolist())
                         if t in id_set]
            if positions:
                idx = positions[0]
            else:
                pad = self._processor.tokenizer.pad_token_id
                non_pad = [i for i, t in enumerate(gen_ids.tolist())
                           if t != pad]
                if not non_pad:
                    return None
                idx = non_pad[0]
            return {c: round(float(scores[0, idx, t].item()), 4)
                    for c, t in self._label_token_ids.items()}
        except (RuntimeError, ValueError, IndexError) as e:
            print(f"[vlm_detect] Prob extraction: {e}")
            return None


app = FastAPI(title="SAGE VLM Detection")
_detector: VLMDetector | None = None


@app.on_event("startup")
def _load():
    global _detector
    try:
        max_pixels = int(os.environ.get("SAGE_VLM_MAX_PIXELS",
                                        str(MAX_PIXELS)))
    except ValueError:
        max_pixels = MAX_PIXELS
    _detector = VLMDetector(
        model_path=os.environ.get("SAGE_VLM_MODEL", "Qwen/Qwen3.5-4B"),
        device=resolve_device(),
        # Default fp16 for the T4 fleet (no native bf16); set SAGE_VLM_DTYPE
        # explicitly for bf16 on Ampere+ hosts.
        dtype=os.environ.get("SAGE_VLM_DTYPE", "fp16"),
        lora_path=os.environ.get("SAGE_VLM_LORA_PATH", "") or None,
        max_pixels=max_pixels,
    )
    _detector._ensure_loaded()


@app.post("/detect")
def detect(req: VLMDetectRequest) -> dict:
    img = decode_base64_image(req.image)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    try:
        return _detector.detect(buf.getvalue(), req.mode, req.max_tokens)
    except Exception as e:
        return {
            "primary_code": "",
            "explanation": "",
            "probabilities": {},
            "raw_output": "",
            "latency_ms": 0.0,
            "error": str(e),
        }


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _detector and _detector._model else "loading"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAGE_VLM_PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
