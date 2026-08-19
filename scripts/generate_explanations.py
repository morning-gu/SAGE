"""
Generate per-sample <explanation> text via three-tier progressive strategy.

Tier 1 (free_match):  Free classify with original system prompt.
                      If model code == GT primary -> accept explanation.
Tier 2 (redirect):    Model picked a co-occurring label (in GT labels but
                      not primary). Ask model to explain the GT primary
                      instead, then verify. -> redirect_match / redirect_mismatch
Tier 3 (forced):      Model completely disagreed. Label-conditioned generation
                      + verification. -> forced_match / forced_mismatch

Output is written to {input_dir}/{input_stem}_with_explanations.json.
Existing explanation fields in the INPUT file are always ignored and
regenerated. Resume support: if the OUTPUT file already exists and --resume
is passed, samples with model_code are skipped.

Usage:
  python scripts/generate_explanations.py --annotations data/sage_eval/annotations_v3.json
  python scripts/generate_explanations.py --annotations data/sage_eval/annotations_v3.json --resume
  python scripts/generate_explanations.py --annotations data/sage_eval/annotations_v3.json --dry-run
  python scripts/generate_explanations.py --annotations data/sage_eval/annotations_v3.json --split train --workers 2
"""

from __future__ import annotations

import argparse, base64, json, os, re, sys, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_infer import render_prompt, parse_output, CODE_TO_NAME, BEHAVIOR_CODES

NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}

# Fields to strip from the INPUT annotations (always regenerate explanations).
RESULT_FIELDS = ("explanation", "model_code", "model_label", "verification", "raw_output")

# Raised when Dashscope's input data inspection rejects an image
# (data_inspection_failed, HTTP 400). This is deterministic per image, so it
# must not be retried; the caller falls back to a template explanation.
class ContentBlocked(Exception):
    pass

# Raised when a corporate NTLM proxy SSL-intercepts traffic to the API host
# and returns an auth challenge / HTML login page instead of forwarding the
# request. Not retryable; the whole run should stop.
class ProxyBlocked(Exception):
    pass

# --- Label definitions (for forced prompt) ---

# Keep in sync with sage_infer.py SYSTEM_TEMPLATE (single source of truth).
LABEL_DEFS = {
    "normal": "upright posture, gaze on study materials",
    "away": "nobody in frame or not at desk",
    "blocked": "body >50% occluded or face occluded, behavior undeterminable",
    "toy": "playing with non-study toys, excluding study items",
    "phone": "holding electronic device, face clearly facing screen",
    "snack": "holding food, handling snack packaging, or eating",
    "eyesclosed": "eyes closed",
    "prone": "upper body slumped onto desk, head resting on desk or arms",
    "bowed": "head clearly lowered or face down, not touching desk",
    "chinrest": "both hands, elbows on desk, supporting chin or cheek; single hand doesn't count",
    "tilt": "head noticeably leaning left or right",
    "turn": "head noticeably turned to one side",
    "slope": "shoulders noticeably not level",
    "recline": "body leaning back against chair",
    "lookup": "face clearly upward, chin raised",
}

FREE_USER_MSG = "Analyze the student's behavior and provide your classification."

FORCED_SYSTEM = """You are an expert in student behavior evaluation. Write concise visual descriptions.
Given an image and the known behavior labels, write ONE concise sentence (10-30 words) in English describing the visual evidence supporting the primary label.
Rules: present tense, active voice, describe only visible physical cues, do NOT mention label names, do NOT use meta phrases.
Output ONLY the explanation sentence."""

VERIFY_SYSTEM = """You are a quality reviewer for student behavior annotations.
Your task: verify whether an explanation accurately describes visible evidence in a classroom image that supports a given behavior label.
Answer ONLY: YES or NO."""

PRICING = {"gpt-4o": {"input": 2.50, "output": 10.00}}
EST_INPUT_TOKENS = 915
EST_OUTPUT_TOKENS = 100


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image_path(sample):
    """Resolve the sample's image_path; return None if not found."""
    rec = sample.get("image_path")
    if not rec:
        return None
    p = Path(rec)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p if p.exists() else None


def _detect_windows_proxy():
    """Read the Windows system proxy from the registry (HKCU Internet Settings)."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
        if not enabled:
            return None
        server, _ = winreg.QueryValueEx(key, "ProxyServer")
        winreg.CloseKey(key)
    except OSError:
        return None
    if "=" in server:
        for part in server.split(";"):
            if part.startswith(("https=", "http=")):
                server = part.split("=", 1)[1]
                break
    server = server.strip()
    if not server:
        return None
    if "://" not in server:
        server = "http://" + server
    return server


def _build_api_url(base_url):
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return url


def _parse_sse(resp):
    parts = []
    for line in resp.text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("[DONE]", ""):
            continue
        try:
            chunk = json.loads(payload)
            choices = chunk.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    parts.append(content)
        except json.JSONDecodeError:
            continue
    return "".join(parts).strip()


def _extract_answer(text):
    """Extract final sentence from a response that may contain reasoning."""
    text = text.strip()
    if len(text) < 200 and text.count("\n") <= 1:
        return text.strip('"').strip("'").strip()
    segments = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    if not segments:
        return text.strip('"').strip("'").strip()
    answer = segments[-1]
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    return answer.strip()


def _call_api(api_url, api_key, model, image_b64, sys_prompt, user_content,
              max_retries=3, timeout=180, max_tokens=1024,
              proxies=None):
    """Generic API call. Returns raw model output text, or None on failure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.001,
        "stream": False,
        "enable_thinking": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json=payload, headers=headers,
                                 timeout=timeout, proxies=proxies)
            if resp.status_code != 200:
                body = resp.text[:500]
                if "data_inspection_failed" in body:
                    raise ContentBlocked(f"HTTP {resp.status_code}: {body}")
                auth_hdr = (resp.headers.get("WWW-Authenticate", "")
                            + resp.headers.get("Proxy-Authenticate", ""))
                low = body.lstrip().lower()
                if (("NTLM" in auth_hdr) or low.startswith(("<!doctype", "<html"))):
                    raise ProxyBlocked(
                        f"HTTP {resp.status_code} from network proxy "
                        f"({auth_hdr or 'HTML login page'}) at {api_url}")
                raise RuntimeError(f"HTTP {resp.status_code}: {body}")
            ct = resp.headers.get("Content-Type", "")
            if "text/event-stream" in ct:
                return _parse_sse(resp)
            return resp.json()["choices"][0]["message"]["content"].strip()
        except ContentBlocked:
            raise
        except ProxyBlocked:
            raise
        except Exception as e:
            status = resp.status_code if resp is not None else None
            retryable = status is None or status == 429 or status >= 500
            if retryable and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}: {e} [status={status}]")
                time.sleep(wait)
            else:
                print(f"    [error] {type(e).__name__}: {e} [status={status}]")
                return None
    return None


def _image_content(image_b64, text):
    """Build user content with text + image."""
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]


def _verify(api_url, api_key, model, image_b64, explanation, gt_primary, **kw):
    """Verify explanation against image. Returns True/False/None."""
    definition = LABEL_DEFS.get(gt_primary, "")
    text = (f'Behavior label: {gt_primary} ({definition})\n'
            f'Explanation: "{explanation}"\n'
            f'Does the explanation accurately describe visible physical evidence '
            f'in the image that supports this behavior?\n'
            f'Answer ONLY: YES or NO.')
    vkw = {k: v for k, v in kw.items() if k != "max_tokens"}
    raw = _call_api(api_url, api_key, model, image_b64, VERIFY_SYSTEM,
                    _image_content(image_b64, text),
                    max_tokens=10, **vkw)
    if raw is None:
        return None
    return "YES" in raw.upper()


def process_sample(sample, api_url, api_key, model, sys_prompt, **kw):
    """Three-tier progressive explanation. Returns (sample_id, result_dict, error)."""
    img_path = resolve_image_path(sample)
    if not img_path:
        return sample["sample_id"], None, "image_not_found"
    img_b64 = encode_image(img_path)

    gt_primary = sample.get("primary_label", "normal")
    gt_code = NAME_TO_CODE.get(gt_primary, "nr")

    # --- Tier 1: free classify ---
    try:
        raw = _call_api(api_url, api_key, model, img_b64, sys_prompt,
                        _image_content(img_b64, FREE_USER_MSG), **kw)
    except ContentBlocked as e:
        print(f"    [content-blocked] {sample['sample_id']}: {e}")
        definition = LABEL_DEFS.get(gt_primary, "the labeled behavior")
        return sample["sample_id"], {
            "model_code": gt_code,
            "model_label": gt_primary,
            "raw_output": f"[content_blocked] {str(e)[:200]}",
            "explanation": f"The student exhibits {definition}.",
            "verification": "content_blocked",
        }, None

    if not raw:
        return sample["sample_id"], None, "api_error"

    result = parse_output(raw)
    model_code = result.primary_code
    model_label = CODE_TO_NAME.get(model_code, model_code)
    gt_labels = sample.get("labels", {})
    primary_match = (model_code == gt_code)
    label_match = gt_labels.get(CODE_TO_NAME.get(model_code, ""), 0) == 1

    base = {
        "model_code": model_code,
        "model_label": model_label,
        "raw_output": raw[:500],
    }

    if primary_match:
        return sample["sample_id"], {**base,
            "explanation": result.explanation,
            "verification": "free_match",
        }, None

    # --- Tier 2: label_match -> redirect ---
    if label_match:
        redirect_text = (
            f"You classified this student as {model_label} ({model_code}).\n"
            f"The ground truth primary label is {gt_primary} ({gt_code}).\n"
            f"Both behaviors may co-occur. Please explain why {gt_primary} "
            f"applies to this image.\n"
            f"Place your explanation between <explanation> and </explanation> tags."
        )
        rraw = _call_api(api_url, api_key, model, img_b64, sys_prompt,
                         _image_content(img_b64, redirect_text), **kw)
        if rraw:
            rresult = parse_output(rraw)
            expl = rresult.explanation or _extract_answer(rraw)
            if expl:
                verified = _verify(api_url, api_key, model, img_b64,
                                   expl, gt_primary, **kw)
                if verified is True:
                    return sample["sample_id"], {**base,
                        "explanation": expl,
                        "verification": "redirect_match",
                    }, None
        return sample["sample_id"], {**base,
            "explanation": "",
            "verification": "redirect_mismatch",
        }, None

    # --- Tier 3: mismatch -> forced + verify ---
    active = [(n, LABEL_DEFS[n]) for n, v in gt_labels.items()
              if v == 1 and n in LABEL_DEFS]
    label_lines = "\n".join(f"- {n}: {d}" for n, d in active)
    forced_text = (f"This image has been annotated with these behaviors:\n"
                   f"{label_lines}\n\n"
                   f"Primary (dominant) label: {gt_primary}\n\n"
                   f"Write the explanation sentence:")
    fraw = _call_api(api_url, api_key, model, img_b64, FORCED_SYSTEM,
                     _image_content(img_b64, forced_text), **kw)
    if fraw:
        expl = _extract_answer(fraw)
        if expl:
            verified = _verify(api_url, api_key, model, img_b64,
                               expl, gt_primary, **kw)
            if verified is True:
                return sample["sample_id"], {**base,
                    "explanation": expl,
                    "verification": "forced_match",
                }, None
    return sample["sample_id"], {**base,
        "explanation": "",
        "verification": "forced_mismatch",
    }, None


def estimate_cost(num_samples, model):
    it, ot = num_samples * EST_INPUT_TOKENS, num_samples * EST_OUTPUT_TOKENS
    r = {"input_tokens": it, "output_tokens": ot}
    p = PRICING.get(model)
    if p:
        r["cost"] = round((it / 1e6) * p["input"] + (ot / 1e6) * p["output"], 2)
    return r


def save_output(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def _print_summary(samples):
    processed = [s for s in samples if s.get("model_code")]
    if not processed:
        print("(no samples processed)")
        return
    n = len(processed)
    tiers = Counter(s.get("verification", "") for s in processed)
    with_expl = sum(1 for s in processed if s.get("explanation"))

    print(f"\n--- Three-Tier Progressive Results ---")
    print(f"Total processed:     {n}")
    print(f"  Tier 1 free_match:       {tiers.get('free_match', 0):>4d} ({100*tiers.get('free_match',0)/n:.1f}%)")
    print(f"  Tier 2 redirect_match:   {tiers.get('redirect_match', 0):>4d} ({100*tiers.get('redirect_match',0)/n:.1f}%)")
    print(f"  Tier 2 redirect_mismatch:{tiers.get('redirect_mismatch', 0):>4d} ({100*tiers.get('redirect_mismatch',0)/n:.1f}%)")
    print(f"  Tier 3 forced_match:     {tiers.get('forced_match', 0):>4d} ({100*tiers.get('forced_match',0)/n:.1f}%)")
    print(f"  Tier 3 forced_mismatch:  {tiers.get('forced_mismatch', 0):>4d} ({100*tiers.get('forced_mismatch',0)/n:.1f}%)")
    print(f"  Content-blocked fallback:{tiers.get('content_blocked', 0):>4d} ({100*tiers.get('content_blocked',0)/n:.1f}%)")
    print(f"  Training-ready (has expl): {with_expl}/{n} ({100*with_expl/n:.1f}%)")

    by_class = {}
    for s in processed:
        gt = s.get("primary_label", "?")
        by_class.setdefault(gt, {"total": 0, "match": 0})
        by_class[gt]["total"] += 1
        if s.get("verification") == "free_match":
            by_class[gt]["match"] += 1
    print(f"\n--- Per-class Tier-1 Accuracy ---")
    print(f"{'Class':<14s} {'Match':>6s} {'Total':>6s} {'Acc':>7s}")
    print("-" * 40)
    for cls in sorted(by_class, key=lambda x: by_class[x]["total"], reverse=True):
        d = by_class[cls]
        acc = 100 * d["match"] / d["total"] if d["total"] > 0 else 0
        print(f"{cls:<14s} {d['match']:>6d} {d['total']:>6d} {acc:>6.1f}%")

    confused = Counter()
    for s in processed:
        gt = s.get("primary_label", "?")
        model = s.get("model_label", "?")
        if gt != model:
            confused[(gt, model)] += 1
    if confused:
        print(f"\n--- Top Confused Pairs (GT -> Model) ---")
        for (gt, model), count in confused.most_common(10):
            print(f"  {gt:<14s} -> {model:<14s}  ({count})")


def _derive_output_path(annotations_path):
    """Derive output path: {stem}_with_explanations.json in the same directory."""
    p = Path(annotations_path)
    return p.parent / f"{p.stem}_with_explanations.json"


def main():
    ap = argparse.ArgumentParser(
        description="Generate per-sample <explanation> via three-tier progressive strategy.")
    ap.add_argument("--annotations", required=True,
                    help="Input annotations JSON file")
    ap.add_argument("--model", default="qwen3.7-plus",
                    help="VLM model name for API calls")
    ap.add_argument("--base-url",
                    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    help="VLM API base URL")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--split", default="", help="Filter by split field")
    ap.add_argument("--sample", default="", help="Process only this sample_id")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip samples already processed (have model_code) in the output file")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess all samples even if model_code is set")
    a = ap.parse_args()

    # Always use reason mode (code + explanation) for the system prompt.
    sys_prompt = render_prompt(no_reason=False)
    output_path = _derive_output_path(a.annotations)

    proxy_url = _detect_windows_proxy()
    proxies = ({"http": proxy_url, "https": proxy_url} if proxy_url else None)
    kw = dict(max_retries=a.max_retries, timeout=a.timeout,
              max_tokens=a.max_tokens, proxies=proxies)

    with open(a.annotations, encoding="utf-8") as f:
        samples = json.load(f)

    # Always strip existing result fields from the INPUT annotations -- we
    # regenerate explanations regardless of what the input file carries.
    for s in samples:
        for k in RESULT_FIELDS:
            s.pop(k, None)

    if a.split:
        samples = [s for s in samples if s.get("split") == a.split]
        print(f"Filtered to split='{a.split}': {len(samples)} samples")
    if a.sample:
        samples = [s for s in samples if s["sample_id"] == a.sample]
        if not samples:
            print(f"Sample '{a.sample}' not found.")
            return

    # Resume: load results from the output file and merge into samples.
    if a.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            prev = json.load(f)
        existing = {s["sample_id"]: s for s in prev if s.get("model_code")}
        for s in samples:
            if s["sample_id"] in existing:
                e = existing[s["sample_id"]]
                for k in RESULT_FIELDS:
                    if k in e:
                        s[k] = e[k]
        print(f"[resume] loaded {len(existing)} existing results from {output_path}")

    if a.force:
        todo = list(samples)
        print("[force] reprocessing all samples regardless of existing model_code")
    else:
        todo = [s for s in samples if not s.get("model_code")]
    print(f"Total: {len(samples)}, done: {len(samples)-len(todo)}, to process: {len(todo)}")
    print(f"Output: {output_path}")
    if not todo:
        print("Nothing to do.")
        save_output(samples, output_path)
        _print_summary(samples)
        return
    if a.dry_run:
        est = estimate_cost(len(todo), a.model)
        print(f"\nDry run (model={a.model}):")
        print(f"  Input tokens:  ~{est['input_tokens']:,}")
        print(f"  Output tokens: ~{est['output_tokens']:,}")
        if "cost" in est:
            print(f"  Est. cost:     ~${est['cost']:.2f}")
        return

    api_key = a.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: no API key. Pass --api-key or set OPENAI_API_KEY.")
        sys.exit(1)
    api_url = _build_api_url(a.base_url)
    print(f"[api] url={api_url}  model={a.model}")
    print(f"[proxy] {proxy_url or 'direct (no proxy)'}")

    missing = sum(1 for s in todo if not resolve_image_path(s))
    if missing:
        print(f"[warn] {missing}/{len(todo)} images not found")

    errors = 0
    t0 = time.perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        future_to_sample = {
            pool.submit(process_sample, s, api_url, api_key, a.model, sys_prompt,
                        **kw): s
            for s in todo
        }
        for future in as_completed(future_to_sample):
            try:
                sid, result, err = future.result()
            except ProxyBlocked as e:
                print(f"\n[abort] {e}")
                print("[abort] Stopping: requests to the API are being blocked by a "
                      "network proxy. Re-run with an internal --base-url host already "
                      "in NO_PROXY, or route via a local NTLM relay (cntlm) / off-network. "
                      "No samples lost (use --resume).")
                for f in future_to_sample:
                    f.cancel()
                break
            sample = future_to_sample[future]
            if result:
                sample["explanation"] = result.get("explanation", "")
                sample["model_code"] = result.get("model_code", "")
                sample["model_label"] = result.get("model_label", "")
                sample["verification"] = result.get("verification", "")
                if result.get("raw_output"):
                    sample["raw_output"] = result["raw_output"]
            else:
                errors += 1
                print(f"  ERROR {sid}: {err}")
            completed += 1
            if completed % 10 == 0:
                save_output(samples, output_path)
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(todo) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(todo)}] {rate:.1f}/s, errors={errors}, ETA={eta:.0f}s")

    save_output(samples, output_path)
    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Processed: {completed-errors}/{len(todo)} succeeded, {errors} failed")
    print(f"Time: {elapsed:.1f}s")
    _print_summary(samples)
    print(f"\nOutput: {output_path}")
    print(f"\n{'=' * 60}")
    print(f"Next: python scripts/convert_to_swift.py --mode reason --annotations {output_path}")


if __name__ == "__main__":
    main()
