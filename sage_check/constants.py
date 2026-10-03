"""VLM prompt template, output parsing, and label-token mapping.

The taxonomy itself lives in ``sage.taxonomy`` (the domain core, shared by
all layers); this module keeps the VLM-client-side prompt/parsing concerns
and re-exports the taxonomy for backward compatibility.
"""
from __future__ import annotations

import re

# Taxonomy re-export (single source of truth: sage.taxonomy).
from sage.taxonomy import (  # noqa: F401
    BEHAVIOR_CODES,
    BEHAVIOR_NAMES,
    CODE_TO_NAME,
    NAME_TO_CODE,
)

# ---------------------------------------------------------------------------
# Visual token budget (T4 16GB protocol)
# ---------------------------------------------------------------------------
# Qwen2VL-family processors accept an image-pixel cap; 200704 = 448*448 bounds
# the pre-merge patch grid to 256 patches (~64 merged tokens) for the 1280x720
# classroom images.  Protocol history: 602112 (768*28*28) was the first T4 cap
# (b9ff47f); fa69ad0 lowered it to 200704 (~1.8x faster train/eval) BEFORE any
# d1 results were reported — all d1 experiments train AND evaluate at 200704.
# It must be applied identically to training (train_sage_vlm.py --max-pixels /
# _defaults.yaml vlm args) and to every eval VLM server (SAGE_VLM_MAX_PIXELS),
# including manually started E3 servers (their default is this constant),
# or train/eval see different resolutions.
MAX_PIXELS = 200704


# -- System prompt template --------------------------------------------------

SYSTEM_TEMPLATE = """You are an expert in student behavior evaluation, possessing strong visual comprehension and behavior identification skills.
Your task is to classify the provided student image into the most appropriate category from the list below as part of a single-person homework session behavior audit.

# Category List
- nr: normal study state (upright posture, gaze on study materials)
- aw: not in seat (nobody in frame, or student not seated: standing or away from seat)
- bl: occluded (body >50% or face occluded such that behavior cannot be determined)
- ty: playing with toys (hands holding non-study toys, excluding study-dependent items)
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


def render_prompt(no_reason: bool = False) -> str:
    """Render the system prompt. no_reason=True omits the explanation section."""
    from jinja2 import Template
    return Template(SYSTEM_TEMPLATE).render(no_reason=no_reason)


def parse_output(text: str) -> tuple[str, str]:
    """Strip think block, extract <explanation> + bare category code.

    Returns (primary_code, explanation).
    """
    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    clean = text[think.end():].strip() if think else text
    m = re.search(r"<explanation>(.*?)</explanation>", clean, re.DOTALL)
    explanation = m.group(1).strip() if m else ""
    no_expl = re.sub(r"<explanation>.*?</explanation>", "", clean, flags=re.DOTALL).strip()
    primary_code = ""
    for code in BEHAVIOR_CODES:
        if re.search(rf"\b{code}\b", no_expl.lower()):
            primary_code = code
            break
    if not primary_code:
        primary_code = "nr"
    return primary_code, explanation


def get_label_token_ids(tok) -> dict[str, int]:
    """Map 2-letter behavior codes to single-token IDs."""
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
