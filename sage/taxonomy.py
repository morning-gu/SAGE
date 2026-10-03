"""Behavior taxonomy — the domain core of SAGE.

This module IS the single source of truth for the 15-class behavior taxonomy
(codes, names, mappings).  Every layer — VLM client, recheck strategies,
fusion, metrics, data, servers — imports from here.  Re-training is required
after any taxonomy change.
"""
from __future__ import annotations

BEHAVIOR_CODES = [
    "nr", "aw", "bl", "ty", "ph", "sn", "ec", "pr",
    "bw", "cr", "tl", "tn", "sl", "rc", "lu",
]

CODE_TO_NAME = {
    "nr": "normal", "aw": "away", "bl": "blocked", "ty": "toy",
    "ph": "phone", "sn": "snack", "ec": "eyesclosed", "pr": "prone",
    "bw": "bowed", "cr": "chinrest", "tl": "tilt", "tn": "turn",
    "sl": "slope", "rc": "recline", "lu": "lookup",
}

BEHAVIOR_NAMES = list(CODE_TO_NAME.values())

NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}

NORMAL_CODE = "nr"
ANOMALY_NAMES = [n for n in BEHAVIOR_NAMES if n != "normal"]

# Evidence-cue grouping (C1's semantic-geometric complementarity axis;
# paper §3.2).  semantic-cued: judged by object/scene semantics or facial
# state; geometric-cued: judged by precise posture deviations. normal is
# excluded.
SEMANTIC_CUED = ["away", "blocked", "phone", "snack", "toy", "eyesclosed"]
GEOMETRIC_CUED = ["prone", "bowed", "chinrest", "tilt", "turn",
                  "slope", "recline", "lookup"]

assert sorted(SEMANTIC_CUED + GEOMETRIC_CUED) == sorted(ANOMALY_NAMES), \
    "cue grouping must partition the 14 anomaly classes"
