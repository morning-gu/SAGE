"""Experiment registry config loading and flag translation.

Shared by scripts/run_experiment.py (the only runner) so the merge semantics
live in one testable place:  common <- types[type] <- experiment, deep-merged,
then translated to argv flags for the target scripts.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Execution order across a batch: cheap eval-only first, then the primary VLM
# group, then the cheaper baselines.
TYPE_ORDER = ["eval_only", "vlm", "veto_eval", "yolo", "yolo_bce", "cnn"]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_defaults(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Path, defaults: dict) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged = deep_merge(defaults.get("common", {}), {})
    merged = deep_merge(merged, defaults.get("types", {}).get(cfg.get("type", ""), {}))
    merged = deep_merge(merged, cfg)
    merged.setdefault("est_hours", 1.0)
    # seed must reach the training script even if the yaml forgot it
    # (training types only — eval_only/veto_eval have no train section)
    train = merged.setdefault("train", {})
    if merged.get("type") not in ("eval_only", "veto_eval"):
        train.setdefault("args", {})
        train["args"].setdefault("seed", merged.get("seed", 42))
    elif not train:
        merged.pop("train", None)
    return merged


def discover_configs(path: Path, defaults: dict) -> list[dict]:
    paths = sorted(p for p in path.glob("*.yaml") if not p.name.startswith("_")) \
        if path.is_dir() else [path]
    return [load_config(p, defaults) for p in paths]


def sort_configs(configs: list[dict]) -> list[dict]:
    return sorted(configs, key=lambda c: (TYPE_ORDER.index(c["type"])
                                          if c["type"] in TYPE_ORDER else 99,
                                          c["exp_id"]))


def flags_from_args(args: dict) -> list[str]:
    """Map a yaml {flag_name: value} dict to ['--flag-name', 'value', ...].

    Underscores become dashes (argparse accepts either for dash-defined flags,
    but the canonical form is dashed).  Booleans: True -> flag only, False ->
    dropped.
    """
    out: list[str] = []
    for k, v in (args or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif v is None:
            continue
        else:
            out += [flag, str(v)]
    return out
