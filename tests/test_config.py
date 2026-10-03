"""Round-trip validation for the experiment registry.

For every experiment yaml: the flags produced by sage.config.flags_from_args
must be accepted by the target script's argparse.  Catches the silent
yaml-key <-> flag-name mismatch (flag renamed in a script, yaml still passes
the old key, failure only at runtime) as a CI error.

The scripts are parsed with ast (no import, no GPU/service needed).
"""
from __future__ import annotations

import ast
from pathlib import Path

from sage import config
from sage.paths import CONFIG_DIR


def _script_flag_names(script: Path) -> set[str]:
    """All long-option names (both dashed and underscore forms) that a
    script's argparse accepts, via static ast walk."""
    src = script.read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                name = arg.value
                if name.startswith("--"):
                    names.add(name[2:])
                    names.add(name[2:].replace("-", "_"))
    return names


def _resolve(repo_relative: str) -> Path:
    p = Path(repo_relative)
    return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p


def test_no_duplicate_argparse_flags():
    """A script must not add_argument the same option twice.

    argparse raises ArgumentError at *runtime* (main() time) on conflict —
    the T4 fleet hit this when a merge left two --max-pixels definitions in
    train_sage_vlm.py and every vlm training died at startup.  A static
    duplicate check makes it a test failure instead.
    """
    offenders: dict[str, list[str]] = {}
    for script in (Path(__file__).resolve().parent.parent / "scripts").glob("*.py"):
        seen: dict[str, int] = {}
        for node in ast.walk(ast.parse(script.read_text(encoding="utf-8-sig"))):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("--"):
                    seen[arg.value] = seen.get(arg.value, 0) + 1
        dups = [f for f, n in seen.items() if n > 1]
        if dups:
            offenders[script.name] = dups
    assert offenders == {}, f"duplicate argparse flags: {offenders}"


def _collect_flag_calls(cfg: dict, out: list[tuple[str, dict]]):
    train = cfg.get("train", {}) or {}
    if train:
        if "steps" in train:
            for st in train["steps"]:
                out.append((st["script"], st.get("args", {}) or {}))
        else:
            out.append((train["script"], train.get("args", {}) or {}))
    ev = cfg.get("eval", {}) or {}
    if ev:
        out.append((ev["script"], ev.get("args", {}) or {}))


# Scripts referenced by the registry but not yet written.  A registry
# entry pointing at a missing script must be a *deliberate* state, not
# something that only surfaces at runtime.  Keep empty when everything exists.
KNOWN_PENDING_SCRIPTS: set[str] = set()


def test_all_registry_flags_are_accepted_by_target_scripts():
    defaults = config.load_defaults(CONFIG_DIR / "_defaults.yaml")
    configs = config.discover_configs(CONFIG_DIR, defaults)
    assert len(configs) >= 30, "registry unexpectedly small — did it move?"

    flag_cache: dict[str, set[str]] = {}
    problems: list[str] = []
    checked = 0
    for cfg in configs:
        calls: list[tuple[str, dict]] = []
        _collect_flag_calls(cfg, calls)
        for script_rel, args in calls:
            script = _resolve(script_rel)
            if not script.exists():
                assert script_rel in KNOWN_PENDING_SCRIPTS, \
                    f"{cfg['exp_id']}: missing script {script_rel} — add it to " \
                    "KNOWN_PENDING_SCRIPTS only if it is deliberately unwritten"
                continue
            if script_rel not in flag_cache:
                flag_cache[script_rel] = _script_flag_names(script)
            known = flag_cache[script_rel]
            for flag, value in args.items():
                checked += 1
                if flag not in known:
                    problems.append(
                        f"{cfg['exp_id']}: {script_rel} does not accept "
                        f"--{flag} (yaml key '{flag}'={value!r})")
    assert not problems, ("yaml keys not accepted by target scripts:\n  "
                          + "\n  ".join(problems))
    print(f"checked {checked} flags across {len(flag_cache)} scripts")


def test_known_pending_scripts_are_actually_referenced():
    """Guards KNOWN_PENDING_SCRIPTS against rot: every entry must still be
    referenced by some config, so stale entries get pruned."""
    defaults = config.load_defaults(CONFIG_DIR / "_defaults.yaml")
    configs = config.discover_configs(CONFIG_DIR, defaults)
    referenced: set[str] = set()
    for cfg in configs:
        calls: list[tuple[str, dict]] = []
        _collect_flag_calls(cfg, calls)
        for script_rel, _ in calls:
            if not _resolve(script_rel).exists():
                referenced.add(script_rel)
    assert referenced == KNOWN_PENDING_SCRIPTS, \
        f"stale KNOWN_PENDING_SCRIPTS entries: {KNOWN_PENDING_SCRIPTS - referenced}"


def test_flags_from_args_translates_registry_dicts():
    assert config.flags_from_args({"gt": "x.json", "image_root": "/img"}) == \
        ["--gt", "x.json", "--image-root", "/img"]
    assert config.flags_from_args({"verbose": True, "reason": False}) == ["--verbose"]
    assert config.flags_from_args({"empty": None}) == []
    assert config.flags_from_args(None) == []


def test_deep_merge_semantics():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    assert config.deep_merge(base, {"a": {"y": 3, "z": 4}}) == \
        {"a": {"x": 1, "y": 3, "z": 4}, "b": 1}
    assert config.deep_merge(base, {"b": {"nested": True}})["b"] == {"nested": True}


def test_load_config_seeds_train_args():
    defaults = config.load_defaults(CONFIG_DIR / "_defaults.yaml")
    single = CONFIG_DIR / "d1_bce_mp_s42.yaml"
    cfg = config.load_config(single, defaults)
    assert cfg["type"] == "vlm"
    assert cfg["train"]["args"]["seed"] == 42
    assert cfg["exp_id"] == "d1_bce_mp_s42"
