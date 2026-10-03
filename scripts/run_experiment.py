#!/usr/bin/env python
"""Unified experiment runner for the SAGE experiments registry.

Reads one or more experiment configs (configs/experiments/*.yaml, deep-merged
over _defaults.yaml: common <- types[type] <- experiment), then runs each in
dependency order: training -> checkpoint discovery -> eval -> eval_report.json
under results/experiments/<exp_id>/.

Usage
-----
  python scripts/run_experiment.py --list
  python scripts/run_experiment.py --config configs/experiments/d1_bce_nomp_s42.yaml --dry-run
  python scripts/run_experiment.py --config configs/experiments --dry-run
  python scripts/run_experiment.py --config configs/experiments/d1_bce_nomp_s42.yaml
  python scripts/run_experiment.py --config configs/experiments --force --until 20

Idempotence: an experiment whose results/experiments/<exp_id>/eval_report.json
already exists is skipped unless --force is given.

Multi-GPU: pass --gpus "0,1,2,3" (optionally --parallel N) to dispatch
experiments concurrently, one exclusive GPU per experiment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from sage import config
from sage.paths import CONFIG_DIR, RESULTS_ROOT, REPO
from sage_check.constants import MAX_PIXELS
from servers.common.lifecycle import find_free_port, run_with_server

MANAGED_EVAL_KEYS = {"model", "output", "vlm_url", "vlm-url"}


# ---------------------------------------------------------------------------
# Command building (no shell=True anywhere; every command is a list)
# ---------------------------------------------------------------------------
deep_merge = config.deep_merge
load_defaults = config.load_defaults
load_config = config.load_config
discover_configs = config.discover_configs
sort_configs = config.sort_configs
flags_from_args = config.flags_from_args


def build_train_cmds(cfg: dict) -> list[dict]:
    """Return a list of steps: [{desc, cmd, env?}]."""
    train = cfg.get("train", {})
    if cfg["type"] == "eval_only" or not train:
        return []
    # train.output_dir is runner-managed: discover_checkpoint() looks for the
    # weights there, so the training script must be told to write there too
    # (its own --output default, e.g. checkpoints/yolo_cls, would land elsewhere).
    out_dir = train.get("output_dir")
    managed_out = []
    if out_dir:
        flag = "--output-dir" if cfg["type"] == "vlm" else "--output"
        managed_out = [flag, str(REPO / out_dir)]
    steps = []
    if "steps" in train:  # multi-stage, e.g. ext_head
        for st in train["steps"]:
            steps.append({
                "desc": f"train stage: {Path(st['script']).name}",
                "cmd": [sys.executable, str(REPO / st["script"])]
                       + flags_from_args(st.get("args", {})),
            })
        return steps
    steps.append({
        "desc": f"train: {Path(train['script']).name}",
        "cmd": [sys.executable, str(REPO / train["script"])]
               + flags_from_args(train.get("args", {})) + managed_out,
    })
    return steps


def build_eval_cmd(cfg: dict, checkpoint: str | None, vlm_url: str | None,
                   dry: bool = False) -> dict:
    """Return {desc, cmd, needs_server, lora_path}."""
    ev = cfg["eval"]
    args = {k: v for k, v in (ev.get("args", {}) or {}).items()
            if k not in MANAGED_EVAL_KEYS}
    cmd = [sys.executable, str(REPO / ev["script"])] + flags_from_args(args)
    exp_id = cfg["exp_id"]
    eval_out = str(RESULTS_ROOT / exp_id)
    t = cfg["type"]

    if t in ("vlm", "eval_only"):
        if vlm_url:
            cmd += ["--vlm-url", vlm_url]
        if t == "vlm":
            lora = checkpoint
        else:
            # eval_only may reuse a trained adapter: eval.lora_from_experiment
            # names an experiment whose output_dir holds the LoRA adapter.
            ref = ev.get("lora_from_experiment")
            if ref:
                lora = discover_vlm_adapter(REPO / "checkpoints" / "experiments" / ref)
                if lora is None and not dry:
                    print(f"[fail] {exp_id}: no adapter under "
                          f"checkpoints/experiments/{ref} (train it first)")
                    return {"desc": "eval (missing adapter)",
                            "cmd": [sys.executable, "-c",
                                    "import sys; print('missing adapter'); sys.exit(2)"],
                            "needs_server": False, "lora_path": None}
            else:
                lora = None
        return {"desc": f"eval: {Path(ev['script']).name}",
                "cmd": cmd + ["--output", eval_out],
                "needs_server": True, "lora_path": lora}

    # veto_eval: only the collect stage talks to a VLM server — spawn it
    # exactly like the vlm/eval_only evals (same pixel-cap/adapter env
    # propagation) so the manual-server path and its config drift
    # (2026-09-28: e3 collect at a different max_pixels than d1) is
    # structurally impossible.  fit/freeze/eval stages stay offline.
    if t == "veto_eval":
        if (ev.get("args", {}) or {}).get("stage") != "collect":
            return {"desc": f"eval: {Path(ev['script']).name}",
                    "cmd": cmd + ["--output", eval_out],
                    "needs_server": False, "lora_path": None}
        ref = ev.get("lora_from_experiment")
        lora = None
        if ref:
            lora = discover_vlm_adapter(REPO / "checkpoints" / "experiments" / ref)
            if lora is None and not dry:
                print(f"[fail] {exp_id}: no adapter under "
                      f"checkpoints/experiments/{ref} (train it first)")
                return {"desc": "eval (missing adapter)",
                        "cmd": [sys.executable, "-c",
                                "import sys; print('missing adapter'); sys.exit(2)"],
                        "needs_server": False, "lora_path": None}
        return {"desc": f"eval: {Path(ev['script']).name}",
                "cmd": cmd + ["--vlm-url", vlm_url] + ["--output", eval_out],
                "needs_server": True, "lora_path": lora}

    # checkpoint-based evals
    if checkpoint:
        cmd += ["--model", checkpoint]
    return {"desc": f"eval: {Path(ev['script']).name}",
            "cmd": cmd + ["--output", eval_out],
            "needs_server": False, "lora_path": None}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def find_latest(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def discover_vlm_adapter(output_dir: Path) -> Path | None:
    """Newest directory containing an adapter (adapter_config.json), else the
    output_dir itself when it directly holds adapter weights."""
    if not output_dir.exists():
        return None
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir
    hits = find_latest(list(output_dir.glob("**/adapter_config.json")))
    return hits.parent if hits else None


def discover_checkpoint(cfg: dict) -> Path | None:
    train = cfg.get("train", {}) or {}
    out = REPO / train["output_dir"] if train.get("output_dir") else None
    t = cfg["type"]
    if t == "eval_only":
        return None
    if t == "vlm":
        return discover_vlm_adapter(out) if out else None
    if out is None or not out.exists():
        return None
    if t in ("yolo", "yolo_bce"):
        return find_latest(list(out.glob("**/best.pt")))
    if t == "cnn":
        for name in ("best.pt", "best.pth", "final.pt"):
            hits = list(out.glob(f"**/{name}"))
            if hits:
                return find_latest(hits)
        return None
    if t == "ext_head":
        # head weights saved by train_ext_head.py (name unknown) -> newest *.pt
        return find_latest(list(out.glob("**/*.pt")) + list(out.glob("**/*.pth")))
    return None


# ---------------------------------------------------------------------------
# VLM pre-flight checks
# ---------------------------------------------------------------------------
def check_swift_importable(dry_run: bool) -> bool:
    """ms-swift probe for vlm-type training (targets ms-swift==4.4.2).

    Probes the full trainer import chain, not just `import swift`: version
    mismatches (e.g. torch < 2.6 lacking FSDPModule) only surface when
    swift.trainers -> callbacks is imported.
    """
    probe = "import swift; from swift.trainers import Seq2SeqTrainer"
    if dry_run:
        print(f"    [probe] (dry-run) {sys.executable} -c '{probe}'")
        return True
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return True
    print("[err] ms-swift trainer import failed in this environment "
          f"(python {sys.version.split()[0]}).")
    print("      The VLM training scripts target ms-swift==4.4.2. Install with:")
    print("        pip install 'ms-swift[llm]==4.4.2'")
    if "FSDPModule" in (proc.stderr or ""):
        print("      This is a torch version mismatch: ms-swift 4.4.2's callback")
        print("      stack needs torch >= 2.6 (torch.distributed.fsdp.FSDPModule).")
        print("      Align torch with the known-good training host (pip show torch")
        print("      there and replicate), e.g.:")
        print("        pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cu124")
    print("      (or run VLM experiments on the training host; this machine only "
          "supports syntax/CLI/dry-run validation)")
    if proc.stderr.strip():
        print(f"      import error: {proc.stderr.strip().splitlines()[-1]}")
    return False


def smoke_check_loss_components(cfg: dict) -> None:
    """Abort before (re)launching a VLM run whose recorded loss components
    contradict the configured weights — catches silently-wrong hybrid loss.
    """
    train = cfg.get("train", {})
    out = train.get("output_dir")
    if not out:
        return
    path = REPO / out / "loss_components.jsonl"
    if not path.exists():
        return
    try:
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    except Exception as e:
        print(f"[warn] cannot parse {path}: {e} — continuing")
        return
    mp_w = float(cfg["train"].get("args", {}).get("mp_weight", 1.0))
    bce_w = float(cfg["train"].get("args", {}).get("bce_weight", 1.0))
    rec_mp = first.get("mp_weight")
    rec_bce = first.get("loss_bce")
    problems = []
    if rec_mp is not None and abs(float(rec_mp) - mp_w) > 1e-6:
        problems.append(f"recorded mp_weight={rec_mp} != configured {mp_w}")
    if mp_w > 0 and first.get("loss_mp") is None:
        problems.append("mp_weight>0 but loss_mp component is null")
    if bce_w > 0 and (rec_bce is None or float(rec_bce) <= 0):
        problems.append(f"loss_bce={rec_bce} (expected >0 with bce_weight={bce_w})")
    if problems:
        print(f"[abort] {cfg['exp_id']}: existing {path} contradicts config:")
        for p in problems:
            print(f"        - {p}")
        print("        Delete the file (or the run dir) after verifying, then retry.")
        sys.exit(3)


def run_vlm_server_eval(step: dict, env: dict | None = None) -> int:
    """Run one eval step against a freshly spawned VLM detection server.

    env: base environment for the server and eval subprocess (parallel
    runners pass a per-experiment copy carrying CUDA_VISIBLE_DEVICES).
    """
    # Extract port from the eval command's --vlm-url so server and eval match
    port = None
    for i, arg in enumerate(step["cmd"]):
        if arg == "--vlm-url" and i + 1 < len(step["cmd"]):
            try:
                port = int(step["cmd"][i + 1].rsplit(":", 1)[-1])
            except ValueError:
                pass
            break
    if port is None:
        port = find_free_port()
    srv_env = {"SAGE_VLM_PORT": str(port), "SAGE_VLM_DTYPE": "fp16"}
    # SAGE_VLM_MODEL / SAGE_VLM_MAX_PIXELS are injected into env by
    # run_experiment from the experiment config (train/eval resolution must
    # match; see the needs_server block there).
    for k in ("SAGE_VLM_MODEL", "SAGE_VLM_MAX_PIXELS"):
        v = (env or {}).get(k)
        if v:
            srv_env[k] = v
    lora = step.get("lora_path")
    if lora:
        srv_env["SAGE_VLM_LORA_PATH"] = lora
    print(f"    [server] (lora={lora}) " if lora else "    [server] ", end="")
    return run_with_server(
        step["cmd"],
        server_cmd=[sys.executable, "-m", "servers.vlm_detect_server"],
        env_vars=srv_env, cwd=REPO, desc=step["desc"], env=env)


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
def run_experiment(cfg: dict, dry_run: bool, force: bool,
                   env: dict | None = None) -> str:
    """Returns one of: 'done' | 'skipped' | 'dry' | 'failed'.

    env: base environment for all subprocesses; parallel runners pass a
    per-experiment copy carrying CUDA_VISIBLE_DEVICES (GPU exclusivity).
    Training is skipped when a checkpoint already exists under
    train.output_dir (re-run a failed eval without retraining).
    """
    exp_id = cfg["exp_id"]
    result_dir = RESULTS_ROOT / exp_id
    report = result_dir / "eval_report.json"
    if report.exists() and not force:
        print(f"[skip] {exp_id}: {report} already exists (use --force to rerun)")
        return "skipped"

    print(f"\n=== {exp_id} (type={cfg['type']}, seed={cfg.get('seed')}) ===")
    print(f"    notes: {cfg.get('notes', '')}")
    result_dir.mkdir(parents=True, exist_ok=True)

    # --- plan -------------------------------------------------------------
    train_cmds = build_train_cmds(cfg)
    checkpoint = discover_checkpoint(cfg)
    vlm_url = f"http://127.0.0.1:{find_free_port()}" \
        if cfg["type"] in ("vlm", "eval_only", "veto_eval") else None
    eval_step = build_eval_cmd(cfg, str(checkpoint) if checkpoint else None,
                               vlm_url, dry=dry_run)

    if cfg["type"] == "vlm":
        smoke_check_loss_components(cfg)

    if dry_run:
        print("    [dry-run] command sequence:")
        if checkpoint:
            print("      (training will be skipped: checkpoint exists)")
        if cfg["type"] == "vlm":
            print(f"      probe: {sys.executable} -c 'import swift' "
                  f"(ms-swift==4.4.2 required)")
        for st in train_cmds:
            print(f"      {st['desc']}: {' '.join(st['cmd'])}")
        if eval_step["needs_server"]:
            if cfg["type"] in ("eval_only", "veto_eval"):
                ref = cfg.get("eval", {}).get("lora_from_experiment")
                lora = ref if ref else "(none: base model, zero-shot)"
            else:
                lora = eval_step["lora_path"] or ("<discovered under "
                        f"{cfg.get('train', {}).get('output_dir', '?')} "
                        "after training>")
            print(f"      server: python -m servers.vlm_detect_server "
                  f"env SAGE_VLM_PORT=<free> SAGE_VLM_LORA_PATH={lora}")
        print(f"      {eval_step['desc']}: {' '.join(eval_step['cmd'])}")
        if checkpoint:
            print(f"      checkpoint: {checkpoint}")
        print(f"      output: {report}")
        return "dry"

    # --- execute ----------------------------------------------------------
    if cfg["type"] == "vlm" and not check_swift_importable(dry_run=False):
        return "failed"

    # T4 16GB protocol: expandable segments absorb the fragmentation that
    # OOMs the visual tower at ~1MP inputs even with fp16 + grad checkpointing.
    # Applies to the training subprocess AND the spawned eval server (same env
    # object flows into run_vlm_server_eval).  setdefault: never clobber an
    # explicit user setting.
    if cfg["type"] in ("vlm", "eval_only"):
        env = dict(os.environ) if env is None else env
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if checkpoint:
        print(f"    [skip-train] checkpoint exists: {checkpoint}")
    else:
        for st in train_cmds:
            print(f"    $ {st['desc']}: {' '.join(st['cmd'])}")
            t0 = time.time()
            proc = subprocess.run(st["cmd"], cwd=REPO, env=env)
            print(f"    [{st['desc']}] exit={proc.returncode} ({time.time()-t0:.0f}s)")
            if proc.returncode != 0:
                print(f"[fail] {exp_id}: training step failed "
                      f"({st['desc']}), stopping experiment")
                return "failed"

    if cfg["type"] not in ("eval_only", "veto_eval"):
        checkpoint = discover_checkpoint(cfg)
        if checkpoint is None:
            print(f"[fail] {exp_id}: no checkpoint found after training")
            return "failed"
        print(f"    checkpoint: {checkpoint}")
        eval_step = build_eval_cmd(cfg, str(checkpoint), vlm_url, dry=dry_run)

    if eval_step["needs_server"]:
        train_args = cfg.get("train", {}).get("args", {})
        env = dict(env) if env is not None else dict(os.environ)
        model_path = train_args.get("model")
        if model_path:
            # setdefault: an explicit SAGE_VLM_MODEL from the shell (e.g. a
            # local path on the training host) wins over the registry id.
            env.setdefault("SAGE_VLM_MODEL", model_path)
        # Eval resolution MUST match training: the spawned server's pixel cap
        # follows the experiment's train max_pixels (registry single source),
        # defaulting to MAX_PIXELS.  A hardcoded different cap here would
        # silently evaluate adapters at a resolution they never trained on
        # and break cross-variant comparability (E1/E2/E3/D0).
        env["SAGE_VLM_MAX_PIXELS"] = str(train_args.get("max_pixels")
                                         or MAX_PIXELS)
        rc = run_vlm_server_eval(eval_step, env=env)
    else:
        print(f"    $ {eval_step['desc']}: {' '.join(eval_step['cmd'])}")
        t0 = time.time()
        proc = subprocess.run(eval_step["cmd"], cwd=REPO, env=env)
        print(f"    [{eval_step['desc']}] exit={proc.returncode} ({time.time()-t0:.0f}s)")
        rc = proc.returncode
    if rc != 0:
        print(f"[fail] {exp_id}: eval failed")
        # A failed eval may still have written predictions + a garbage
        # all-error eval_report.json; it must not poison idempotence (a
        # later re-run would [skip] on it forever).
        if report.exists():
            report.unlink()
            print(f"[fail] {exp_id}: removed partial {report.name}")
        return "failed"
    print(f"[done] {exp_id} -> {report}")
    return "done"


# ---------------------------------------------------------------------------
# Listing / budget
# ---------------------------------------------------------------------------
def list_configs(configs: list[dict]) -> None:
    print(f"{'exp_id':38s} {'type':10s} {'seed':>4s} {'est_h':>5s}  "
          f"{'done':4s}  notes")
    for c in sort_configs(configs):
        done = "yes" if (RESULTS_ROOT / c["exp_id"] / "eval_report.json").exists() \
            else "no"
        print(f"{c['exp_id']:38s} {c['type']:10s} {str(c.get('seed', '')):>4s} "
              f"{c.get('est_hours', 1.0):5.1f}  {done:4s}  {c.get('notes', '')[:60]}")
    print(f"\n{len(configs)} experiments; results root: {RESULTS_ROOT}")


# ---------------------------------------------------------------------------
# Batch runners (sequential / multi-GPU parallel)
# ---------------------------------------------------------------------------
def _run_one(cfg: dict, a, env: dict) -> str:
    """Thread worker: one experiment under `env`; never raises."""
    try:
        return run_experiment(cfg, a.dry_run, a.force, env=env)
    except SystemExit as e:
        # smoke-check aborts exit(3); keep batch behavior explicit
        print(f"[abort] {cfg['exp_id']}: pre-flight check failed (exit {e.code})")
        return "failed"


def _remaining_ids(configs: list[dict]) -> list[str]:
    return [c["exp_id"] for c in sort_configs(configs)
            if not (RESULTS_ROOT / c["exp_id"] / "eval_report.json").exists()]


def _print_budget_stop(cfg: dict, est: float, elapsed_h: float, until: float) -> None:
    print(f"\n[budget] stopping: {cfg['exp_id']} needs ~{est:.1f}h, "
          f"elapsed {elapsed_h:.1f}h of {until:.1f}h budget")


def run_sequential(configs: list[dict], gpus: list[str], a,
                   t_start: float) -> dict[str, int]:
    """One experiment at a time; first failure stops the batch.

    When `gpus` is given, each experiment is pinned via CUDA_VISIBLE_DEVICES
    (cycled) — single-GPU selection must work in sequential mode too, not
    just in --parallel dispatch.
    """
    gpu_cycle = itertools.cycle(gpus) if gpus else None
    results: dict[str, int] = {}
    for cfg in sort_configs(configs):
        est = float(cfg.get("est_hours", 1.0))
        if a.until > 0:
            elapsed_h = (time.time() - t_start) / 3600.0
            if elapsed_h + est > a.until:
                _print_budget_stop(cfg, est, elapsed_h, a.until)
                print("[budget] remaining experiments:")
                for r in _remaining_ids(configs):
                    print(f"  - {r}")
                break
        env = None
        gpu = next(gpu_cycle) if gpu_cycle else ""
        if gpu:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"[gpu] {cfg['exp_id']} -> GPU {gpu}")
        outcome = _run_one(cfg, a, env=env)
        results[outcome] = results.get(outcome, 0) + 1
        if outcome == "failed":
            break
    return results


def run_parallel(configs: list[dict], gpus: list[str], max_workers: int,
                 a, t_start: float) -> dict[str, int]:
    """Dispatch experiments concurrently, one exclusive GPU per experiment.

    GPUs are cycled over the sorted experiment order; each worker gets its own
    environment copy with CUDA_VISIBLE_DEVICES pinned, so both its training and
    its spawned eval server stay on that GPU. Failures are recorded, not fatal
    (other workers keep going). Output lines from concurrent experiments
    interleave in the terminal — per-experiment logs are still identifiable by
    their `=== <exp_id>` banner.
    """
    print(f"[parallel] GPUs={gpus} max_concurrent={max_workers}")
    pending = deque(sort_configs(configs))
    gpu_cycle = itertools.cycle(gpus)
    inflight: dict[concurrent.futures.Future, dict] = {}
    results: dict[str, int] = {}
    stopped_by_budget = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        while pending or inflight:
            # Fill free slots; --until is checked before each dispatch.
            while not stopped_by_budget and pending and len(inflight) < max_workers:
                cfg = pending.popleft()
                est = float(cfg.get("est_hours", 1.0))
                elapsed_h = (time.time() - t_start) / 3600.0
                if a.until > 0 and elapsed_h + est > a.until:
                    _print_budget_stop(cfg, est, elapsed_h, a.until)
                    stopped_by_budget = True
                    break
                gpu = next(gpu_cycle)
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = gpu
                inflight[ex.submit(_run_one, cfg, a, env)] = cfg
                print(f"[dispatch] {cfg['exp_id']} -> GPU {gpu}")
            if not inflight:
                break
            done, _ = concurrent.futures.wait(
                inflight, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                cfg = inflight.pop(fut)
                outcome = fut.result()
                results[outcome] = results.get(outcome, 0) + 1

    if stopped_by_budget:
        print("[budget] remaining experiments:")
        for r in _remaining_ids(configs):
            print(f"  - {r}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append",
                    default=None,
                    help="experiment yaml file, directory, or glob pattern "
                         "(repeatable; default: all of configs/experiments/). "
                         "Use globs to target one phase, e.g. "
                         "--config 'configs/experiments/d1_*.yaml'")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command sequence without executing")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if eval_report.json already exists")
    ap.add_argument("--until", type=float, default=0.0,
                    help="wall-clock budget in hours; stop before the next "
                         "experiment whose estimate would exceed it and print "
                         "the remaining list")
    ap.add_argument("--list", action="store_true",
                    help="list experiments and exit")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default=None,
                    help="override training precision for vlm-type runs "
                         "(default: script default fp16 for T4; use bf16 "
                         "for Ampere+/A10 historical protocol)")
    ap.add_argument("--gpus", default="",
                    help='comma-separated CUDA device ids, e.g. "0,1,2,3"; '
                         "enables parallel dispatch with one exclusive GPU "
                         "per experiment (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--parallel", type=int, default=0,
                    help="max concurrent experiments (default: one per GPU "
                         "when --gpus is given); capped at the GPU count")
    a = ap.parse_args()

    defaults = load_defaults(CONFIG_DIR / "_defaults.yaml")
    by_id: dict[str, dict] = {}
    for raw in (a.config or [str(CONFIG_DIR)]):
        matches = sorted(glob.glob(os.path.expanduser(raw)))
        if not matches:
            if Path(raw).exists():
                matches = [raw]
            else:
                print(f"[err] no experiment yaml matched {raw!r}")
                sys.exit(1)
        for m in matches:
            for cfg in discover_configs(Path(m).resolve(), defaults):
                by_id[cfg["exp_id"]] = cfg  # repeatable --config: dedup by exp_id
    configs = list(by_id.values())
    if not configs:
        print("[err] no experiment yaml matched the given --config pattern(s)")
        sys.exit(1)

    if a.dtype:
        n_overridden = 0
        for cfg in configs:
            if cfg.get("type") == "vlm":
                cfg["train"]["args"]["dtype"] = a.dtype  # type: ignore[index]
                n_overridden += 1
        print(f"[runner] --dtype {a.dtype}: overridden {n_overridden} vlm experiment(s)")

    if a.list:
        list_configs(configs)
        return

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    max_workers = a.parallel or len(gpus)
    if gpus and max_workers > len(gpus):
        print(f"[runner] --parallel {max_workers} capped at GPU count {len(gpus)}")
        max_workers = len(gpus)
    if not gpus and max_workers > 1:
        gpus = [str(i) for i in range(max_workers)]  # assume local ids 0..N-1

    t_start = time.time()
    if max_workers > 1 and not a.dry_run:
        results = run_parallel(configs, gpus, max_workers, a, t_start)
    else:
        if a.dry_run and max_workers > 1:
            print("[runner] --dry-run: ignoring parallel dispatch")
        results = run_sequential(configs, gpus, a, t_start)

    print("\n==== runner summary ====")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"  wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
