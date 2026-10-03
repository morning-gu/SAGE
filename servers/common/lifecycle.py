"""Inference server lifecycle helpers: free ports, health polling, spawn.

Single implementation of the spawn/health-check/teardown dance that used to
live inside scripts/run_experiment.py — usable both by the experiment runner
and by manual debugging sessions.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(url: str, timeout_s: float = 900.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_with_server(step_cmd: list[str], *, server_cmd: list[str],
                    env_vars: dict[str, str], cwd, desc: str,
                    health_url: str | None = None,
                    timeout_s: float = 900.0,
                    env: dict[str, str] | None = None) -> int:
    """Start a server, wait for its health endpoint, run *step_cmd*, stop it.

    server_cmd: e.g. ["python", "-m", "servers.vlm_detect_server"]
    env_vars:   extra env for the server, e.g. {"SAGE_VLM_PORT": ...}
    health_url: defaults to http://127.0.0.1:<port from env_vars>/health
    Returns the step's exit code (1 if the server never became healthy).
    """
    print(f"    [server] -> {' '.join(server_cmd)}")
    full_env = dict(env if env is not None else os.environ)
    full_env.update(env_vars)
    server = subprocess.Popen(server_cmd, cwd=cwd, env=full_env)
    port = env_vars.get("SAGE_VLM_PORT", "")
    url = health_url or f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    try:
        ok = wait_for_health(url, timeout_s=timeout_s)
        if not ok:
            print(f"    [server] health check timed out after {time.time()-t0:.0f}s")
            return 1
        print(f"    [server] ready in {time.time()-t0:.0f}s")
        print(f"    $ {desc}: {' '.join(step_cmd)}")
        t1 = time.time()
        proc = subprocess.run(step_cmd, cwd=cwd)
        print(f"    [{desc}] exit={proc.returncode} ({time.time()-t1:.0f}s)")
        return proc.returncode
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        print("    [server] stopped")
