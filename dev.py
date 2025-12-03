"""One-command developer entrypoint.

This script installs backend/frontend dependencies (idempotently) and launches
both servers with live reload. Press Ctrl+C to stop both processes.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).parent
BACKEND_PATH = ROOT / "backend" / "App.py"


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True)


def find_npm() -> list[str]:
    """Return the appropriate npm command for the current platform.

    On Windows, npm is typically exposed as "npm.cmd"; on Unix it is "npm". We
    resolve the executable explicitly to avoid FileNotFoundError when the shell
    does not auto-resolve the shim.
    """

    npm_candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
    for candidate in npm_candidates:
        path = shutil.which(candidate)
        if path:
            return [path]
    raise FileNotFoundError(
        "npm is not installed or not on PATH. Please install Node.js/npm first."
    )


def ensure_backend():
    req = ROOT / "requirements.txt"
    if req.exists():
        print("[dev] Installing backend requirements...")
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def ensure_frontend():
    node_modules = ROOT / "node_modules"
    if not node_modules.exists():
        print("[dev] Installing frontend dependencies...")
        npm_cmd = find_npm()
        run([*npm_cmd, "install"], cwd=ROOT)


def start_processes(host: str, frontend_port: int):
    backend_cmd = [sys.executable, str(BACKEND_PATH)]
    npm_cmd = find_npm()
    frontend_cmd = [*npm_cmd, "run", "dev", "--", "--host", host, "--port", str(frontend_port)]

    print(f"[dev] Starting backend: {' '.join(backend_cmd)}")
    backend = subprocess.Popen(backend_cmd, cwd=ROOT)

    print(f"[dev] Starting frontend: {' '.join(frontend_cmd)}")
    frontend = subprocess.Popen(frontend_cmd, cwd=ROOT)

    return backend, frontend


def main():
    parser = argparse.ArgumentParser(description="Run backend and frontend for development")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for frontend")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()

    if not args.skip_install:
        ensure_backend()
        ensure_frontend()

    backend, frontend = start_processes(args.host, args.frontend_port)

    def shutdown(*_):
        print("\n[dev] Shutting down...")
        for proc in (backend, frontend):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in (backend, frontend):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait indefinitely until interrupted
    backend.wait()
    frontend.wait()


if __name__ == "__main__":
    main()
