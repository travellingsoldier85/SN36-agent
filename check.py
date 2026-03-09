#!/usr/bin/env python3
"""check.py

"Ready for submission" checks for Autoppia Subnet 36 miner agent repos.

Validator entrypoint:
  uvicorn main:app --host 0.0.0.0 --port $SANDBOX_AGENT_PORT

So the repo must provide:
- main.py exporting `app`
- GET /health
- POST /act (and optionally /step)

Gateway requirement (Subnet/IWA): every LLM request must:
- go to OPENAI_BASE_URL (sandbox gateway proxy)
- include header: IWA-Task-ID: <task_id>

This script is intentionally conservative: it fails on missing entrypoints/
endpoints and obvious shape problems, and warns on common footguns.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import py_compile
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent

# Keep aligned with: autoppia_web_agents_subnet/opensource/sandbox/requirements.txt
EXPECTED_SANDBOX_PACKAGES = {
    "fastapi",
    "uvicorn",
    "httpx",
    "openai",
    "pydantic",
    "beautifulsoup4",
    "lxml",
    "orjson",
    "tenacity",
    "python-dateutil",
    "rich",
    "jsonschema",    "python-dotenv",
    "loguru",
    "aiohttp",

}


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        # Ignore local eval outputs.
        if "data" in p.parts:
            continue
        files.append(p)
    return files


def _scan_for_secrets() -> None:
    # Conservative: fail if we find an OpenAI-style key prefix in tracked text files.
    # Avoid scanning .env because it may exist for local dev and should be gitignored.
    key_re = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{10,}")
    text_suffixes = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json"}

    for p in _iter_repo_files():
        if p.name == ".env":
            continue
        if p.suffix.lower() not in text_suffixes:
            continue
        txt = _read_text(p)
        if key_re.search(txt):
            _fail(
                f"Possible secret key found in {p.relative_to(REPO_ROOT)}. "
                "Remove it before submission."
            )


def _scan_for_pyc() -> None:
    pyc = [p for p in _iter_repo_files() if p.suffix == ".pyc"]
    if pyc:
        _warn(f"Found .pyc files (should not be committed): {[str(p.relative_to(REPO_ROOT)) for p in pyc][:5]}")


def _check_env_file() -> None:
    env_path = REPO_ROOT / '.env'
    if not env_path.exists():
        return

    # If .env is tracked by git, this is almost certainly a submission footgun.
    try:
        r = subprocess.run(
            ['git', '-C', str(REPO_ROOT), 'ls-files', '--error-unmatch', '.env'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if r.returncode == 0:
            _fail('.env is tracked by git. Remove it (and any secrets) before submission.')
    except Exception:
        # If git isn't available, fall back to a warning.
        pass

    _warn('.env exists in repo folder. Ensure it is gitignored and contains no secrets before submission.')


def _load_module(path: Path, name: str):
    if not path.exists():
        _fail(f"Missing {path.name}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _find_route(app, path: str, method: str) -> bool:
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) == path:
            methods = {m.upper() for m in getattr(route, "methods", [])}
            if method.upper() in methods:
                return True
    return False


def _call_act(app) -> dict[str, Any] | None:
    # Try to call the /act endpoint function with a minimal payload.
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) == "/act":
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                return None

            payload = {
                "task_id": "check",
                "prompt": "open the homepage",
                "url": "http://localhost",
                "snapshot_html": "<html><body><a href='/' aria-label='Home'>Home</a><button>OK</button></body></html>",
                "screenshot": None,
                "step_index": 0,
                "history": [],
                "relevant_data": {},
            }

            if inspect.iscoroutinefunction(endpoint):
                import asyncio

                return asyncio.run(endpoint(payload))  # type: ignore[arg-type]
            return endpoint(payload)  # type: ignore[arg-type]

    return None


def _validate_actions_shape(resp: dict[str, Any]) -> Optional[str]:
    if "actions" not in resp:
        return "Missing top-level 'actions' key"

    actions = resp.get("actions")
    if not isinstance(actions, list):
        return f"'actions' must be a list, got {type(actions).__name__}"

    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            return f"actions[{i}] must be an object, got {type(a).__name__}"

        t = a.get("type")
        if not isinstance(t, str) or not t:
            return f"actions[{i}].type must be a non-empty string"

    return None


def _parse_requirements_pkgs(req_text: str) -> set[str]:
    pkgs: set[str] = set()
    for raw in (req_text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Drop inline comments.
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        # Roughly strip versions/extras/markers.
        # Examples:
        #   uvicorn[standard]>=0.23.0
        #   python-dateutil>=2.9.0.post0; python_version>='3.9'
        base = line.split(";", 1)[0].strip()
        base = base.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("~=", 1)[0].strip()
        name = base.split("[", 1)[0].strip().lower()
        if name:
            pkgs.add(name)
    return pkgs


def main() -> None:
    # Ensure repo root is importable for `from agent import app` etc.
    sys.path.insert(0, str(REPO_ROOT))

    main_py = REPO_ROOT / "main.py"
    agent_py = REPO_ROOT / "agent.py"
    llm_gateway_py = REPO_ROOT / "llm_gateway.py"
    requirements_txt = REPO_ROOT / "requirements.txt"

    if not main_py.exists():
        _fail("Missing main.py (required entrypoint for subnet: uvicorn main:app)")
    _ok("Found main.py")

    if agent_py.exists():
        _ok("Found agent.py")
    else:
        _warn("agent.py not found (recommended). main.py must still expose app.")

    if llm_gateway_py.exists():
        _ok("Found llm_gateway.py")
    else:
        _warn("llm_gateway.py not found; ensure your agent injects IWA-Task-ID header in LLM calls")

    if requirements_txt.exists():
        _ok("Found requirements.txt")
        req_text = _read_text(requirements_txt)
        pkgs = _parse_requirements_pkgs(req_text)

        missing = sorted(p for p in EXPECTED_SANDBOX_PACKAGES if p not in pkgs)
        if missing:
            _fail(
                "requirements.txt is missing sandbox packages you said you ship in the subnet image: "
                f"{missing}. Align it with autoppia_web_agents_subnet/opensource/sandbox/requirements.txt"
            )

        extra = sorted(p for p in pkgs if p not in EXPECTED_SANDBOX_PACKAGES)
        if extra:
            _warn(f"requirements.txt contains extra packages not in the curated sandbox list: {extra}")
    else:
        _warn("requirements.txt not found; validator sandbox image only includes fastapi/uvicorn/httpx")

    # Common footguns for public repos.
    _scan_for_secrets()
    _scan_for_pyc()

    _check_env_file()

    if (REPO_ROOT / "api.py").exists():
        _warn("api.py exists. Subnet entrypoint is main:app; consider removing to avoid confusion.")

    # Best-effort gateway checks.
    if llm_gateway_py.exists():
        gw_text = _read_text(llm_gateway_py)
        if "IWA-Task-ID" not in gw_text:
            _warn("llm_gateway.py does not contain 'IWA-Task-ID'; gateway header injection may be missing")
        if "OPENAI_BASE_URL" not in gw_text:
            _warn("llm_gateway.py does not reference OPENAI_BASE_URL; agent may bypass the sandbox gateway")

    # Compile key python files to catch syntax errors.
    for p in (main_py, agent_py, llm_gateway_py):
        if p.exists():
            try:
                py_compile.compile(str(p), doraise=True)
                _ok(f"Python compile OK: {p.name}")
            except Exception as e:
                _fail(f"Python compile failed for {p.name}: {e}")

    # Load main:app and validate routes.
    main_mod = _load_module(main_py, "main")
    app = getattr(main_mod, "app", None)
    if app is None:
        _fail("main.py does not expose `app`")
    _ok("main.py exposes `app`")

    try:
        from fastapi import FastAPI  # type: ignore

        if not isinstance(app, FastAPI):
            _warn(f"`app` is not a FastAPI instance (got {type(app).__name__})")
    except Exception:
        _warn("Unable to import FastAPI to validate app type")

    if not _find_route(app, "/health", "GET"):
        _fail("GET /health route not found")
    _ok("GET /health route found")

    if not _find_route(app, "/act", "POST"):
        _fail("POST /act route not found")
    _ok("POST /act route found")

    if _find_route(app, "/step", "POST"):
        _ok("POST /step route found")
    else:
        _warn("POST /step route not found (optional)")

    # Basic response shape check
    resp = _call_act(app)
    if resp is None:
        _fail("Unable to invoke /act for shape check")

    if not isinstance(resp, dict):
        _fail(f"/act response must be an object, got {type(resp).__name__}")

    err = _validate_actions_shape(resp)
    if err:
        _fail(f"/act response shape invalid: {err}. Response: {json.dumps(resp)[:200]}")

    _ok("/act response shape looks subnet-compatible")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
