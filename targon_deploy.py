"""Manage deployment of the MANTIS agent server to Targon or local Docker.

Handles the full lifecycle: deploy, check status, get URL, tear down.
Supports two backends:
  - "targon": Deploy the full agent server to Targon. Claude Code, evaluation,
              and all agent management run remotely. The SDK ships the code via
              add_local_dir — no published Docker image required.
  - "local":  Run agents as local Docker containers (existing behavior).
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PKG_DIR = Path(__file__).parent
TARGON_APP_NAME = "mantis-server"
DEFAULT_PORT = 8080

_deploy_lock = threading.Lock()
_deploy_state_file = PKG_DIR / ".deploy_state.json"

IGNORE_PATTERNS = [
    "agents", ".data", "__pycache__", "*.pyc",
    ".targon_config", ".anthropic_key", ".coinglass_key",
    "_targon_deploy_*", "*.tmp", "*.lock",
]


def _run(cmd, timeout=60, env=None, **kw):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, **kw)
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", " ".join(cmd[:4]))
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="timeout")
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="not found")


# ── Targon CLI helpers ───────────────────────────────────────────────────────

def _targon_bin() -> str:
    """Find the targon CLI binary, preferring the current venv."""
    venv_bin = Path(sys.executable).parent / "targon"
    if venv_bin.exists():
        return str(venv_bin)
    return "targon"


def targon_sdk_importable() -> bool:
    try:
        import targon  # noqa: F401
        return True
    except ImportError:
        return False


def targon_cli_available() -> bool:
    r = _run([_targon_bin(), "--version"])
    return r.returncode == 0


def _targon_env(api_key: str):
    env = os.environ.copy()
    if api_key:
        env["TARGON_API_KEY"] = api_key
    return env


def _cli_output(r) -> str:
    """Targon CLI writes formatted output to stderr; combine both streams."""
    return ((r.stdout or "") + "\n" + (r.stderr or "")).strip()


def _targon_app_list(api_key: str) -> list[dict]:
    env = _targon_env(api_key)
    r = _run([_targon_bin(), "app", "ls"], env=env, timeout=30)
    if r.returncode != 0:
        return []
    return _parse_app_list(_cli_output(r))


def _parse_app_list(output: str) -> list[dict]:
    apps = []
    current_app = None
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        app_match = re.search(r'(app-[a-zA-Z0-9]+)', line)
        if app_match and TARGON_APP_NAME in line:
            current_app = {"id": app_match.group(1), "name": TARGON_APP_NAME, "raw": line}
            apps.append(current_app)
        elif current_app and "wrk-" in line:
            wrk_match = re.search(r'(wrk-[a-zA-Z0-9]+)', line)
            if wrk_match:
                current_app["workload_id"] = wrk_match.group(1)
    return apps


def _targon_app_get(api_key: str, app_id: str) -> str:
    env = _targon_env(api_key)
    r = _run([_targon_bin(), "app", "get", app_id], env=env, timeout=30)
    return _cli_output(r) if r.returncode == 0 else ""


def _extract_url_from_output(output: str) -> str:
    url_match = re.search(r'https?://[^\s"\']+\.targon\.[^\s"\']+', output)
    if url_match:
        return url_match.group(0).rstrip("/")
    url_match = re.search(r'https?://[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z]+', output)
    if url_match:
        return url_match.group(0).rstrip("/")
    return ""


# ── Deploy file generation ───────────────────────────────────────────────────

def _generate_deploy_file(server_auth_key: str = "",
                          anthropic_key: str = "",
                          coinglass_key: str = "") -> Path:
    """Generate a Targon deployment script.

    The generated file is placed at ``_targon_deploy_app.py`` and
    ``add_local_file`` copies it to ``/app/`` inside the container so
    the Targon runtime can ``import _targon_deploy_app`` and find the
    ``serve`` function.  Credentials are baked into ``Image.env()``.
    """
    env_vars = {
        "PYTHONPATH": "/app",
        "MANTIS_AGENTS_DIR": "/data/agents",
        "MANTIS_DATA_DIR": "/data/cache",
    }
    if server_auth_key:
        env_vars["MANTIS_SERVER_AUTH_KEY"] = server_auth_key
    if anthropic_key:
        env_vars["ANTHROPIC_API_KEY"] = anthropic_key
    if coinglass_key:
        env_vars["COINGLASS_API_KEY"] = coinglass_key

    pkg_path = str(PKG_DIR).replace("\\", "\\\\")
    deploy_file = PKG_DIR / "_targon_deploy_app.py"
    deploy_path = str(deploy_file).replace("\\", "\\\\")

    content = (
        '"""Auto-generated MANTIS Targon deployment. Do not edit."""\n'
        "import targon\n\n"
        "image = (\n"
        "    targon.Image.debian_slim()\n"
        '    .apt_install("curl", "git", "nodejs", "npm")\n'
        '    .run_commands("npm install -g pm2 @anthropic-ai/claude-code")\n'
        "    .pip_install(\n"
        '        "numpy", "pandas", "scipy", "scikit-learn",\n'
        '        "requests", "pyarrow", "fastapi", "uvicorn",\n'
        '        "boto3", "bittensor", "cryptography",\n'
        "    )\n"
        f'    .run_commands("echo MANTIS_BUILD_{int(time.time())}")\n'
        f"    .add_local_dir({json.dumps(pkg_path)}, "
        f'"/app/mantis_model_iteration_tool", ignore={json.dumps(IGNORE_PATTERNS)})\n'
        f"    .add_local_file({json.dumps(deploy_path)}, "
        f'"/app/_targon_deploy_app.py")\n'
        f"    .env({json.dumps(env_vars)})\n"
        '    .workdir("/app")\n'
        ")\n\n"
        f'app = targon.App(name="{TARGON_APP_NAME}", image=image)\n\n\n'
        "@app.function(\n"
        '    resource=targon.Compute.CPU_LARGE,\n'
        "    min_replicas=1,\n"
        "    max_replicas=2,\n"
        "    timeout=3600,\n"
        ")\n"
        "@targon.web_server(port=8080, startup_timeout=600)\n"
        "def serve():\n"
        "    import subprocess as _sp\n"
        '    _sp.Popen(\n'
        '        ["python", "/app/mantis_model_iteration_tool/targon_server.py"],\n'
        '        env={**__import__("os").environ, "PORT": "8080"},\n'
        "    )\n"
    )

    deploy_file.write_text(content)
    return deploy_file


# ── Targon deployment ────────────────────────────────────────────────────────

def deploy_targon(api_key: str, server_auth_key: str = "",
                  anthropic_key: str = "", coinglass_key: str = ""):
    """Deploy the full agent server to Targon via ``targon deploy`` CLI.

    Generates a deployment script with the App, Image, and decorated
    functions, then invokes ``targon deploy <script>`` to ship everything
    to Targon's infrastructure.
    """
    if not targon_sdk_importable():
        raise RuntimeError(
            "Targon SDK not installed. Install it with: pip install targon-sdk")
    if not targon_cli_available():
        raise RuntimeError(
            "Targon CLI not found. Ensure targon-sdk is installed "
            "in the active Python environment.")
    if not api_key:
        raise ValueError("Targon API key is required")
    if not server_auth_key:
        raise ValueError("MANTIS server auth key is required")

    deploy_file = _generate_deploy_file(
        server_auth_key=server_auth_key,
        anthropic_key=anthropic_key,
        coinglass_key=coinglass_key,
    )
    env = _targon_env(api_key)

    logger.info("Deploying full agent server via targon deploy CLI...")

    _patch_script = deploy_file.parent / "_targon_patch_deploy.py"
    _patch_script.write_text(
        "import targon.core.console as _tc\n"
        "_orig = _tc.Console.final\n"
        "def _p(self, msg, details=None):\n"
        "    if details: details = [d for d in details if d is not None]\n"
        "    return _orig(self, msg, details)\n"
        "_tc.Console.final = _p\n"
        "from targon.cli.main import cli; cli()\n"
    )

    try:
        r = _run(
            [sys.executable, str(_patch_script), "deploy", str(deploy_file),
             "--name", TARGON_APP_NAME],
            env=env, timeout=900,
        )
    finally:
        deploy_file.unlink(missing_ok=True)
        _patch_script.unlink(missing_ok=True)

    combined = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    logger.info("targon deploy exit=%d stdout:\n%s", r.returncode,
                combined[-2000:])

    _SUCCESS_MARKERS = ["Published", "Created", "Build complete"]
    looks_successful = any(m in combined for m in _SUCCESS_MARKERS)

    if r.returncode != 0 and not looks_successful:
        tail = combined[-500:] if combined else "unknown error"
        raise RuntimeError(f"Targon deploy failed: {tail}")

    if r.returncode != 0 and looks_successful:
        logger.warning("targon deploy exited %d but output contains success "
                        "markers — treating as success", r.returncode)

    app_id = ""
    app_id_match = re.search(r'(app-[a-zA-Z0-9]+)', combined)
    if app_id_match:
        app_id = app_id_match.group(1)

    wrk_match = re.search(r'(wrk-[a-zA-Z0-9]+)', combined)
    wrk_id = wrk_match.group(1) if wrk_match else ""

    url = _extract_url_from_output(combined)

    if not app_id or not url:
        apps = _targon_app_list(api_key)
        for a in apps:
            aid = a.get("id", "")
            if aid:
                app_id = app_id or aid
                detail = _targon_app_get(api_key, aid)
                url = url or _extract_url_from_output(detail)
                if url:
                    break

    logger.info("Deployed to Targon: app_id=%s wrk=%s url=%s",
                app_id or "?", wrk_id or "?", url or "(not yet available)")
    return {"url": url, "app_id": app_id}


def teardown_targon(api_key: str):
    if not targon_cli_available():
        raise RuntimeError("Targon CLI not installed")
    apps = _targon_app_list(api_key)
    env = _targon_env(api_key)
    removed = 0
    for a in apps:
        app_id = a.get("id", "")
        if app_id:
            r = _run([_targon_bin(), "app", "rm", app_id, "-y"], env=env, timeout=30)
            if r.returncode == 0:
                removed += 1
    return removed


def status_targon(api_key: str) -> dict:
    if not targon_cli_available():
        return {"status": "cli_not_installed"}
    apps = _targon_app_list(api_key)
    if not apps:
        return {"status": "not_deployed"}
    app = apps[0]
    detail = _targon_app_get(api_key, app["id"]) if app.get("id") else ""
    url = _extract_url_from_output(detail)
    return {
        "status": "deployed",
        "app_id": app.get("id", ""),
        "url": url,
        "raw": detail[:2000] if detail else "",
    }


# ── Unified deployment interface ─────────────────────────────────────────────
# State is persisted to disk so it survives across gunicorn worker processes.

def get_deploy_state() -> dict:
    try:
        if _deploy_state_file.exists():
            raw = _deploy_state_file.read_text().strip()
            if raw:
                return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _update_state(**kw):
    with _deploy_lock:
        state = get_deploy_state()
        state.update(kw)
        state["updated_at"] = time.time()
        try:
            _deploy_state_file.write_text(json.dumps(state))
        except OSError:
            logger.warning("Failed to write deploy state file")


def _clear_state():
    with _deploy_lock:
        try:
            _deploy_state_file.unlink(missing_ok=True)
        except OSError:
            pass


def deploy_async(mode: str, targon_api_key: str = "",
                 server_auth_key: str = "", anthropic_key: str = "",
                 coinglass_key: str = ""):
    """Start deployment in a background thread. Returns immediately."""
    current = get_deploy_state()
    if current.get("deploying"):
        return False

    _clear_state()
    _update_state(deploying=True, mode=mode, status="starting", error=None, url=None)

    def _do_deploy():
        try:
            if mode == "targon":
                _update_state(status="packaging_and_deploying")
                result = deploy_targon(
                    api_key=targon_api_key,
                    server_auth_key=server_auth_key,
                    anthropic_key=anthropic_key,
                    coinglass_key=coinglass_key,
                )
                _update_state(
                    status="deployed",
                    url=result.get("url") or "",
                    app_id=result.get("app_id") or "",
                    deploying=False,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception as exc:
            logger.exception("Deployment failed")
            _update_state(status="failed", error=str(exc)[:1000], deploying=False)

    threading.Thread(target=_do_deploy, daemon=True).start()
    return True
