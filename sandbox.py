"""Docker-based sandbox for agent execution.

Each agent runs in an isolated container with:
- Its own filesystem (only workspace + data cache mounted)
- Full network access (for web search, API calls)
- No access to host filesystem, other agents, or credentials on disk
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_NAME = "mantis-agent"
MINER_IMAGE_NAME = "mantis-miner"
CONTAINER_PREFIX = "mantis-ag-"
MINER_CONTAINER_PREFIX = "mantis-miner-"

try:
    MAX_RUNNING = int(os.environ.get("MANTIS_MAX_AGENTS", "5"))
except (ValueError, TypeError):
    MAX_RUNNING = 5
    logging.getLogger(__name__).warning("Invalid MANTIS_MAX_AGENTS, defaulting to %d", MAX_RUNNING)

try:
    CONTAINER_MEM_PCT = float(os.environ.get("MANTIS_CONTAINER_MEM_PCT", "15"))
except (ValueError, TypeError):
    CONTAINER_MEM_PCT = 15.0
    logging.getLogger(__name__).warning("Invalid MANTIS_CONTAINER_MEM_PCT, defaulting to %.0f", CONTAINER_MEM_PCT)
CONTAINER_MEM_MIN_GB = 4
CONTAINER_MEM_MAX_GB = 64


def _parse_json_from_stdout(stdout: str) -> dict | None:
    """Extract the first top-level JSON object from container stdout.

    Container output often has log lines before/after the JSON payload.
    Returns the parsed dict, or None if no valid JSON object is found.
    """
    text = stdout.strip()
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return None
    try:
        return json.loads(text[brace_start:brace_end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _container_memory_limit():
    """Compute per-container memory limit as a percentage of system RAM.

    Reads /proc/meminfo for total RAM, applies MANTIS_CONTAINER_MEM_PCT
    (default 15%), and clamps between CONTAINER_MEM_MIN_GB and
    CONTAINER_MEM_MAX_GB.  Falls back to 8g if /proc/meminfo is unavailable.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_gb = kb / (1024 * 1024)
                    limit_gb = total_gb * CONTAINER_MEM_PCT / 100.0
                    limit_gb = max(CONTAINER_MEM_MIN_GB,
                                   min(CONTAINER_MEM_MAX_GB, limit_gb))
                    limit_gb = int(limit_gb)
                    return f"{limit_gb}g"
    except (OSError, ValueError):
        pass
    return "8g"

_container_cache = {"containers": {}, "ts": 0}
_last_cleanup = 0


def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw)
    except subprocess.TimeoutExpired:
        logger.warning("Docker command timed out: %s", " ".join(cmd[:4]))
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="timeout")


def image_exists():
    r = _run(["docker", "images", "-q", IMAGE_NAME])
    return r.returncode == 0 and r.stdout.strip() != ""


def build_image(pkg_dir):
    dockerfile = os.path.join(pkg_dir, "Dockerfile")
    if not os.path.exists(dockerfile):
        raise FileNotFoundError(f"No Dockerfile at {dockerfile}")
    logger.info("Building Docker image %s ...", IMAGE_NAME)
    r = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "-f", dockerfile, pkg_dir],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        logger.error("Docker build failed:\n%s", r.stderr[-2000:])
        raise RuntimeError(f"Docker build failed: {r.stderr[-500:]}")
    logger.info("Docker image %s built", IMAGE_NAME)


def _image_stale(pkg_dir):
    """Check if the Dockerfile is newer than the built image."""
    dockerfile = os.path.join(pkg_dir, "Dockerfile")
    if not os.path.exists(dockerfile):
        return False
    r = _run(["docker", "inspect", "--format", "{{.Created}}", IMAGE_NAME])
    if r.returncode != 0 or not r.stdout.strip():
        return True
    from datetime import datetime as _dt, timezone as _tz
    try:
        raw = r.stdout.strip().split(".")[0]
        img_time = _dt.fromisoformat(raw).replace(tzinfo=_tz.utc)
        file_mtime = _dt.fromtimestamp(os.path.getmtime(dockerfile), tz=_tz.utc)
        return file_mtime > img_time
    except (ValueError, OSError):
        return False


def ensure_image(pkg_dir):
    if not image_exists() or _image_stale(pkg_dir):
        build_image(pkg_dir)


def container_name(agent_id):
    return f"{CONTAINER_PREFIX}{agent_id}"


def _list_containers():
    """Return dict of container_name -> running (bool)."""
    global _last_cleanup
    t = time.monotonic()
    if t - _container_cache["ts"] < 1.0:
        return _container_cache["containers"]
    if t - _last_cleanup > 300:
        _last_cleanup = t
        cleanup_dead_containers()
    r = _run(["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}",
              "--format", "{{.Names}}\t{{.State}}"])
    containers = {}
    if r.returncode == 0 and r.stdout.strip():
        for line in r.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                containers[parts[0]] = parts[1] == "running"
    _container_cache["containers"] = containers
    _container_cache["ts"] = t
    return containers


def running_count():
    return sum(1 for v in _list_containers().values() if v)


def container_running(agent_id):
    return _list_containers().get(container_name(agent_id), False)


def launch_container(agent_id, config, agent_dir, pkg_dir, data_dir,
                     anthropic_key, coinglass_key=None):
    """Launch agent_runner.py inside a Docker container."""
    cname = container_name(agent_id)

    kill_container(agent_id)

    agent_dir_abs = os.path.abspath(agent_dir)
    pkg_abs = os.path.abspath(pkg_dir)

    def _env_safe(v):
        return str(v).replace("\n", "").replace("\r", "").replace("\0", "")

    mem_limit = _container_memory_limit()
    logger.info("Container %s memory limit: %s", cname, mem_limit)
    cmd = [
        "docker", "run", "-d",
        "--name", cname,
        "--restart=on-failure:1",
        f"--memory={mem_limit}",
        "--cpus=2",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "--pids-limit=512",
        "--add-host=metadata.google.internal:127.0.0.1",
        "--add-host=169.254.169.254:127.0.0.1",
        "-v", f"{agent_dir_abs}:/agent",
        "-v", f"{pkg_abs}:/app/mantis_model_iteration_tool:ro",
        "-e", f"ANTHROPIC_API_KEY={_env_safe(anthropic_key)}",
        "-e", "ANTHROPIC_AUTH_TOKEN=",
        "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
    ]
    if coinglass_key:
        cmd.extend(["-e", f"COINGLASS_API_KEY={_env_safe(coinglass_key)}"])

    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        cmd.extend(["-v", f"{os.path.abspath(data_dir)}:/data"])
        cmd.extend(["-e", "MANTIS_DATA_DIR=/data"])

    runner_args = [
        "python3", "/app/mantis_model_iteration_tool/agent_runner.py",
        "--challenge", config["challenge"],
        "--goal", config["goal"],
        "--min-iterations", str(config.get("min_iterations", 5)),
        "--max-iterations", str(config.get("max_iterations", 20)),
        "--model", config.get("model", "sonnet"),
        "--days-back", str(config.get("days_back", 60)),
        "--id", agent_id,
        "--agent-dir", "/agent",
    ]
    if data_dir:
        runner_args.extend(["--data-dir", "/data"])

    shell_cmd = " ".join(
        _sh_escape(a) for a in runner_args
    ) + " 2>&1 | tee /agent/stdout.log"

    cmd.extend([IMAGE_NAME, "bash", "-c", shell_cmd])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        logger.error("Failed to launch container %s: %s", cname, r.stderr)
        raise RuntimeError(f"Docker launch failed: {r.stderr[-300:]}")

    _container_cache["ts"] = 0
    logger.info("Launched container %s", cname)


def kill_container(agent_id):
    cname = container_name(agent_id)
    _run(["docker", "stop", "-t", "5", cname])
    _run(["docker", "rm", "-f", cname])
    _container_cache["ts"] = 0


def cleanup_dead_containers():
    """Remove exited containers with our prefix to avoid accumulation."""
    r = _run(["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}",
              "--filter", "status=exited",
              "--format", "{{.Names}}"])
    if r.returncode == 0 and r.stdout.strip():
        names = [n for n in r.stdout.strip().split("\n") if n]
        if names:
            _run(["docker", "rm"] + names)
            for name in names:
                logger.info("Cleaned up exited container %s", name)
    _container_cache["ts"] = 0


def _sh_escape(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


# ── Miner container management ──────────────────────────────────────────────

MINER_DOCKERFILE_CONTENT = """\
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl git && \\
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \\
    "numpy==2.2.6" \\
    "pandas==2.3.3" \\
    "scipy==1.15.3" \\
    "scikit-learn==1.7.2" \\
    "requests==2.32.5" \\
    "pyarrow==20.0.0" \\
    "boto3>=1.35" \\
    "bittensor>=7.0" \\
    "cryptography>=42"

COPY _timelock_wheels/ /tmp/_timelock_wheels/
COPY _timelock_py/ /tmp/_timelock_py/
RUN pip install --no-cache-dir --break-system-packages /tmp/_timelock_wheels/*.whl \\
    && pip install --no-cache-dir --break-system-packages /tmp/_timelock_py/ \\
    && rm -rf /tmp/_timelock_wheels /tmp/_timelock_py

WORKDIR /app
"""


def _stage_timelock_files(pkg_dir: str):
    """Copy timelock wheel + Python source into the build context."""
    import shutil
    repo_root = str(Path(pkg_dir).resolve().parent)
    tl_src = os.path.join(repo_root, "timelock-src")

    wheels_dst = os.path.join(pkg_dir, "_timelock_wheels")
    py_dst = os.path.join(pkg_dir, "_timelock_py")

    if os.path.isdir(wheels_dst):
        shutil.rmtree(wheels_dst)
    if os.path.isdir(py_dst):
        shutil.rmtree(py_dst)

    whl_dir = os.path.join(tl_src, "target", "wheels")
    if not os.path.isdir(whl_dir):
        raise FileNotFoundError(
            f"Timelock wheels not found at {whl_dir}. "
            "Build them first: cd timelock-src && cargo build --release"
        )
    shutil.copytree(whl_dir, wheels_dst)

    py_src = os.path.join(tl_src, "py")
    if not os.path.isdir(py_src):
        raise FileNotFoundError(f"Timelock Python source not found at {py_src}")
    shutil.copytree(py_src, py_dst)

    return wheels_dst, py_dst


def build_miner_image(pkg_dir):
    """Build the miner Docker image with R2/crypto/bittensor/timelock deps."""
    import shutil
    import tempfile

    staged = []
    try:
        staged = list(_stage_timelock_files(pkg_dir))
    except FileNotFoundError as exc:
        logger.warning("Timelock staging skipped: %s", exc)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Dockerfile", dir=pkg_dir, delete=False,
    ) as f:
        content = MINER_DOCKERFILE_CONTENT
        if not staged:
            content = content.split("COPY _timelock")[0] + "WORKDIR /app\n"
        f.write(content)
        dockerfile = f.name
    try:
        logger.info("Building miner image %s ...", MINER_IMAGE_NAME)
        r = subprocess.run(
            ["docker", "build", "-t", MINER_IMAGE_NAME,
             "-f", dockerfile, pkg_dir],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            logger.error("Miner image build failed:\n%s", r.stderr[-2000:])
            raise RuntimeError(f"Miner image build failed: {r.stderr[-500:]}")
        logger.info("Miner image %s built", MINER_IMAGE_NAME)
    finally:
        try:
            os.unlink(dockerfile)
        except OSError:
            pass
        for d in staged:
            try:
                shutil.rmtree(d)
            except OSError:
                pass


def _miner_image_exists():
    r = _run(["docker", "images", "-q", MINER_IMAGE_NAME])
    return r.returncode == 0 and r.stdout.strip() != ""


def ensure_miner_image(pkg_dir):
    if not _miner_image_exists():
        build_miner_image(pkg_dir)


def miner_container_name(miner_id):
    return f"{MINER_CONTAINER_PREFIX}{miner_id}"


def miner_running(miner_id=None):
    """Check if a miner container is running. If no ID, check for any."""
    prefix = MINER_CONTAINER_PREFIX
    r = _run(["docker", "ps", "--filter", f"name={prefix}",
              "--format", "{{.Names}}\t{{.State}}"])
    if r.returncode != 0 or not r.stdout.strip():
        return False
    for line in r.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == "running":
            if miner_id is None or parts[0] == miner_container_name(miner_id):
                return True
    return False


def launch_miner_container(
    miner_id,
    miner_dir,
    pkg_dir,
    model_specs,
    env_vars,
    agent_dirs=None,
    interval_seconds=60,
    lock_seconds=30,
    lookback=5000,
    network="finney",
):
    """Launch the miner process inside a Docker container.

    Args:
        miner_id: Unique identifier for this miner
        miner_dir: Host directory for miner status/control files
        pkg_dir: Path to mantis_model_iteration_tool package
        model_specs: List of dicts with agent_id, iteration, challenge,
                     host_path (host), container_path (inside container)
        env_vars: Dict of environment variables (secrets, R2 config, etc.)
        agent_dirs: List of (host_path, container_path) tuples to mount
                    agent workspace directories (read-only) so strategy
                    files are accessible
        interval_seconds: How often to submit predictions
        lock_seconds: Timelock duration
        lookback: Minutes of history to fetch
        network: Bittensor network (finney/test)
    """
    cname = miner_container_name(miner_id)
    kill_miner_container(miner_id)

    miner_dir_abs = os.path.abspath(miner_dir)
    pkg_abs = os.path.abspath(pkg_dir)
    os.makedirs(miner_dir_abs, exist_ok=True)

    def _env_safe(v):
        return str(v).replace("\n", "").replace("\r", "").replace("\0", "")

    mem_limit = _container_memory_limit()
    cmd = [
        "docker", "run", "-d",
        "--name", cname,
        "--restart=on-failure:3",
        f"--memory={mem_limit}",
        "--cpus=1",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "--pids-limit=256",
        "--add-host=metadata.google.internal:127.0.0.1",
        "--add-host=169.254.169.254:127.0.0.1",
        "-v", f"{miner_dir_abs}:/miner",
        "-v", f"{pkg_abs}:/app/mantis_model_iteration_tool:ro",
    ]

    if agent_dirs:
        for host_path, container_path in agent_dirs:
            cmd.extend(["-v", f"{os.path.abspath(host_path)}:{container_path}:ro"])

    for key, val in env_vars.items():
        if val:
            cmd.extend(["-e", f"{key}={_env_safe(val)}"])

    cli_model_specs = []
    for spec in model_specs:
        if isinstance(spec, dict):
            cli_model_specs.append(
                f"{spec['agent_id']}:{spec['iteration']}:"
                f"{spec['challenge']}:{spec['container_path']}"
            )
        else:
            cli_model_specs.append(str(spec))

    runner_args = [
        "python3", "-m", "mantis_model_iteration_tool.miner",
        "--miner-dir", "/miner",
        "--interval", str(interval_seconds),
        "--lock-seconds", str(lock_seconds),
        "--lookback", str(lookback),
        "--network", network,
    ]
    for s in cli_model_specs:
        runner_args.extend(["--model", s])

    shell_cmd = " ".join(
        _sh_escape(a) for a in runner_args
    ) + " 2>&1 | tee /miner/stdout.log"

    cmd.extend([MINER_IMAGE_NAME, "bash", "-c", shell_cmd])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        logger.error("Failed to launch miner container %s: %s", cname, r.stderr)
        raise RuntimeError(f"Miner Docker launch failed: {r.stderr[-300:]}")

    logger.info("Launched miner container %s", cname)
    return cname


def run_miner_registration(
    pkg_dir,
    coldkey_mnemonic,
    hotkey_mnemonic,
    network="finney",
    miner_dir=None,
):
    """Run registration inside a short-lived Docker container.

    Returns the parsed registration result dict.  The container is
    removed after the command completes.
    """
    pkg_abs = os.path.abspath(pkg_dir)

    def _env_safe(v):
        return str(v).replace("\n", "").replace("\r", "").replace("\0", "")

    cmd = [
        "docker", "run", "--rm",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "-v", f"{pkg_abs}:/app/mantis_model_iteration_tool:ro",
        "-e", f"COLDKEY_MNEMONIC={_env_safe(coldkey_mnemonic)}",
        "-e", f"HOTKEY_MNEMONIC={_env_safe(hotkey_mnemonic)}",
    ]

    if miner_dir:
        miner_dir_abs = os.path.abspath(miner_dir)
        os.makedirs(miner_dir_abs, exist_ok=True)
        cmd.extend(["-v", f"{miner_dir_abs}:/miner"])

    cmd.extend([
        MINER_IMAGE_NAME,
        "python3", "-m", "mantis_model_iteration_tool.miner",
        "register",
        "--network", network,
        "--miner-dir", "/miner" if miner_dir else "/tmp",
    ])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        logger.error("Registration container failed: %s", r.stderr[-500:])

    result = _parse_json_from_stdout(r.stdout)
    return result or {
        "success": False,
        "error": f"Registration failed: {r.stderr[-300:] or r.stdout[-300:]}",
    }


def run_miner_check_registration(
    pkg_dir,
    hotkey_ss58="",
    hotkey_mnemonic="",
    network="finney",
):
    """Check registration status inside a short-lived Docker container."""
    pkg_abs = os.path.abspath(pkg_dir)

    def _env_safe(v):
        return str(v).replace("\n", "").replace("\r", "").replace("\0", "")

    cmd = [
        "docker", "run", "--rm",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "-v", f"{pkg_abs}:/app/mantis_model_iteration_tool:ro",
    ]
    if hotkey_mnemonic:
        cmd.extend(["-e", f"HOTKEY_MNEMONIC={_env_safe(hotkey_mnemonic)}"])

    check_cmd = [
        "python3", "-m", "mantis_model_iteration_tool.miner",
        "check",
        "--network", network,
    ]
    if hotkey_ss58:
        check_cmd.extend(["--hotkey-ss58", hotkey_ss58])

    cmd.extend([MINER_IMAGE_NAME] + check_cmd)

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    result = _parse_json_from_stdout(r.stdout)
    return result or {
        "registered": False,
        "error": f"Check failed: {r.stderr[-300:] or r.stdout[-300:]}",
    }


def run_miner_generate_wallet(pkg_dir):
    """Generate a new coldkey + hotkey pair inside a short-lived Docker container.

    Returns dict with coldkey_mnemonic, coldkey_ss58, hotkey_mnemonic,
    hotkey_ss58. Mnemonics are only in container stdout — never on disk.
    """
    pkg_abs = os.path.abspath(pkg_dir)
    cmd = [
        "docker", "run", "--rm",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "-v", f"{pkg_abs}:/app/mantis_model_iteration_tool:ro",
        MINER_IMAGE_NAME,
        "python3", "-m", "mantis_model_iteration_tool.miner", "generate",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        logger.error("Wallet generation container failed: %s", r.stderr[-500:])

    result = _parse_json_from_stdout(r.stdout)
    return result or {"error": f"Generation failed: {r.stderr[-300:] or r.stdout[-300:]}"}


def run_miner_check_balance(pkg_dir, ss58_address, network="finney"):
    """Check TAO balance of an SS58 address inside a short-lived Docker container."""
    pkg_abs = os.path.abspath(pkg_dir)
    cmd = [
        "docker", "run", "--rm",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "-v", f"{pkg_abs}:/app/mantis_model_iteration_tool:ro",
        MINER_IMAGE_NAME,
        "python3", "-m", "mantis_model_iteration_tool.miner", "balance",
        "--ss58", ss58_address,
        "--network", network,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    result = _parse_json_from_stdout(r.stdout)
    return result or {
        "balance": 0.0,
        "error": f"Balance check failed: {r.stderr[-300:] or r.stdout[-300:]}",
    }


def kill_miner_container(miner_id):
    cname = miner_container_name(miner_id)
    _run(["docker", "stop", "-t", "10", cname])
    _run(["docker", "rm", "-f", cname])


def miner_container_status(miner_id):
    """Get miner container state dict."""
    cname = miner_container_name(miner_id)
    r = _run(["docker", "inspect", "--format",
              "{{.State.Status}}\t{{.State.ExitCode}}", cname])
    if r.returncode != 0:
        return {"exists": False, "running": False, "status": "not_found"}
    parts = r.stdout.strip().split("\t")
    status = parts[0] if parts else "unknown"
    exit_code = int(parts[1]) if len(parts) > 1 else -1
    return {
        "exists": True,
        "running": status == "running",
        "status": status,
        "exit_code": exit_code,
        "container": cname,
    }
