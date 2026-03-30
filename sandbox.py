"""Docker-based sandbox for agent execution.

Each agent runs in an isolated container with:
- Its own filesystem (only workspace + data cache mounted)
- Full network access (for web search, API calls)
- No access to host filesystem, other agents, or credentials on disk
"""

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

IMAGE_NAME = "mantis-agent"
CONTAINER_PREFIX = "mantis-ag-"

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
        "-v", f"{pkg_abs}:/app/model_iteration_tool:ro",
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
        "python3", "/app/model_iteration_tool/agent_runner.py",
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
