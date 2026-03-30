"""MANTIS Agentic Mining -- web GUI for autonomous strategy agents."""

import argparse
import hashlib
import hmac
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import math
from flask import Flask, request, jsonify, render_template, abort


def _safe_json_body():
    """Parse JSON request body, returning empty dict on malformed input."""
    try:
        return request.get_json(force=True) or {}
    except Exception:
        return {}


def _safe_int(val, default):
    """Convert to int, returning default on any failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _mask_key(key, prefix_len=6, suffix_len=4):
    """Mask an API key for display. Never returns the full key."""
    if not key:
        return ""
    if len(key) <= prefix_len + suffix_len + 3:
        return "***"
    return key[:prefix_len] + "..." + key[-suffix_len:]


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for valid JSON output."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _safe_json_load(path, default=None):
    """Load JSON from a file path, returning default on any failure."""
    if not path.exists():
        return default
    raw = path.read_text().strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return default


def _atomic_json_write(path, data):
    """Write JSON atomically via tempfile + rename."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2, default=str))
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _make_default(default):
    return list(default) if isinstance(default, list) else dict(default or {})


def _locked_json_update(path, mutate_fn, default=None):
    """Read-modify-write a JSON file under an exclusive flock on a .lock file."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / (path.name + ".lock")
    lock_fd = -1
    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        raw = ""
        if path.exists():
            raw = path.read_text(errors="replace").strip()
        try:
            state = json.loads(raw) if raw else _make_default(default)
        except (json.JSONDecodeError, ValueError):
            state = _make_default(default)
        if not isinstance(state, (dict, list)):
            state = _make_default(default)
        mutate_fn(state)
        _atomic_json_write(path, state)
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


AGENTS_DIR = Path(__file__).parent / "agents"
CG_KEY_PATH = Path(__file__).parent / ".coinglass_key"
ANTHROPIC_KEY_PATH = Path(__file__).parent / ".anthropic_key"
WORKING_DIR = Path(__file__).parent.parent

if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

logging.basicConfig(
    level=getattr(logging, os.environ.get("MANTIS_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("mantis.gui")

_start_time = _time.monotonic()
_rate_buckets = defaultdict(list)
_rate_last_cleanup = [0.0]
_ask_semaphore = threading.Semaphore(3)

_SAFE_ID = re.compile(r'^[a-f0-9\-]{1,40}$')


@app.after_request
def _response_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' https://cdn.jsdelivr.net; "
        "frame-ancestors 'none';"
    )
    token_param = request.args.get("token", "")
    auth_token = os.environ.get("MANTIS_AUTH_TOKEN", "")
    if token_param and auth_token and hmac.compare_digest(token_param, auth_token):
        response.set_cookie("mantis_token", token_param, httponly=True,
                            samesite="Strict", max_age=86400 * 30,
                            secure=request.is_secure)
    return response


@app.before_request
def _access_check():
    """If MANTIS_AUTH_TOKEN is set, require it via Bearer header, cookie, or
    query param. Otherwise restrict to localhost only."""
    auth_token = os.environ.get("MANTIS_AUTH_TOKEN", "")
    if request.path == "/health":
        return
    if not auth_token:
        if request.remote_addr not in ("127.0.0.1", "::1"):
            abort(403)
        return
    token = ""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("mantis_token", "")
    if not token:
        token = request.args.get("token", "")
    if not token or not hmac.compare_digest(token, auth_token):
        abort(401)


@app.before_request
def _csrf_check():
    """Reject cross-origin state-changing requests."""
    if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
        return
    origin = request.headers.get("Origin", "")
    if not origin:
        if request.content_type and "application/json" not in request.content_type:
            logger.warning("CSRF: rejected non-JSON POST without Origin for %s", request.path)
            abort(403)
        return
    host = request.host
    allowed = {f"http://{host}", f"https://{host}"}
    if origin not in allowed:
        logger.warning("CSRF: rejected Origin=%s for %s", origin, request.path)
        abort(403)


_RATE_LIMIT_RPM = 120
try:
    _RATE_LIMIT_RPM = int(os.environ.get("MANTIS_RATE_LIMIT_RPM", "120"))
except (ValueError, TypeError):
    pass


@app.before_request
def _rate_limit():
    """Sliding-window rate limiter. Set MANTIS_RATE_LIMIT_RPM=0 to disable."""
    rpm = _RATE_LIMIT_RPM
    if rpm <= 0:
        return
    ip = request.remote_addr
    now = _time.monotonic()
    if now - _rate_last_cleanup[0] > 300:
        stale = [k for k, v in _rate_buckets.items() if not v or now - v[-1] > 120]
        for k in stale:
            del _rate_buckets[k]
        _rate_last_cleanup[0] = now
    bucket = _rate_buckets[ip]
    _rate_buckets[ip] = [t for t in bucket if now - t < 60]
    if len(_rate_buckets[ip]) >= rpm:
        abort(429)
    _rate_buckets[ip].append(now)


def _get_coinglass_key():
    if CG_KEY_PATH.exists():
        return CG_KEY_PATH.read_text().strip() or None
    return None


def _set_coinglass_key(key):
    CG_KEY_PATH.write_text(key.strip())
    CG_KEY_PATH.chmod(0o600)


def _delete_coinglass_key():
    if CG_KEY_PATH.exists():
        CG_KEY_PATH.unlink()


def _get_anthropic_key():
    if ANTHROPIC_KEY_PATH.exists():
        return ANTHROPIC_KEY_PATH.read_text().strip() or None
    return None


def _set_anthropic_key(key):
    ANTHROPIC_KEY_PATH.write_text(key.strip())
    ANTHROPIC_KEY_PATH.chmod(0o600)


def _delete_anthropic_key():
    if ANTHROPIC_KEY_PATH.exists():
        ANTHROPIC_KEY_PATH.unlink()




def _validate_agent_id(agent_id):
    if not _SAFE_ID.match(agent_id):
        abort(400)
    return agent_id


# ---- container helpers ----

from model_iteration_tool.sandbox import (
    ensure_image, launch_container, kill_container,
    container_running, running_count, MAX_RUNNING,
)
DATA_DIR = Path(__file__).parent / ".data"


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "agents_running": running_count(),
        "max_agents": MAX_RUNNING,
        "uptime_s": round(_time.monotonic() - _start_time, 1),
    })


def _agent_is_alive(agent_id):
    return container_running(agent_id)


def _kill_agent_process(agent_id):
    kill_container(agent_id)


def _launch_agent(agent_id, config):
    """Launch agent_runner.py inside a Docker container."""
    agent_dir = AGENTS_DIR / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    import datetime as _dt
    initial_state = {
        "id": agent_id,
        "status": "starting",
        "config": config,
        "iteration": 0,
        "iterations": [],
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    log_path = agent_dir / "log.json"
    log_path.write_text(json.dumps(initial_state, indent=2))

    ant_key = _get_anthropic_key()
    cg_key = _get_coinglass_key()

    ensure_image(str(Path(__file__).parent))

    try:
        launch_container(
            agent_id=agent_id,
            config=config,
            agent_dir=str(agent_dir),
            pkg_dir=str(Path(__file__).parent),
            data_dir=str(DATA_DIR),
            anthropic_key=ant_key,
            coinglass_key=cg_key,
        )
    except Exception:
        import shutil
        shutil.rmtree(agent_dir, ignore_errors=True)
        raise


_VALID_STATUSES = frozenset({
    "running", "stopped", "crashed", "completed", "paused",
    "starting", "max_iterations_reached", "error",
})


def _sanitize_status(status):
    return status if status in _VALID_STATUSES else "unknown"


# ---- agent state helpers ----

def _list_agents_summary():
    """Lightweight: only id, status, config, created_at, iteration count, best metric."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    agents = []
    for d in sorted(AGENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        log_path = d / "log.json"
        state = _safe_json_load(log_path)
        if not state:
            continue
        agent_id = d.name
        actually_running = _agent_is_alive(agent_id)
        if state.get("status") in ("running", "paused") and not actually_running:
            state["status"] = "crashed"
        state["status"] = _sanitize_status(state.get("status", "unknown"))
        iterations = state.get("iterations", [])
        summary = {
            "id": state.get("id", agent_id),
            "status": state["status"],
            "config": state.get("config", {}),
            "created_at": state.get("created_at", ""),
            "iteration_count": len(iterations),
            "paused": (d / "PAUSE").exists(),
        }
        if iterations:
            last = iterations[-1]
            summary["last_metrics"] = last.get("metrics", {})
            summary["has_error"] = last.get("has_error", False)
        if state.get("status") == "running" and not iterations:
            stdout_p = d / "stdout.log"
            if stdout_p.exists():
                try:
                    sz = stdout_p.stat().st_size
                    with open(stdout_p, "r", errors="replace") as sf:
                        if sz > 4096:
                            sf.seek(sz - 4096)
                            sf.readline()
                        tail = sf.read()
                    for line in reversed(tail.strip().split("\n")):
                        if "[prefetch]" in line or "Fetching" in line or "OHLCV" in line:
                            summary["phase"] = line.strip()[:120]
                            break
                except OSError:
                    pass
        agents.append(summary)
    agents.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return _sanitize_for_json(agents)


def _list_agents():
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    agents = []
    for d in sorted(AGENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        log_path = d / "log.json"
        state = _safe_json_load(log_path)
        if not state:
            continue
        agent_id = d.name
        actually_running = _agent_is_alive(agent_id)
        if state.get("status") in ("running", "paused") and not actually_running:
            state["status"] = "crashed"
        state["status"] = _sanitize_status(state.get("status", "unknown"))
        _attach_live_activity(state, agent_id)
        agents.append(state)
    agents.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return _sanitize_for_json(agents)


def _get_agent(agent_id):
    log_path = AGENTS_DIR / agent_id / "log.json"
    state = _safe_json_load(log_path)
    if not state:
        return None
    if state.get("status") in ("running", "paused") and not _agent_is_alive(agent_id):
        state["status"] = "crashed"
    state["status"] = _sanitize_status(state.get("status", "unknown"))
    _attach_live_activity(state, agent_id)
    return _sanitize_for_json(state)


def _attach_live_activity(state, agent_id):
    agent_dir = AGENTS_DIR / agent_id
    ws = agent_dir / "workspace"
    notes_path = ws / "notes.txt"
    try:
        if notes_path.exists():
            raw = notes_path.read_text(errors="replace").strip()
            if raw:
                state["agent_notes"] = raw[-2000:]
    except OSError:
        pass

    state_md_path = ws / "STATE.md"
    try:
        if state_md_path.exists():
            state["state_md"] = state_md_path.read_text(errors="replace")[-2000:]
    except OSError:
        pass

    chat_path = agent_dir / "chat.json"
    msgs = _safe_json_load(chat_path, [])
    if msgs:
        state["chat"] = msgs[-50:]

    state["paused"] = (agent_dir / "PAUSE").exists()

    iterations = state.get("iterations", [])
    for it in iterations:
        idx = it.get("iteration", 0)
        if idx < 1:
            continue
        analysis_path = ws / f"iteration_{idx}_analysis.json"
        if analysis_path.exists() and not it.get("analysis"):
            it["analysis"] = _safe_json_load(analysis_path)

    if state.get("status") not in ("running", "paused"):
        return

    current_iter = len(iterations) + 1
    for check_iter in [current_iter, current_iter - 1]:
        if check_iter < 1:
            continue
        result_path = ws / f"_result_{check_iter}.json"
        analysis_path = ws / f"iteration_{check_iter}_analysis.json"
        already = any(it.get("iteration") == check_iter for it in iterations)
        if not already and result_path.exists():
            raw_result = _safe_json_load(result_path, {})
            raw_result.pop("feature_stats", None)
            raw_result.pop("feature_analysis", None)
            analysis = _safe_json_load(analysis_path) if analysis_path.exists() else None
            iterations.append({
                "iteration": check_iter,
                "timestamp": state.get("last_updated", ""),
                "metrics": raw_result,
                "analysis": analysis,
                "activity": [],
                "thoughts": [],
                "tokens": {"input": 0, "output": 0, "cost": 0},
                "elapsed_s": 0,
                "result_text": "",
                "code_path": f"iteration_{check_iter}.py",
                "has_error": "error" in raw_result,
                "done_signal": False,
                "timed_out": False,
                "_partial": True,
            })

    stdout_path = AGENTS_DIR / agent_id / "stdout.log"
    if not stdout_path.exists():
        return
    tail_bytes = 64 * 1024
    try:
        fsize = stdout_path.stat().st_size
        with open(stdout_path, "r", errors="replace") as sf:
            if fsize > tail_bytes:
                sf.seek(fsize - tail_bytes)
                sf.readline()
            raw = sf.read()
    except OSError:
        return
    lines = raw.strip().split("\n")
    marker = f"[iter {current_iter}]"
    activity = []
    for line in lines:
        if marker in line:
            activity = []
        stripped = line.strip()
        if stripped:
            activity.append(stripped)
    state["live_activity"] = activity[-30:]


METRIC_INFO = {
    "binary": {
        "primary": "mean_auc",
        "all": ["mean_auc"],
        "direction": "higher",
        "label": "AUC",
    },
    "hitfirst": {
        "primary": "direct_log_loss",
        "all": ["direct_log_loss", "up_auc", "dn_auc"],
        "direction": "lower",
        "label": "Log Loss",
    },
    "lbfgs": {
        "primary": "mean_balanced_accuracy",
        "all": ["mean_balanced_accuracy"],
        "direction": "higher",
        "label": "Bal Acc",
    },
    "breakout": {
        "primary": "mean_auc",
        "all": ["mean_auc"],
        "direction": "higher",
        "label": "AUC",
    },
    "xsec_rank": {
        "primary": "mean_spearman",
        "all": ["mean_spearman"],
        "direction": "higher",
        "label": "Spearman",
    },
}


def _get_metric_info(challenge_type):
    return METRIC_INFO.get(challenge_type, METRIC_INFO["binary"])


# ---- HTML routes ----

CHALLENGE_DESCRIPTIONS = {
    "ETH-1H-BINARY": {"short": "ETH 1h direction", "desc": "Predict whether ETH goes up or down in the next hour", "asset": "ETH", "type": "binary", "metric": "AUC"},
    "ETH-HITFIRST-100M": {"short": "ETH hit-first", "desc": "Predict which barrier ETH hits first within 100 minutes", "asset": "ETH", "type": "hitfirst", "metric": "Log Loss"},
    "ETH-LBFGS": {"short": "ETH bucket forecast", "desc": "Forecast ETH return distribution into z-score buckets", "asset": "ETH", "type": "lbfgs", "metric": "Bal Acc"},
    "BTC-LBFGS-6H": {"short": "BTC 6h bucket forecast", "desc": "Forecast BTC 6-hour return distribution into z-score buckets", "asset": "BTC", "type": "lbfgs", "metric": "Bal Acc"},
    "MULTI-BREAKOUT": {"short": "Multi-asset breakout", "desc": "Predict range breakout continuation vs reversal across assets", "asset": "Multi", "type": "breakout", "metric": "AUC"},
    "XSEC-RANK": {"short": "Cross-section rank", "desc": "Rank which assets will outperform the cross-sectional median", "asset": "Multi", "type": "xsec_rank", "metric": "Spearman"},
}


@app.route("/")

def dashboard():
    summaries = _list_agents_summary()
    from model_iteration_tool.evaluator import CHALLENGES
    challenge_names = list(CHALLENGES.keys())
    challenge_metrics = {}
    challenge_info = {}
    for cname, cfg in CHALLENGES.items():
        challenge_metrics[cname] = _get_metric_info(cfg.challenge_type)
        challenge_info[cname] = CHALLENGE_DESCRIPTIONS.get(cname, {"short": cname, "desc": "", "asset": "?", "type": cfg.challenge_type, "metric": "?"})
    cg_key = _get_coinglass_key()
    cg_masked = _mask_key(cg_key)
    ant_key = _get_anthropic_key()
    ant_masked = _mask_key(ant_key, prefix_len=8)
    def _safe_json_script(obj):
        return json.dumps(obj, default=str).replace("</", "<\\/")

    return render_template("dashboard.html",
                           agents_json=_safe_json_script(summaries),
                           challenges=challenge_names,
                           challenge_metrics=_safe_json_script(challenge_metrics),
                           challenge_info=_safe_json_script(challenge_info),
                           cg_key_set=bool(cg_key),
                           cg_key_masked=cg_masked or "",
                           ant_key_set=bool(ant_key),
                           ant_key_masked=ant_masked or "",
                           )


@app.route("/agent/<agent_id>")

def agent_detail(agent_id):
    _validate_agent_id(agent_id)
    return dashboard()


# ---- API routes ----

@app.route("/api/challenges", methods=["GET"])

def api_list_challenges():
    from model_iteration_tool.evaluator import CHALLENGES
    result = {}
    for cname, cfg in CHALLENGES.items():
        desc = CHALLENGE_DESCRIPTIONS.get(cname, {})
        result[cname] = {
            "type": cfg.challenge_type,
            "short": desc.get("short", cname),
            "desc": desc.get("desc", ""),
            "asset": desc.get("asset", "?"),
            "metric": desc.get("metric", "?"),
            "metric_info": _get_metric_info(cfg.challenge_type),
        }
    return jsonify(result)


@app.route("/api/agents", methods=["GET"])

def api_list_agents():
    full = request.args.get("full", "0") == "1"
    data = _list_agents() if full else _list_agents_summary()
    body = json.dumps(data, default=str)
    etag = hashlib.sha256(body.encode()).hexdigest()[:16]
    if request.headers.get("If-None-Match") == etag:
        return "", 304
    resp = app.response_class(body, mimetype="application/json")
    resp.headers["ETag"] = etag
    return resp


@app.route("/api/agents/<agent_id>", methods=["GET"])

def api_get_agent(agent_id):
    _validate_agent_id(agent_id)
    state = _get_agent(agent_id)
    if not state:
        abort(404)
    return jsonify(state)


@app.route("/api/agents", methods=["POST"])

def api_create_agent():
    data = _safe_json_body()
    challenge = str(data.get("challenge", ""))[:64]
    goal = str(data.get("goal", ""))[:10000]
    if not challenge or not goal:
        return jsonify({"error": "challenge and goal required"}), 400

    from model_iteration_tool.evaluator import CHALLENGES
    if challenge not in CHALLENGES:
        return jsonify({"error": f"unknown challenge: {challenge}"}), 400

    model = str(data.get("model", "sonnet"))[:32]
    if model not in ("sonnet", "opus", "haiku"):
        model = "sonnet"

    import uuid
    agent_id = uuid.uuid4().hex[:12]

    config = {
        "challenge": challenge,
        "goal": goal,
        "min_iterations": max(1, min(100, _safe_int(data.get("min_iterations"), 5))),
        "max_iterations": max(1, min(500, _safe_int(data.get("max_iterations"), 20))),
        "model": model,
        "days_back": max(60, min(365, _safe_int(data.get("days_back"), 60))),
    }
    n_running = running_count()
    if n_running >= MAX_RUNNING:
        return jsonify({"error": f"max {MAX_RUNNING} concurrent agents reached"}), 429

    ant_key = _get_anthropic_key()
    if not ant_key:
        return jsonify({"error": "Anthropic API key not configured. Set it in Settings first."}), 400

    _launch_agent(agent_id, config)
    return jsonify({"id": agent_id, "status": "starting"}), 201


@app.route("/api/agents/<agent_id>/stop", methods=["POST"])

def api_stop_agent(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)

    (agent_dir / "STOP").touch()
    (agent_dir / "PAUSE").unlink(missing_ok=True)
    _kill_agent_process(agent_id)

    log_path = agent_dir / "log.json"
    if log_path.exists():
        def _mark_stopped(state):
            if state.get("status") == "running":
                state["status"] = "stopped"
        _locked_json_update(log_path, _mark_stopped)
    return jsonify({"status": "stopped"})


@app.route("/api/agents/<agent_id>/delete", methods=["POST"])

def api_delete_agent(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    _kill_agent_process(agent_id)
    import shutil
    shutil.rmtree(agent_dir, ignore_errors=True)
    return jsonify({"status": "deleted"})


@app.route("/api/agents/<agent_id>/code/<int:iteration>", methods=["GET"])

def api_get_code(agent_id, iteration):
    _validate_agent_id(agent_id)
    code_path = AGENTS_DIR / agent_id / "workspace" / f"iteration_{iteration}.py"
    if not code_path.exists():
        code_path = AGENTS_DIR / agent_id / f"iteration_{iteration}.py"
    if not code_path.exists():
        abort(404)
    return code_path.read_text(errors="replace"), 200, {"Content-Type": "text/plain"}


@app.route("/api/agents/<agent_id>/inbox", methods=["GET"])

def api_get_inbox(agent_id):
    _validate_agent_id(agent_id)
    inbox_path = AGENTS_DIR / agent_id / "workspace" / "INBOX.md"
    content = inbox_path.read_text(errors="replace") if inbox_path.exists() else ""
    return jsonify({"content": content})


@app.route("/api/agents/<agent_id>/inbox", methods=["POST"])

def api_set_inbox(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    data = _safe_json_body()
    content = str(data.get("content", ""))[:50000]
    workspace = agent_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "INBOX.md").write_text(content)
    return jsonify({"status": "ok"})


def _chat_append(agent_id, role, text, msg_type="message"):
    p = AGENTS_DIR / agent_id / "chat.json"
    def _append(msgs):
        msgs.append({"role": role, "text": text, "ts": datetime.now().isoformat(), "type": msg_type})
    _locked_json_update(p, _append, default=[])


def _build_agent_context(agent_id):
    agent_dir = AGENTS_DIR / agent_id
    ws = agent_dir / "workspace"
    parts = []
    log_path = agent_dir / "log.json"
    state = _safe_json_load(log_path, {})
    if state:
        cfg = state.get("config", {})
        parts.append(
            f"## Agent Config\n"
            f"Challenge: {cfg.get('challenge')}\n"
            f"Status: {state.get('status')}\n"
            f"Iterations completed: {len(state.get('iterations', []))}"
        )

    for fname in ["GOAL.md", "STATE.md", "HISTORY.md", "notes.txt"]:
        p = ws / fname
        if p.exists():
            content = p.read_text(errors="replace").strip()
            if content:
                parts.append(f"## {fname}\n\n{content[-4000:]}")

    if state:
        for it in state.get("iterations", [])[-3:]:
            m = it.get("metrics", {})
            display_m = {k: v for k, v in m.items() if k not in ("windows",)}
            section = f"## Iteration {it['iteration']} Metrics\n{json.dumps(_sanitize_for_json(display_m), indent=2, default=str)}"
            an = it.get("analysis")
            if an and isinstance(an, dict):
                section += f"\nAnalysis: {json.dumps(_sanitize_for_json(an), indent=2, default=str)}"
            parts.append(section)

    return "\n\n".join(parts)


def _run_ask_claude(agent_id, question):
    if not _ask_semaphore.acquire(blocking=False):
        _chat_append(agent_id, "assistant", "Too many concurrent questions. Try again in a moment.", "ask")
        return
    try:
        agent_dir = AGENTS_DIR / agent_id
        if not agent_dir.exists():
            return
        context = _build_agent_context(agent_id)
        system_prompt = (
            "You are a research assistant analyzing an autonomous prediction strategy agent. "
            "Answer the operator's question using the agent context below. Be concise and specific. "
            "Do not make up data -- only reference what is in the context."
        )
        user_msg = f"{context}\n\n## Operator Question\n\n{question}"
        ant_key = _get_anthropic_key()
        if not ant_key:
            _chat_append(agent_id, "assistant", "Error: Anthropic API key not configured.", "ask")
            return

        try:
            import requests as _req
            resp = _req.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ant_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_msg[:100000]}],
                },
                timeout=120,
            )
            if resp.status_code == 200:
                body = resp.json()
                response = ""
                for block in body.get("content", []):
                    if block.get("type") == "text":
                        response += block.get("text", "")
                response = response.strip() or "No response generated."
            else:
                response = f"Error: Anthropic API returned {resp.status_code}: {resp.text[:500]}"
        except Exception as exc:
            response = f"Error: {str(exc)[:500]}"
        if agent_dir.exists():
            _chat_append(agent_id, "assistant", response[:5000], "ask")
    finally:
        _ask_semaphore.release()


@app.route("/api/agents/<agent_id>/chat", methods=["GET"])

def api_get_chat(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    chat_path = agent_dir / "chat.json"
    msgs = _safe_json_load(chat_path, [])
    paused = (agent_dir / "PAUSE").exists()
    return jsonify({"messages": msgs, "paused": paused})


@app.route("/api/agents/<agent_id>/message", methods=["POST"])

def api_message_agent(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    data = _safe_json_body()
    text = str(data.get("text") or "").strip()[:10000]
    if not text:
        return jsonify({"error": "text required"}), 400

    _chat_append(agent_id, "user", text, "message")

    workspace = agent_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    inbox_path = workspace / "INBOX.md"
    lock_path = workspace / "INBOX.md.lock"
    import fcntl, tempfile
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing = inbox_path.read_text(errors="replace").strip() if inbox_path.exists() else ""
        ts = datetime.now().strftime("%H:%M")
        new_content = f"{existing}\n\n[{ts}] Operator: {text}".strip()
        fd, tmp = tempfile.mkstemp(dir=str(workspace), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(new_content)
        os.replace(tmp, str(inbox_path))
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    return jsonify({"status": "sent"})


@app.route("/api/agents/<agent_id>/ask", methods=["POST"])

def api_ask_agent(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    data = _safe_json_body()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400

    _chat_append(agent_id, "user", text, "ask")
    threading.Thread(target=_run_ask_claude, args=(agent_id, text), daemon=True).start()
    return jsonify({"status": "processing"})


@app.route("/api/agents/<agent_id>/pause", methods=["POST"])

def api_pause_agent(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    (agent_dir / "PAUSE").touch()
    return jsonify({"paused": True})


@app.route("/api/agents/<agent_id>/resume", methods=["POST"])

def api_resume_agent(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    pause_path = agent_dir / "PAUSE"
    pause_path.unlink(missing_ok=True)
    return jsonify({"paused": False})


@app.route("/api/agents/<agent_id>/goal", methods=["POST"])

def api_update_goal(agent_id):
    _validate_agent_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        abort(404)
    data = _safe_json_body()
    new_goal = str(data.get("goal") or "").strip()[:10000]
    if not new_goal:
        return jsonify({"error": "goal required"}), 400

    log_path = agent_dir / "log.json"
    if log_path.exists():
        def _set_goal(state):
            state.setdefault("config", {})["goal"] = new_goal
        _locked_json_update(log_path, _set_goal)

    workspace = agent_dir / "workspace"
    if workspace.exists():
        (workspace / "GOAL.md").write_text(f"# Research Goal\n\n{new_goal}\n")

    return jsonify({"status": "updated"})


@app.route("/api/agents/<agent_id>/notes", methods=["GET"])

def api_get_notes(agent_id):
    _validate_agent_id(agent_id)
    notes_path = AGENTS_DIR / agent_id / "workspace" / "notes.txt"
    try:
        if notes_path.exists():
            return jsonify({"content": notes_path.read_text(errors="replace")})
    except OSError:
        pass
    return jsonify({"content": ""})


@app.route("/api/agents/<agent_id>/state", methods=["GET"])

def api_get_state(agent_id):
    _validate_agent_id(agent_id)
    state_path = AGENTS_DIR / agent_id / "workspace" / "STATE.md"
    if not state_path.exists():
        return jsonify({"content": ""})
    return jsonify({"content": state_path.read_text(errors="replace")})


@app.route("/api/agents/<agent_id>/goal", methods=["GET"])

def api_get_goal(agent_id):
    _validate_agent_id(agent_id)
    goal_path = AGENTS_DIR / agent_id / "workspace" / "GOAL.md"
    if not goal_path.exists():
        return jsonify({"content": ""})
    return jsonify({"content": goal_path.read_text(errors="replace")})


@app.route("/api/agents/<agent_id>/stdout", methods=["GET"])

def api_get_stdout(agent_id):
    _validate_agent_id(agent_id)
    stdout_path = AGENTS_DIR / agent_id / "stdout.log"
    if not stdout_path.exists():
        return jsonify({"content": "", "lines": 0})
    tail = min(max(1, _safe_int(request.args.get("tail"), 200)), 5000)
    tail_bytes = min(tail * 200, 2 * 1024 * 1024)
    try:
        fsize = stdout_path.stat().st_size
        with open(stdout_path, "r", errors="replace") as sf:
            if fsize > tail_bytes:
                sf.seek(fsize - tail_bytes)
                sf.readline()
            raw = sf.read()
    except OSError:
        return jsonify({"content": "", "lines": 0})
    lines = raw.strip().split("\n")
    out = lines[-tail:]
    return jsonify({"content": "\n".join(out), "lines": len(lines), "showing": len(out)})


@app.route("/api/agents/<agent_id>/iterations", methods=["GET"])

def api_get_iterations(agent_id):
    _validate_agent_id(agent_id)
    state = _get_agent(agent_id)
    if not state:
        abort(404)
    iterations = state.get("iterations", [])
    return jsonify({"iterations": iterations, "count": len(iterations)})


@app.route("/api/agents/<agent_id>/iterations/<int:iteration>/features", methods=["GET"])

def api_get_features(agent_id, iteration):
    _validate_agent_id(agent_id)
    state = _get_agent(agent_id)
    if not state:
        abort(404)
    for it in state.get("iterations", []):
        if it.get("iteration") == iteration:
            analysis = it.get("analysis") or {}
            feature_report = analysis.get("feature_report", analysis.get("features", []))
            return jsonify({"iteration": iteration, "feature_report": feature_report})
    abort(404)


@app.route("/api/agents/<agent_id>/iterations/<int:iteration>/metrics", methods=["GET"])

def api_get_metrics(agent_id, iteration):
    _validate_agent_id(agent_id)
    state = _get_agent(agent_id)
    if not state:
        abort(404)
    for it in state.get("iterations", []):
        if it.get("iteration") == iteration:
            return jsonify({"iteration": iteration, "metrics": it.get("metrics", {}), "analysis": it.get("analysis")})
    abort(404)


@app.route("/api/coinglass-key", methods=["GET"])

def api_get_cg_key():
    key = _get_coinglass_key()
    if not key:
        return jsonify({"set": False, "masked": ""})
    return jsonify({"set": True, "masked": _mask_key(key)})


@app.route("/api/coinglass-key", methods=["POST"])

def api_set_cg_key():
    data = _safe_json_body()
    key = str(data.get("key") or "").strip()[:256]
    if not key or not all(c.isalnum() or c in "-_." for c in key):
        return jsonify({"error": "key is required (alphanumeric, hyphens, underscores, dots)"}), 400
    _set_coinglass_key(key)
    return jsonify({"set": True, "masked": _mask_key(key)})


@app.route("/api/coinglass-key", methods=["DELETE"])

def api_delete_cg_key():
    _delete_coinglass_key()
    return jsonify({"set": False})


@app.route("/api/anthropic-key", methods=["GET"])

def api_get_ant_key():
    key = _get_anthropic_key()
    if not key:
        return jsonify({"set": False, "masked": ""})
    return jsonify({"set": True, "masked": _mask_key(key, prefix_len=8)})


@app.route("/api/anthropic-key", methods=["POST"])

def api_set_ant_key():
    data = _safe_json_body()
    key = str(data.get("key") or "").strip()[:256]
    if not key or not key.startswith("sk-ant-"):
        return jsonify({"error": "key must start with sk-ant-"}), 400
    if not all(c.isalnum() or c in "-_" for c in key):
        return jsonify({"error": "key contains invalid characters"}), 400
    _set_anthropic_key(key)
    return jsonify({"set": True, "masked": _mask_key(key, prefix_len=8)})


@app.route("/api/anthropic-key", methods=["DELETE"])

def api_delete_ant_key():
    _delete_anthropic_key()
    return jsonify({"set": False})


@app.route("/api/data-cache", methods=["GET"])

def api_data_cache_status():
    from model_iteration_tool.data_cache import cache_dir_for, is_cached
    import time as _t
    results = {}
    for days in (60, 90, 120, 180):
        manifest_path = cache_dir_for(days) / "manifest.json"
        m = _safe_json_load(manifest_path)
        if m:
            age_h = (_t.time() - m.get("fetched_at", 0)) / 3600
            results[str(days)] = {
                "cached": True,
                "assets": len(m.get("assets", [])),
                "has_coinglass": m.get("has_coinglass", False),
                "age_hours": round(age_h, 1),
            }
        else:
            results[str(days)] = {"cached": False}
    return jsonify(results)


@app.route("/api/data-cache/prefetch", methods=["POST"])

def api_prefetch():
    data = _safe_json_body()
    days_back = max(60, min(365, _safe_int(data.get("days_back"), 60)))
    cg_key = data.get("coinglass_key") or _get_coinglass_key()
    if cg_key and not _get_coinglass_key():
        _set_coinglass_key(cg_key)
    from model_iteration_tool.data_cache import prefetch_background
    started = prefetch_background(days_back, coinglass_api_key=cg_key, force=True)
    if not started:
        return jsonify({"status": "already_running"}), 409
    return jsonify({"status": "started", "days_back": days_back})


@app.route("/api/data-cache/status", methods=["GET"])

def api_prefetch_status():
    from model_iteration_tool.data_cache import get_prefetch_status
    return jsonify(get_prefetch_status())


@app.route("/api/data-cache/delete", methods=["POST"])

def api_delete_cache():
    import shutil
    data = _safe_json_body()
    days_back = max(60, _safe_int(data.get("days_back"), 60))
    from model_iteration_tool.data_cache import cache_dir_for, get_prefetch_status
    if get_prefetch_status().get("running"):
        return jsonify({"error": "Cannot delete while fetch is running"}), 409
    d = cache_dir_for(days_back)
    if d.exists():
        shutil.rmtree(d)
    return jsonify({"status": "deleted", "days_back": days_back})


def _graceful_shutdown(signum, _frame):
    logger.info("Received signal %d, shutting down...", signum)
    sys.exit(0)


def _token_strength_check(token):
    """Refuse to start with a weak auth token."""
    issues = []
    if len(token) < 12:
        issues.append(f"too short ({len(token)} chars, need >= 12)")
    char_classes = sum([
        any(c.islower() for c in token),
        any(c.isupper() for c in token),
        any(c.isdigit() for c in token),
        any(not c.isalnum() for c in token),
    ])
    if char_classes < 2:
        issues.append("needs at least 2 of: lowercase, uppercase, digits, symbols")
    entropy = len(token) * math.log2(max(len(set(token)), 1))
    if entropy < 40:
        issues.append(f"too predictable (estimated {entropy:.0f} bits, need >= 40)")
    if issues:
        print("\n" + "!" * 60)
        print("  REFUSED TO START -- weak MANTIS_AUTH_TOKEN")
        for iss in issues:
            print(f"    - {iss}")
        print("")
        print("  Generate a strong token:")
        print('    export MANTIS_AUTH_TOKEN="$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")"')
        print("!" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("MANTIS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MANTIS_PORT", "8420")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MANTIS_WORKERS", "2")))
    args = parser.parse_args()

    auth_token = os.environ.get("MANTIS_AUTH_TOKEN", "")
    auth_configured = bool(auth_token)

    is_public = args.host not in ("127.0.0.1", "localhost")
    if is_public and not auth_configured:
        print("\n" + "!" * 60)
        print("  REFUSED TO START")
        print("  Binding to a non-localhost address without MANTIS_AUTH_TOKEN")
        print("  is a security risk (bots WILL find you and burn API credits).")
        print("")
        print("  Either:")
        print("    export MANTIS_AUTH_TOKEN=\"$(python3 -c \"import secrets; print(secrets.token_urlsafe(32))\")")
        print("  Or bind to localhost only:")
        print(f"    python -m model_iteration_tool.gui --host 127.0.0.1")
        print("!" * 60 + "\n")
        sys.exit(1)

    if auth_configured:
        _token_strength_check(auth_token)

    print(f"\n{'='*60}")
    print(f"  MANTIS Model Iteration Tool")
    print(f"  http://{args.host}:{args.port}/")
    print(f"{'='*60}")
    if auth_configured:
        print(f"  Auth: enabled (MANTIS_AUTH_TOKEN set)")
        print(f"  Browser: http://{args.host}:{args.port}/?token=<your-token>")
    else:
        print(f"  Auth: localhost-only (set MANTIS_AUTH_TOKEN to expose publicly)")
    print(f"  Health: http://{args.host}:{args.port}/health (always public)")
    print(f"  Agents run in Docker containers (sandboxed, survive GUI restart)")
    print(f"  List containers: docker ps --filter name=mantis-")
    print(f"{'='*60}\n")

    _has_gunicorn = importlib.util.find_spec("gunicorn") is not None

    if _has_gunicorn:
        from gunicorn.app.wsgiapp import run as _gunicorn_run
        logger.info("Starting with gunicorn (%d workers)", args.workers)
        sys.argv = [
            "gunicorn",
            f"--bind={args.host}:{args.port}",
            f"--workers={args.workers}",
            "--timeout=120",
            "--graceful-timeout=30",
            "--access-logfile=-",
            "--error-logfile=-",
            "model_iteration_tool.gui:app",
        ]
        _gunicorn_run()
    else:
        logger.warning("gunicorn not installed -- using Flask dev server (not for production)")
        signal.signal(signal.SIGTERM, _graceful_shutdown)
        signal.signal(signal.SIGINT, _graceful_shutdown)
        app.run(host=args.host, port=args.port, debug=False)
