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


try:
    from mantis_model_iteration_tool.utils import (
        sanitize_for_json as _sanitize_for_json,
        safe_json_load as _safe_json_load,
        locked_json_update as _locked_json_update,
        sanitize_status as _sanitize_status,
        attach_live_activity as _attach_live_activity_shared,
        get_agent_detail as _get_agent_detail_shared,
    )
except ImportError:
    from utils import (
        sanitize_for_json as _sanitize_for_json,
        safe_json_load as _safe_json_load,
        locked_json_update as _locked_json_update,
        sanitize_status as _sanitize_status,
        attach_live_activity as _attach_live_activity_shared,
        get_agent_detail as _get_agent_detail_shared,
    )


AGENTS_DIR = Path(__file__).parent / "agents"
CG_KEY_PATH = Path(__file__).parent / ".coinglass_key"
ANTHROPIC_KEY_PATH = Path(__file__).parent / ".anthropic_key"
TARGON_CONFIG_PATH = Path(__file__).parent / ".targon_config"
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
        has_xhr = request.headers.get("X-Requested-With", "").lower() == "xmlhttprequest"
        is_json = request.content_type and "application/json" in request.content_type
        if not has_xhr and not is_json:
            logger.warning("CSRF: rejected POST without Origin or XHR header for %s", request.path)
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



# ── Targon proxy ─────────────────────────────────────────────────────────────

_PROXY_ROUTES = frozenset({"/api/agents", "/api/challenges"})
_PROXY_PREFIXES = ("/api/agents/",)


def _targon_is_active():
    cfg = _get_targon_config()
    return bool(cfg.get("deployed") and cfg.get("server_url"))


_NO_PROXY_SUFFIXES = ("estimate-salience", "salience-status")

def _should_proxy():
    if not _targon_is_active():
        return False
    path = request.path
    if any(path.endswith(s) for s in _NO_PROXY_SUFFIXES):
        return False
    if path == "/api/agents" and request.method == "GET":
        return False
    if request.method == "GET" and path.startswith("/api/agents/"):
        parts = path.rstrip("/").split("/")
        if len(parts) == 4:
            return False
        if len(parts) >= 5 and parts[4] in ("code", "iterations",
                "state", "notes", "goal", "chat", "stdout", "features",
                "metrics"):
            return False
    if path in _PROXY_ROUTES:
        return True
    return any(path.startswith(p) for p in _PROXY_PREFIXES)


@app.before_request
def _maybe_proxy_to_targon():
    """If the Targon agent server is deployed, proxy agent/challenge
    requests to it. The GUI becomes a thin pass-through for agent ops."""
    if not _should_proxy():
        return None

    import requests as _req

    cfg = _get_targon_config()
    server_url = cfg["server_url"].rstrip("/")
    auth_key = cfg.get("server_auth_key", "")

    target_url = server_url + request.path
    headers = {}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"
    if request.content_type:
        headers["Content-Type"] = request.content_type

    try:
        resp = _req.request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=request.args.to_dict(),
            data=request.get_data() if request.method in ("POST", "PUT", "PATCH") else None,
            timeout=60,
        )
        excluded = {"transfer-encoding", "content-encoding", "connection"}
        resp_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in excluded
        }
        return app.response_class(
            resp.content,
            status=resp.status_code,
            headers=resp_headers,
        )
    except _req.exceptions.Timeout:
        return jsonify({"error": "Targon server timed out"}), 504
    except _req.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach Targon server"}), 502
    except Exception as exc:
        logger.warning("Targon proxy error: %s", exc)
        return jsonify({"error": "Targon proxy error"}), 502


def _targon_get(path: str, *, params: dict | None = None,
                timeout: int = 30):
    """GET a path on the active Targon server. Returns (response, None) on
    success, or (None, error_string) on failure."""
    import requests as _req

    cfg = _get_targon_config()
    srv = cfg.get("server_url", "").rstrip("/")
    if not srv:
        return None, "No Targon server URL"
    headers = {}
    auth = cfg.get("server_auth_key", "")
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    try:
        resp = _req.get(f"{srv}{path}", headers=headers,
                        params=params, timeout=timeout)
        return resp, None
    except Exception as exc:
        return None, str(exc)


def _targon_post(path: str, *, json_body: dict | None = None,
                 timeout: int = 30):
    """POST to the active Targon server. Returns (response, None) on
    success, or (None, error_string) on failure."""
    import requests as _req

    cfg = _get_targon_config()
    srv = cfg.get("server_url", "").rstrip("/")
    if not srv:
        return None, "No Targon server URL"
    headers = {}
    auth = cfg.get("server_auth_key", "")
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    try:
        resp = _req.post(f"{srv}{path}", json=json_body, headers=headers,
                         timeout=timeout)
        return resp, None
    except Exception as exc:
        return None, str(exc)


def _merge_targon_agents(local_agents: list, *, full: bool = False) -> list:
    """Merge local agents with remote Targon agents.

    Remote agents take precedence for running/active agents (same ID).
    Local-only agents (completed, stopped, etc.) are always included so
    they remain visible even after a Targon instance is discontinued.

    Always fetches full data from Targon to compute cost summaries, then
    strips iterations in summary mode.
    """
    resp, err = _targon_get("/api/agents", params={"full": "1"})
    if err or not resp or resp.status_code != 200:
        if err:
            logger.warning("Failed to fetch Targon agents: %s", err)
        return local_agents
    try:
        remote_agents = resp.json()
    except (ValueError, Exception):
        return local_agents
    if not isinstance(remote_agents, list):
        return local_agents

    for agent in remote_agents:
        if agent.get("total_cost") is None:
            tc = 0.0
            ti = 0
            to_ = 0
            for it in agent.get("iterations", []):
                tok = it.get("tokens") or {}
                tc += float(tok.get("cost", 0) or 0)
                ti += int(tok.get("input", 0) or 0)
                to_ += int(tok.get("output", 0) or 0)
            agent["total_cost"] = tc
            agent["total_tokens"] = ti + to_
        if not full:
            iters = agent.get("iterations", [])
            agent["iteration_count"] = agent.get("iteration_count") or len(iters)
            if iters:
                last = iters[-1]
                agent.setdefault("last_metrics", last.get("metrics", {}))
                agent.setdefault("has_error", last.get("has_error", False))
            agent.pop("iterations", None)

    local_by_id = {a.get("id", a.get("agent_id", "")): a for a in local_agents}
    remote_by_id = {a.get("id", a.get("agent_id", "")): a for a in remote_agents}

    merged = {}
    for aid, agent in remote_by_id.items():
        agent["_source"] = "targon"
        merged[aid] = agent

    for aid, agent in local_by_id.items():
        if aid in merged:
            continue
        agent["_source"] = "local"
        merged[aid] = agent

    result = list(merged.values())
    result.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return result


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


def _get_targon_config():
    if TARGON_CONFIG_PATH.exists():
        raw = TARGON_CONFIG_PATH.read_text().strip()
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
    return {}


def _save_targon_config(cfg: dict):
    TARGON_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    TARGON_CONFIG_PATH.chmod(0o600)


def _delete_targon_config():
    if TARGON_CONFIG_PATH.exists():
        TARGON_CONFIG_PATH.unlink()



def _validate_agent_id(agent_id):
    if not _SAFE_ID.match(agent_id):
        abort(400)
    return agent_id


# ---- container helpers ----

from mantis_model_iteration_tool.sandbox import (
    ensure_image, launch_container, kill_container,
    container_running, running_count, MAX_RUNNING,
    ensure_miner_image, launch_miner_container, kill_miner_container,
    miner_running,
    run_miner_registration, run_miner_check_registration,
    run_miner_generate_wallet, run_miner_check_balance,
)
DATA_DIR = Path(__file__).parent / ".data"
MINER_DIR = Path(__file__).parent / ".miner"
MINER_CONFIG_PATH = Path(__file__).parent / ".miner_config"

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_WALLETS_DIR = ARTIFACTS_DIR / "wallets"
ARTIFACTS_MODELS_DIR = ARTIFACTS_DIR / "models"

for _d in (ARTIFACTS_DIR, ARTIFACTS_WALLETS_DIR, ARTIFACTS_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _backfill_model_artifacts():
    """One-time copy of existing agent iteration files into artifacts/models/,
    including metadata extracted from the agent's log.json."""
    import shutil

    if not AGENTS_DIR.exists():
        return
    for agent_dir in AGENTS_DIR.iterdir():
        if not agent_dir.is_dir():
            continue
        ws = agent_dir / "workspace"
        if not ws.is_dir():
            continue

        state = _safe_json_load(agent_dir / "log.json")
        iter_lookup = {}
        if state:
            challenge = (state.get("config") or {}).get("challenge", "")
            for it in state.get("iterations", []):
                iter_lookup[it.get("iteration")] = {
                    "challenge": challenge,
                    "timestamp": it.get("timestamp", ""),
                    "metrics": it.get("metrics", {}),
                }

        for src in sorted(ws.glob("iteration_*.py")):
            dest_dir = ARTIFACTS_MODELS_DIR / agent_dir.name
            dest = dest_dir / src.name
            if dest.exists():
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

            try:
                iter_num = int(src.stem.replace("iteration_", ""))
            except ValueError:
                continue
            meta_path = dest_dir / f"iteration_{iter_num}_meta.json"
            if not meta_path.exists() and iter_num in iter_lookup:
                meta = {"agent_id": agent_dir.name, "iteration": iter_num,
                        **iter_lookup[iter_num]}
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")

            logger.info("Backfilled model artifact: %s", dest)


_backfill_model_artifacts()


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

    ensure_image(_PKG_DIR_STR)

    try:
        launch_container(
            agent_id=agent_id,
            config=config,
            agent_dir=str(agent_dir),
            pkg_dir=_PKG_DIR_STR,
            data_dir=str(DATA_DIR),
            anthropic_key=ant_key,
            coinglass_key=cg_key,
        )
    except Exception:
        import shutil
        shutil.rmtree(agent_dir, ignore_errors=True)
        raise


# ---- agent state helpers ----

def _list_agents(full=False):
    """List agents. full=False returns lightweight summaries; full=True returns
    complete state including iterations and live activity."""
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

        if full:
            _attach_live_activity_shared(state, AGENTS_DIR, agent_id)
            agents.append(state)
        else:
            iterations = state.get("iterations", [])
            total_cost = 0.0
            total_tokens = 0
            for it in iterations:
                tok = it.get("tokens") or {}
                total_cost += float(tok.get("cost", 0) or 0)
                total_tokens += int(tok.get("input", 0) or 0) + int(tok.get("output", 0) or 0)
            summary = {
                "id": state.get("id", agent_id),
                "status": state["status"],
                "config": state.get("config", {}),
                "created_at": state.get("created_at", ""),
                "iteration_count": len(iterations),
                "paused": (d / "PAUSE").exists(),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
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


def _get_agent(agent_id):
    return _get_agent_detail_shared(AGENTS_DIR, agent_id, _agent_is_alive)


METRIC_INFO = {
    "binary": {
        "primary": "mean_auc",
        "all": ["mean_auc"],
        "direction": "higher",
        "label": "AUC",
        "baseline": 0.5,
    },
    "hitfirst": {
        "primary": "direct_log_loss",
        "all": ["direct_log_loss", "up_auc", "dn_auc"],
        "direction": "lower",
        "label": "Log Loss",
        "baseline": 0.693,
    },
    "lbfgs": {
        "primary": "mean_balanced_accuracy",
        "all": ["mean_balanced_accuracy"],
        "direction": "higher",
        "label": "Bal Acc",
        "baseline": 0.2,
    },
    "breakout": {
        "primary": "mean_auc",
        "all": ["mean_auc"],
        "direction": "higher",
        "label": "AUC",
        "baseline": 0.5,
    },
    "xsec_rank": {
        "primary": "mean_spearman",
        "all": ["mean_spearman"],
        "direction": "higher",
        "label": "Spearman",
        "baseline": 0.0,
    },
    "funding_xsec": {
        "primary": "mean_spearman",
        "all": ["mean_spearman"],
        "direction": "higher",
        "label": "Spearman",
        "baseline": 0.0,
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
    "FUNDING-XSEC": {"short": "Funding rate cross-section", "desc": "Predict which assets' funding rate changes will exceed the cross-sectional median over 8 hours", "asset": "Multi (20)", "type": "funding_xsec", "metric": "Spearman"},
}


def _safe_json_script(obj):
    return json.dumps(obj, default=str).replace("</", "<\\/")


_PKG_DIR_STR = str(Path(__file__).parent)


def _ensure_miner_image_or_500():
    """Build miner Docker image, return error response or None."""
    try:
        ensure_miner_image(_PKG_DIR_STR)
        return None
    except Exception as exc:
        return jsonify({"error": f"Failed to build miner image: {exc}"}), 500


def _anthropic_chat(api_key, system, user_msg, *, max_tokens=2048, timeout=120):
    """Call Anthropic messages API, return text or error string."""
    import requests as _req
    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user_msg[:100000]}],
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return f"Error: Anthropic API {resp.status_code}: {resp.text[:500]}"
        return "".join(
            b.get("text", "") for b in resp.json().get("content", [])
            if b.get("type") == "text"
        ).strip() or "No response generated."
    except Exception as exc:
        return f"Error: {str(exc)[:500]}"


def _build_challenge_catalog():
    from mantis_model_iteration_tool.evaluator import CHALLENGES
    catalog = {}
    for cname, cfg in CHALLENGES.items():
        desc = CHALLENGE_DESCRIPTIONS.get(cname, {})
        catalog[cname] = {
            "type": cfg.challenge_type,
            "short": desc.get("short", cname),
            "desc": desc.get("desc", ""),
            "asset": desc.get("asset", "?"),
            "metric": desc.get("metric", "?"),
            "metric_info": _get_metric_info(cfg.challenge_type),
        }
    return catalog


@app.route("/")
def dashboard():
    summaries = _list_agents(full=False)
    if _targon_is_active():
        summaries = _merge_targon_agents(summaries, full=False)
    catalog = _build_challenge_catalog()
    cg_key = _get_coinglass_key()
    ant_key = _get_anthropic_key()
    targon_cfg = _get_targon_config()
    targon_api_key = targon_cfg.get("api_key", "")
    return render_template("dashboard.html",
                           agents_json=_safe_json_script(summaries),
                           challenges=list(catalog.keys()),
                           challenge_metrics=_safe_json_script(
                               {k: v["metric_info"] for k, v in catalog.items()}),
                           challenge_info=_safe_json_script(catalog),
                           cg_key_set=bool(cg_key),
                           cg_key_masked=_mask_key(cg_key) or "",
                           ant_key_set=bool(ant_key),
                           ant_key_masked=_mask_key(ant_key, prefix_len=8) or "",
                           targon_key_set=bool(targon_api_key),
                           targon_key_masked=_mask_key(targon_api_key) or "",
                           targon_server_url=targon_cfg.get("server_url", ""),
                           targon_deployed=targon_cfg.get("deployed", False),
                           targon_mode=targon_cfg.get("mode", ""),
                           )


@app.route("/agent/<agent_id>")
def agent_detail(agent_id):
    _validate_agent_id(agent_id)
    return dashboard()


# ---- API routes ----

@app.route("/api/challenges", methods=["GET"])
def api_list_challenges():
    return jsonify(_build_challenge_catalog())


@app.route("/api/agents", methods=["GET"])
def api_list_agents():
    full = request.args.get("full", "0") == "1"
    local_data = _list_agents(full=full)

    if _targon_is_active():
        local_data = _merge_targon_agents(local_data, full=full)

    data = local_data
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

    if _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}")
        if resp and resp.status_code == 200:
            return jsonify(resp.json())

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

    from mantis_model_iteration_tool.evaluator import CHALLENGES
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


@app.route("/api/agents/<agent_id>/clone", methods=["POST"])
def api_clone_agent(agent_id):
    """Clone an agent, optionally forking from a specific iteration."""
    _validate_agent_id(agent_id)
    agent = _get_agent(agent_id)
    if not agent and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}")
        if resp and resp.status_code == 200:
            agent = resp.json()
    if not agent:
        abort(404)

    data = _safe_json_body()
    src_cfg = agent.get("config") or {}
    new_goal = str(data.get("goal", src_cfg.get("goal", "")))[:10000]
    fork_iter = data.get("iteration")

    ant_key = _get_anthropic_key()
    if not ant_key:
        return jsonify({"error": "Anthropic API key not configured"}), 400

    import uuid
    new_id = uuid.uuid4().hex[:12]
    config = {
        "challenge": src_cfg.get("challenge", ""),
        "goal": new_goal,
        "min_iterations": src_cfg.get("min_iterations", 5),
        "max_iterations": _safe_int(data.get("max_iterations"), src_cfg.get("max_iterations", 20)),
        "model": src_cfg.get("model", "sonnet"),
        "days_back": src_cfg.get("days_back", 60),
    }

    from mantis_model_iteration_tool.evaluator import CHALLENGES
    if config["challenge"] not in CHALLENGES:
        return jsonify({"error": f"unknown challenge: {config['challenge']}"}), 400

    n_running = running_count()
    if n_running >= MAX_RUNNING:
        return jsonify({"error": f"max {MAX_RUNNING} concurrent agents reached"}), 429

    new_dir = AGENTS_DIR / new_id
    new_dir.mkdir(parents=True, exist_ok=True)
    ws = new_dir / "workspace"
    ws.mkdir(exist_ok=True)

    if fork_iter is not None:
        fork_iter = int(fork_iter)
        code_path = _resolve_strategy_file(agent_id, fork_iter)
        if code_path:
            code = Path(code_path).read_text(errors="replace")
            inbox = (f"# Forked from agent {agent_id}, iteration {fork_iter}\n\n"
                     f"Use this working strategy as your starting point. "
                     f"Improve it according to the research goal.\n\n"
                     f"```python\n{code}\n```\n")
            (ws / "INBOX.md").write_text(inbox)

    _launch_agent(new_id, config)
    return jsonify({"id": new_id, "cloned_from": agent_id,
                    "fork_iteration": fork_iter}), 201


@app.route("/api/agents/<agent_id>/export", methods=["GET"])
def api_export_agent(agent_id):
    """Export an agent as a self-contained JSON bundle for backup/transfer."""
    _validate_agent_id(agent_id)
    agent = _get_agent(agent_id)
    if not agent and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}")
        if resp and resp.status_code == 200:
            agent = resp.json()
    if not agent:
        abort(404)

    bundle = {
        "format": "mantis-agent-export-v1",
        "exported_at": datetime.now().isoformat(),
        "agent": agent,
        "code": {},
    }
    for it in agent.get("iterations", []):
        iter_num = it.get("iteration")
        if iter_num is None:
            continue
        code_path = _resolve_strategy_file(agent_id, iter_num)
        if code_path and Path(code_path).exists():
            bundle["code"][str(iter_num)] = Path(code_path).read_text(errors="replace")

    from flask import Response
    resp = Response(
        json.dumps(bundle, indent=2, default=str),
        mimetype="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="agent_{agent_id}.json"'
        },
    )
    return resp


@app.route("/api/agents/import", methods=["POST"])
def api_import_agent():
    """Import an agent from a previously exported JSON bundle."""
    data = _safe_json_body()
    if data.get("format") != "mantis-agent-export-v1":
        return jsonify({"error": "Invalid or missing export format. Expected mantis-agent-export-v1"}), 400

    agent_data = data.get("agent")
    if not agent_data:
        return jsonify({"error": "No agent data in bundle"}), 400

    import uuid
    new_id = uuid.uuid4().hex[:12]
    agent_dir = AGENTS_DIR / new_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    ws = agent_dir / "workspace"
    ws.mkdir(exist_ok=True)

    import datetime as _dt
    state = {
        "id": new_id,
        "status": "completed",
        "config": agent_data.get("config", {}),
        "iteration": len(agent_data.get("iterations", [])),
        "iterations": agent_data.get("iterations", []),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "imported_from": agent_data.get("id", "unknown"),
    }
    (agent_dir / "log.json").write_text(json.dumps(state, indent=2))

    code_map = data.get("code", {})
    for iter_str, code_text in code_map.items():
        (ws / f"iteration_{iter_str}.py").write_text(code_text)

    return jsonify({"id": new_id, "imported": True, "iterations": len(code_map)}), 201


@app.route("/api/agents/<agent_id>/code/<int:iteration>", methods=["GET"])
def api_get_code(agent_id, iteration):
    _validate_agent_id(agent_id)
    fname = f"iteration_{iteration}.py"

    code_path = AGENTS_DIR / agent_id / "workspace" / fname
    if not code_path.exists():
        code_path = ARTIFACTS_MODELS_DIR / agent_id / fname
    if not code_path.exists():
        code_path = AGENTS_DIR / agent_id / fname
    if code_path.exists():
        return code_path.read_text(errors="replace"), 200, {"Content-Type": "text/plain"}

    if _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}/code/{iteration}")
        if resp and resp.status_code == 200:
            return resp.text, 200, {"Content-Type": "text/plain"}

    abort(404)


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
    from mantis_model_iteration_tool.utils import chat_append
    chat_append(AGENTS_DIR / agent_id / "chat.json", role, text, msg_type)


def _build_agent_context(agent_id):
    from mantis_model_iteration_tool.utils import build_agent_context
    return build_agent_context(AGENTS_DIR, agent_id)


_ask_semaphore = threading.Semaphore(3)


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
        response = _anthropic_chat(ant_key, system_prompt, user_msg)
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
    if not state and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}/iterations")
        if resp and resp.status_code == 200:
            return jsonify(resp.json())
    if not state:
        abort(404)
    iterations = state.get("iterations", [])
    return jsonify({"iterations": iterations, "count": len(iterations)})


@app.route("/api/agents/<agent_id>/iterations/<int:iteration>/features", methods=["GET"])
def api_get_features(agent_id, iteration):
    _validate_agent_id(agent_id)
    state = _get_agent(agent_id)
    if not state and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}/iterations/{iteration}/features")
        if resp and resp.status_code == 200:
            return jsonify(resp.json())
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
    if not state and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}/iterations/{iteration}/metrics")
        if resp and resp.status_code == 200:
            return jsonify(resp.json())
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


@app.route("/api/targon", methods=["GET"])
def api_get_targon():
    cfg = _get_targon_config()
    api_key = cfg.get("api_key", "")
    from mantis_model_iteration_tool.targon_deploy import (
        targon_cli_available, targon_sdk_importable, get_deploy_state,
        status_targon,
    )
    deploy_state = get_deploy_state()
    sdk_ok = targon_sdk_importable()
    cli_ok = targon_cli_available()

    deployed = cfg.get("deployed", False)
    server_url = cfg.get("server_url", "")
    app_id = cfg.get("app_id", "")

    if not deployed and api_key and cli_ok:
        live = status_targon(api_key)
        if live.get("status") == "deployed":
            deployed = True
            app_id = live.get("app_id", app_id)
            server_url = live.get("url", server_url) or server_url
            cfg["deployed"] = True
            if app_id:
                cfg["app_id"] = app_id
            if server_url:
                cfg["server_url"] = server_url
            _save_targon_config(cfg)

    return jsonify({
        "api_key_set": bool(api_key),
        "api_key_masked": _mask_key(api_key) if api_key else "",
        "sdk_installed": sdk_ok,
        "cli_installed": cli_ok,
        "deployed": deployed,
        "mode": cfg.get("mode", ""),
        "server_url": server_url,
        "app_id": app_id,
        "deploy_state": deploy_state,
    })


@app.route("/api/targon/key", methods=["POST"])
def api_set_targon_key():
    data = _safe_json_body()
    api_key = str(data.get("api_key") or "").strip()[:256]
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400
    cfg = _get_targon_config()
    cfg["api_key"] = api_key
    _save_targon_config(cfg)
    return jsonify({
        "api_key_set": True,
        "api_key_masked": _mask_key(api_key),
    })


@app.route("/api/targon/key", methods=["DELETE"])
def api_delete_targon_key():
    cfg = _get_targon_config()
    cfg.pop("api_key", None)
    _save_targon_config(cfg)
    return jsonify({"api_key_set": False})


@app.route("/api/targon/deploy", methods=["POST"])
def api_targon_deploy():
    """Deploy the full agent server to Targon."""
    cfg = _get_targon_config()

    if not cfg.get("api_key"):
        return jsonify({"error": "Set your Targon API key first"}), 400

    ant_key = _get_anthropic_key()
    if not ant_key:
        return jsonify({"error": "Set your Anthropic API key first (needed on the server)"}), 400

    import secrets
    server_auth_key = cfg.get("server_auth_key") or secrets.token_urlsafe(32)

    cfg["mode"] = "targon"
    cfg["server_auth_key"] = server_auth_key
    _save_targon_config(cfg)

    cg_key = _get_coinglass_key() or ""

    from mantis_model_iteration_tool.targon_deploy import deploy_async
    started = deploy_async(
        mode="targon",
        targon_api_key=cfg.get("api_key", ""),
        server_auth_key=server_auth_key,
        anthropic_key=ant_key,
        coinglass_key=cg_key,
    )
    if not started:
        return jsonify({"error": "Deployment already in progress"}), 409
    return jsonify({"status": "deploying", "mode": "targon"})


@app.route("/api/targon/deploy/status", methods=["GET"])
def api_targon_deploy_status():
    from mantis_model_iteration_tool.targon_deploy import get_deploy_state
    state = get_deploy_state()

    if state.get("status") == "deployed":
        cfg = _get_targon_config()
        changed = False
        if not cfg.get("deployed"):
            cfg["deployed"] = True
            changed = True
        url = state.get("url", "")
        if url and cfg.get("server_url") != url:
            cfg["server_url"] = url
            changed = True
        app_id = state.get("app_id", "")
        if app_id and cfg.get("app_id") != app_id:
            cfg["app_id"] = app_id
            changed = True
        if changed:
            _save_targon_config(cfg)
            cg_key = _get_coinglass_key()
            if cg_key and not _load_cg_state().get("running"):
                t = threading.Thread(target=_upload_cg_to_targon,
                                     args=(60,), daemon=True)
                t.start()

    return jsonify(state)


@app.route("/api/targon/teardown", methods=["POST"])
def api_targon_teardown():
    cfg = _get_targon_config()
    try:
        from mantis_model_iteration_tool.targon_deploy import teardown_targon, _clear_state
        api_key = cfg.get("api_key", "")
        teardown_targon(api_key)
        _clear_state()
    except Exception as exc:
        logger.warning("Teardown error: %s", exc)

    cfg.pop("server_url", None)
    cfg.pop("deployed", None)
    cfg.pop("app_id", None)
    cfg.pop("mode", None)
    cfg.pop("server_auth_key", None)
    _save_targon_config(cfg)
    return jsonify({"status": "torn_down"})


@app.route("/api/targon/health", methods=["GET"])
def api_targon_health():
    cfg = _get_targon_config()
    url = cfg.get("server_url", "")
    if not url:
        return jsonify({"error": "Agent server not deployed"}), 400
    auth_key = cfg.get("server_auth_key", "")
    try:
        import requests as _req
        headers = {}
        if auth_key:
            headers["Authorization"] = f"Bearer {auth_key}"
        resp = _req.get(f"{url.rstrip('/')}/health", headers=headers, timeout=10)
        resp.raise_for_status()
        return jsonify({"status": "connected", "remote_health": resp.json()})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)[:500]})


CG_UPLOAD_STATE_PATH = Path(__file__).parent / ".cg_upload_state.json"


def _save_cg_state(state):
    CG_UPLOAD_STATE_PATH.write_text(json.dumps(state))


def _load_cg_state():
    if CG_UPLOAD_STATE_PATH.exists():
        try:
            return json.loads(CG_UPLOAD_STATE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"running": False, "error": None, "result": None}


def _upload_cg_to_targon(days_back=60):
    """Fetch CoinGlass data locally and upload to Targon."""
    import io
    import tarfile
    import time as _t
    import numpy as np
    import requests as _req

    _save_cg_state({"running": True, "error": None, "result": None})
    try:
        cg_key = _get_coinglass_key()
        if not cg_key:
            _save_cg_state({"running": False, "error": "No CoinGlass key"})
            return
        cfg = _get_targon_config()
        server_url = cfg.get("server_url", "")
        auth_key = cfg.get("server_auth_key", "")
        if not server_url:
            _save_cg_state({"running": False, "error": "No Targon server URL"})
            return

        from mantis_model_iteration_tool.data import BREAKOUT_ASSETS
        from mantis_model_iteration_tool.coinglass import fetch_coinglass_features

        now_ms = int(_t.time() * 1000)
        now_ms = now_ms - (now_ms % 60_000)  # snap to minute boundary
        start_ms = now_ms - days_back * 86_400_000
        n_minutes = days_back * 1440
        minute_ts = np.arange(n_minutes, dtype=np.int64) * 60_000 + start_ms

        buf = io.BytesIO()
        count = 0
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for asset in BREAKOUT_ASSETS:
                logger.info("CG upload: fetching %s...", asset)
                _save_cg_state({"running": True, "error": None,
                                "result": None, "progress": f"Fetching {asset} ({count}/{len(BREAKOUT_ASSETS)})"})
                feats = fetch_coinglass_features(
                    asset, minute_ts, cg_key, days_back=days_back)
                if not feats:
                    continue
                npz_buf = io.BytesIO()
                np.savez_compressed(npz_buf, **feats)
                npz_buf.seek(0)
                info = tarfile.TarInfo(name=f"{asset}.npz")
                info.size = len(npz_buf.getvalue())
                tar.addfile(info, npz_buf)
                count += 1

        if count == 0:
            _save_cg_state({"running": False, "error": "No CG data fetched"})
            return

        _save_cg_state({"running": True, "error": None,
                        "result": None, "progress": f"Waiting for container..."})
        for _ in range(30):
            try:
                h = _req.get(f"{server_url.rstrip('/')}/health", timeout=10)
                if h.status_code == 200:
                    break
            except _req.RequestException:
                pass
            _t.sleep(10)

        _save_cg_state({"running": True, "error": None,
                        "result": None, "progress": f"Uploading {count} assets..."})
        headers = {"Authorization": f"Bearer {auth_key}"}
        resp = _req.post(
            f"{server_url.rstrip('/')}/api/coinglass-bundle",
            data=buf.getvalue(),
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info("CG upload complete: %s", result)
        _save_cg_state({"running": False, "result": result})
    except Exception as exc:
        logger.error("CG upload failed: %s", exc)
        _save_cg_state({"running": False, "error": str(exc)[:500]})


@app.route("/api/targon/upload-coinglass", methods=["POST"])
def api_upload_coinglass():
    state = _load_cg_state()
    if state.get("running"):
        return jsonify({"status": "already_running"}), 409
    data = _safe_json_body()
    days_back = max(60, min(365, _safe_int(data.get("days_back"), 60)))
    t = threading.Thread(target=_upload_cg_to_targon, args=(days_back,), daemon=True)
    t.start()
    return jsonify({"status": "started", "days_back": days_back})


@app.route("/api/targon/upload-coinglass/status", methods=["GET"])
def api_upload_coinglass_status():
    return jsonify(_load_cg_state())


@app.route("/api/data-cache", methods=["GET"])
def api_data_cache_status():
    from mantis_model_iteration_tool.data_cache import cache_dir_for, audit_cache
    import time as _t
    results = {}
    for days in (60, 90, 120, 180):
        manifest_path = cache_dir_for(days) / "manifest.json"
        m = _safe_json_load(manifest_path)
        audit = audit_cache(days)

        if audit["exists"]:
            age_h = None
            if m:
                age_h = round((_t.time() - m.get("fetched_at", 0)) / 3600, 1)
            results[str(days)] = {
                "cached": True,
                "assets": audit["assets"],
                "expected_assets": audit["expected_assets"],
                "has_coinglass": audit["has_coinglass"],
                "coinglass_assets": audit["coinglass_assets"],
                "coinglass_key_set": bool(_get_coinglass_key()),
                "age_hours": age_h,
                "source": audit["source"],
                "row_pct": audit["row_pct"],
                "complete": audit["complete"],
                "per_asset": audit["per_asset"],
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
    from mantis_model_iteration_tool.data_cache import prefetch_background
    started = prefetch_background(days_back, coinglass_api_key=cg_key, force=True)
    if not started:
        return jsonify({"status": "already_running"}), 409
    return jsonify({"status": "started", "days_back": days_back})


@app.route("/api/data-cache/status", methods=["GET"])
def api_prefetch_status():
    from mantis_model_iteration_tool.data_cache import get_prefetch_status
    return jsonify(get_prefetch_status())


@app.route("/api/data-cache/delete", methods=["POST"])
def api_delete_cache():
    import shutil
    data = _safe_json_body()
    days_back = max(60, _safe_int(data.get("days_back"), 60))
    from mantis_model_iteration_tool.data_cache import cache_dir_for, get_prefetch_status
    if get_prefetch_status().get("running"):
        return jsonify({"error": "Cannot delete while fetch is running"}), 409
    d = cache_dir_for(days_back)
    if d.exists():
        shutil.rmtree(d)
    return jsonify({"status": "deleted", "days_back": days_back})


# ── Salience estimation ──────────────────────────────────────────────────────

SALIENCE_STATE_PATH = Path(__file__).parent / ".salience_state.json"
DATALOG_DL_STATE_PATH = Path(__file__).parent / ".datalog_dl_state.json"


def _save_salience_state(state):
    SALIENCE_STATE_PATH.write_text(json.dumps(state, default=str))


def _load_salience_state():
    if SALIENCE_STATE_PATH.exists():
        try:
            return json.loads(SALIENCE_STATE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"running": False, "error": None, "result": None}


def _save_dl_state(state):
    DATALOG_DL_STATE_PATH.write_text(json.dumps(state, default=str))


def _load_dl_state():
    if DATALOG_DL_STATE_PATH.exists():
        try:
            return json.loads(DATALOG_DL_STATE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"running": False, "pct": 0, "message": "", "done": False, "error": None}


@app.route("/api/datalog/status", methods=["GET"])
def api_datalog_status():
    from mantis_model_iteration_tool.salience_estimator import datalog_exists
    dl = _load_dl_state()
    dl["datalog"] = datalog_exists()
    return jsonify(dl)


@app.route("/api/datalog/download", methods=["POST"])
def api_datalog_download():
    dl = _load_dl_state()
    if dl.get("running"):
        return jsonify({"status": "already_running"}), 409

    def _run():
        from mantis_model_iteration_tool.salience_estimator import download_datalog
        _save_dl_state({"running": True, "pct": 0, "message": "Starting...",
                        "done": False, "error": None})
        try:
            download_datalog(progress_cb=lambda s: _save_dl_state({
                "running": not s.get("done", False),
                "pct": s.get("pct", 0),
                "message": s.get("message", ""),
                "done": s.get("done", False),
                "error": s.get("error"),
            }))
        except Exception as exc:
            _save_dl_state({"running": False, "pct": 0,
                            "message": f"Failed: {exc}",
                            "done": True, "error": str(exc)[:500]})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/agents/<agent_id>/estimate-salience", methods=["POST"])
def api_estimate_salience(agent_id):
    _validate_agent_id(agent_id)
    state = _load_salience_state()
    if state.get("running"):
        return jsonify({"status": "already_running"}), 409

    agent = _get_agent(agent_id)
    if not agent and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}")
        if resp and resp.status_code == 200:
            agent = resp.json()
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    challenge = (agent.get("config") or {}).get("challenge", "")
    if not challenge:
        return jsonify({"error": "Agent has no challenge configured"}), 400

    from mantis_model_iteration_tool.evaluator import CHALLENGES
    cfg = CHALLENGES.get(challenge)
    ctype = cfg.challenge_type if cfg else "binary"
    minfo = _get_metric_info(ctype)
    metric_key = minfo["primary"]
    higher_is_better = minfo["direction"] == "higher"

    iters = agent.get("iterations", [])
    best_iter = None
    best_score = None
    for it in iters:
        if it.get("has_error"):
            continue
        score = (it.get("metrics") or {}).get(metric_key)
        if score is None:
            continue
        if best_score is None or (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
            best_score = score
            best_iter = it
    if best_iter is None:
        return jsonify({"error": "No successful iteration found"}), 400

    iter_num = best_iter.get("iteration", 0)

    strategy_path = _resolve_strategy_file(agent_id, iter_num)
    if not strategy_path:
        return jsonify({"error": f"Strategy file not found: iteration_{iter_num}.py. "
                        "Checked workspace, artifacts/models/, and Targon."}), 400

    from mantis_model_iteration_tool.salience_estimator import datalog_exists
    dl_status = datalog_exists()
    if not dl_status["exists"]:
        return jsonify({"error": "Datalog not downloaded yet. Download it first (~30 GB)."}), 400

    data = _safe_json_body()
    days_back = max(60, min(365, _safe_int(data.get("days_back"), 60)))
    holdout_days = max(5, min(60, _safe_int(data.get("holdout_days"), 20)))
    cg_key = _get_coinglass_key()

    _save_salience_state({"running": True, "agent_id": agent_id,
                          "challenge": challenge, "error": None,
                          "result": None, "progress": "Starting..."})

    def _run():
        from mantis_model_iteration_tool.salience_estimator import estimate_salience
        import dataclasses

        def _progress(s):
            _save_salience_state({
                "running": True, "agent_id": agent_id,
                "challenge": challenge, "error": None, "result": None,
                "progress": s.get("message", ""),
                "pct": s.get("pct", 0),
            })

        try:
            result = estimate_salience(
                strategy_path=strategy_path,
                challenge_name=challenge,
                days_back=days_back,
                holdout_days=holdout_days,
                coinglass_key=cg_key,
                progress_cb=_progress,
            )
            _save_salience_state({
                "running": False, "agent_id": agent_id,
                "challenge": challenge,
                "error": result.error or None,
                "result": dataclasses.asdict(result),
                "progress": "Done",
                "pct": 100,
            })
        except Exception as exc:
            logger.exception("Salience estimation failed")
            _save_salience_state({
                "running": False, "agent_id": agent_id,
                "challenge": challenge,
                "error": str(exc)[:500],
                "result": None, "progress": "Failed",
            })

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "challenge": challenge,
                    "iteration": iter_num})


@app.route("/api/agents/<agent_id>/salience-status", methods=["GET"])
def api_salience_status(agent_id):
    _validate_agent_id(agent_id)
    state = _load_salience_state()
    if state.get("agent_id") != agent_id:
        return jsonify({"running": False, "result": None, "error": None})
    return jsonify(state)


# ── Batch salience: multiple models × challenges ─────────────────────────

BATCH_SALIENCE_STATE_PATH = Path(__file__).parent / ".batch_salience_state.json"


def _save_batch_salience_state(state):
    BATCH_SALIENCE_STATE_PATH.write_text(json.dumps(state, default=str))


def _load_batch_salience_state():
    if BATCH_SALIENCE_STATE_PATH.exists():
        try:
            return json.loads(BATCH_SALIENCE_STATE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"running": False, "jobs": [], "results": []}


def _resolve_strategy_file(agent_id: str, iteration: int) -> str | None:
    """Find an iteration .py file across workspace, artifacts, and Targon."""
    fname = f"iteration_{iteration}.py"

    ws_path = AGENTS_DIR / agent_id / "workspace" / fname
    if ws_path.exists():
        return str(ws_path)

    art_path = ARTIFACTS_MODELS_DIR / agent_id / fname
    if art_path.exists():
        return str(art_path)

    alt_path = AGENTS_DIR / agent_id / fname
    if alt_path.exists():
        return str(alt_path)

    if _targon_is_active():
        resp, err = _targon_get(f"/api/agents/{agent_id}/code/{iteration}")
        if resp and resp.status_code == 200 and resp.text.strip():
            local_ws = AGENTS_DIR / agent_id / "workspace"
            local_ws.mkdir(parents=True, exist_ok=True)
            dest = local_ws / fname
            dest.write_text(resp.text)
            logger.info("Fetched strategy from Targon → %s", dest)
            return str(dest)
        elif err:
            logger.warning("Targon fetch failed for %s/%d: %s", agent_id, iteration, err)

    return None


# ── Eval-Inference Parity Diagnostic ─────────────────────────────────────

PARITY_STATE_PATH = Path(__file__).parent / ".parity_diag_state.json"


def _save_parity_state(state):
    PARITY_STATE_PATH.write_text(json.dumps(state, default=str))


def _load_parity_state():
    if PARITY_STATE_PATH.exists():
        try:
            return json.loads(PARITY_STATE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"running": False, "agent_id": None}


def _run_parity_diagnostic(agent_id, strategy_path, challenge_name, iteration):
    """Background worker: loads the strategy, builds a DataProvider,
    runs both eval and inference paths, and compares everything."""
    import numpy as np
    import traceback as _tb
    try:
        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "loading_strategy"})

        from mantis_model_iteration_tool.inferencer import load_strategy, run_inference_single, ModelSlot
        from mantis_model_iteration_tool.evaluator import CHALLENGES, _generate_embeddings
        from mantis_model_iteration_tool.data import DataProvider, fetch_assets

        config = CHALLENGES[challenge_name]
        checks = []

        def check(name, passed, detail=""):
            checks.append({"name": name, "passed": passed, "detail": detail})

        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "loading_strategy"})
        try:
            feat_eval, pred_eval = load_strategy(strategy_path)
            feat_inf, pred_inf = load_strategy(strategy_path)
        except Exception as e:
            check("strategy_load", False, f"Failed to load strategy: {e}")
            _save_parity_state({"running": False, "agent_id": agent_id,
                                "iteration": iteration, "verdict": "red",
                                "checks": checks, "error": str(e)})
            return

        check("strategy_load", True, "TechFeaturizer and TechPredictor loaded successfully")

        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "fetching_data"})
        try:
            data, _ = fetch_assets(
                assets=config.assets, interval="1m", days_back=90)
            provider = DataProvider(data)
        except Exception as e:
            check("data_fetch", False, f"Failed to build DataProvider: {e}")
            _save_parity_state({"running": False, "agent_id": agent_id,
                                "iteration": iteration, "verdict": "red",
                                "checks": checks, "error": str(e)})
            return

        check("data_fetch", True,
              f"{len(data)} assets, {provider.length} candles")

        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "running_eval_path"})

        # --- instrument featurizer/predictor for tracing ---
        eval_traces = []
        inf_traces = []
        eval_features = []
        inf_features = []
        eval_outputs = []
        inf_outputs = []

        orig_eval_compute = feat_eval.compute
        orig_inf_compute = feat_inf.compute

        def _wrap_compute(orig_fn, trace_list):
            def wrapped(view):
                trace = {"t": view.t, "assets": view.assets}
                for a in view.assets:
                    p = view.prices(a)
                    trace[f"prices_{a}_len"] = len(p)
                    trace[f"prices_{a}_sum"] = float(np.sum(p))
                    o = view.ohlcv(a)
                    trace[f"ohlcv_{a}_sum"] = float(np.sum(o))
                    if view.has_cg():
                        for field in view.cg_fields(a):
                            cg_arr = view.cg(a, field)
                            trace[f"cg_{a}_{field}_len"] = len(cg_arr)
                            trace[f"cg_{a}_{field}_sum"] = float(np.sum(cg_arr))
                trace_list.append(trace)
                return orig_fn(view)
            return wrapped

        feat_eval.compute = _wrap_compute(orig_eval_compute, eval_traces)
        feat_inf.compute = _wrap_compute(orig_inf_compute, inf_traces)

        orig_eval_predict = pred_eval.predict
        orig_inf_predict = pred_inf.predict

        def _wrap_predict(orig_fn, feat_list, out_list):
            def wrapped(features):
                snap = {}
                for k, v in features.items():
                    arr = np.asarray(v)
                    snap[k] = {"shape": list(arr.shape), "sum": float(np.sum(arr)),
                               "hash": hash(arr.tobytes())}
                feat_list.append(snap)
                result = orig_fn(features)
                out_list.append(np.asarray(result).ravel().copy())
                return result
            return wrapped

        pred_eval.predict = _wrap_predict(orig_eval_predict, eval_features, eval_outputs)
        pred_inf.predict = _wrap_predict(orig_inf_predict, inf_features, inf_outputs)

        try:
            emb_matrix, warmup = _generate_embeddings(
                feat_eval, pred_eval, provider)
        except Exception as e:
            check("eval_run", False, f"Eval crashed: {e}")
            _save_parity_state({"running": False, "agent_id": agent_id,
                                "iteration": iteration, "verdict": "red",
                                "checks": checks, "error": str(e)})
            return

        check("eval_run", True,
              f"{emb_matrix.shape[0]} embeddings, warmup={warmup}")

        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "running_inference_path"})

        slot = ModelSlot(
            agent_id=agent_id, iteration=iteration,
            challenge_name=challenge_name, strategy_path=strategy_path,
            featurizer=feat_inf, predictor=pred_inf, loaded=True,
        )

        try:
            inf_result = run_inference_single(slot, provider)
        except Exception as e:
            check("inference_run", False, f"Inference crashed: {e}")
            _save_parity_state({"running": False, "agent_id": agent_id,
                                "iteration": iteration, "verdict": "red",
                                "checks": checks, "error": str(e)})
            return

        check("inference_run", True,
              f"Output shape: {inf_result.shape}")

        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "comparing_results"})

        # --- Compare raw input data at final timestep ---
        if eval_traces and inf_traces:
            et = eval_traces[-1]
            it = inf_traces[-1]
            t_match = et.get("t") == it.get("t")
            check("input_t_match", t_match,
                  f"Eval t={et.get('t')}, Inference t={it.get('t')}")

            mismatches = []
            all_keys = set(list(et.keys()) + list(it.keys()))
            for key in sorted(all_keys):
                ev = et.get(key)
                iv = it.get(key)
                if ev is None or iv is None:
                    mismatches.append(f"{key}: {'missing in eval' if ev is None else 'missing in inference'}")
                elif isinstance(ev, float) and isinstance(iv, float):
                    if abs(ev - iv) > 1e-10:
                        mismatches.append(f"{key}: eval={ev:.6g} inf={iv:.6g}")
                elif ev != iv:
                    mismatches.append(f"{key}: eval={ev!r} inf={iv!r}")
            check("input_data_identical", len(mismatches) == 0,
                  f"{len(mismatches)} mismatches" + (": " + "; ".join(mismatches[:5]) if mismatches else ""))
        else:
            check("input_data_identical", False,
                  f"No traces captured (eval={len(eval_traces)}, inf={len(inf_traces)})")

        # --- Compare feature dicts at final timestep ---
        if eval_features and inf_features:
            ef = eval_features[-1]
            ief = inf_features[-1]
            keys_match = set(ef.keys()) == set(ief.keys())
            check("feature_keys_match", keys_match,
                  f"Eval keys={sorted(ef.keys())}, Inference keys={sorted(ief.keys())}")

            feat_mismatches = []
            for k in sorted(set(list(ef.keys()) + list(ief.keys()))):
                e_info = ef.get(k, {})
                i_info = ief.get(k, {})
                if e_info.get("shape") != i_info.get("shape"):
                    feat_mismatches.append(f"{k}: shape eval={e_info.get('shape')} inf={i_info.get('shape')}")
                elif e_info.get("hash") != i_info.get("hash"):
                    feat_mismatches.append(f"{k}: values differ (sum eval={e_info.get('sum',0):.6g} inf={i_info.get('sum',0):.6g})")
            check("feature_values_identical", len(feat_mismatches) == 0,
                  f"{len(feat_mismatches)} feature mismatches" + (": " + "; ".join(feat_mismatches[:5]) if feat_mismatches else ""))
        else:
            check("feature_values_identical", False,
                  f"No feature traces (eval={len(eval_features)}, inf={len(inf_features)})")

        # --- Compare final embeddings ---
        eval_final = emb_matrix[-1] if emb_matrix.shape[0] > 0 else np.array([])
        emb_match = np.allclose(eval_final, inf_result, atol=1e-7)
        max_diff = float(np.max(np.abs(eval_final - inf_result))) if eval_final.shape == inf_result.shape else float('inf')
        check("embedding_match", emb_match,
              f"Max absolute diff: {max_diff:.2e}")

        byte_match = np.array_equal(
            np.asarray(eval_final, dtype=np.float32),
            np.asarray(inf_result, dtype=np.float32))
        check("embedding_byte_identical", byte_match,
              "float32 byte-level identity" if byte_match else f"Byte mismatch, max diff={max_diff:.2e}")

        # --- Determinism: run eval twice ---
        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "testing_determinism"})
        try:
            feat2, pred2 = load_strategy(strategy_path)
            emb2, _ = _generate_embeddings(feat2, pred2, provider)
            det_match = np.array_equal(emb_matrix, emb2)
            check("eval_deterministic", det_match,
                  "Two eval runs produce identical embeddings" if det_match
                  else f"Embeddings differ: max delta={float(np.max(np.abs(emb_matrix - emb2))):.2e}")
        except Exception as e:
            check("eval_deterministic", False, f"Second eval failed: {e}")

        # --- CausalView leakage guard ---
        _save_parity_state({"running": True, "agent_id": agent_id,
                            "iteration": iteration, "status": "testing_leakage_guards"})
        t_mid = provider.length // 2
        view = provider.view(t_mid)
        p = view.prices(view.assets[0])
        check("causal_view_truncation", len(p) == t_mid + 1,
              f"At t={t_mid}, prices length={len(p)}, expected {t_mid+1}")

        try:
            view.new_attr = 42
            check("causal_view_immutable", False, "setattr succeeded (should be blocked)")
        except AttributeError:
            check("causal_view_immutable", True, "setattr correctly blocked")

        p1 = view.prices(view.assets[0])
        p2 = view.prices(view.assets[0])
        p1[0] = -999.0
        check("causal_view_copy_isolation", p2[0] != -999.0,
              "Mutations on returned array don't leak" if p2[0] != -999.0
              else "LEAK: mutation propagated between copies")

        # --- Verdict ---
        n_passed = sum(1 for c in checks if c["passed"])
        n_total = len(checks)
        all_critical = all(c["passed"] for c in checks
                           if c["name"] in ("embedding_match", "input_data_identical",
                                            "feature_values_identical", "eval_run",
                                            "inference_run", "strategy_load"))
        if n_passed == n_total:
            verdict = "green"
        elif all_critical:
            verdict = "yellow"
        else:
            verdict = "red"

        _save_parity_state({
            "running": False, "agent_id": agent_id, "iteration": iteration,
            "verdict": verdict,
            "summary": f"{n_passed}/{n_total} checks passed",
            "checks": checks, "error": None,
        })
    except Exception as e:
        _save_parity_state({
            "running": False, "agent_id": agent_id,
            "iteration": iteration if 'iteration' in dir() else 0,
            "verdict": "red",
            "checks": checks if 'checks' in dir() else [],
            "error": f"Unexpected error: {_tb.format_exc()}",
        })


@app.route("/api/agents/<agent_id>/diagnose-parity", methods=["POST"])
def api_diagnose_parity(agent_id):
    """Run a live eval-inference parity diagnostic on the agent's best iteration."""
    _validate_agent_id(agent_id)

    state = _load_parity_state()
    if state.get("running"):
        return jsonify({"status": "already_running",
                        "agent_id": state.get("agent_id")}), 409

    agent = _get_agent(agent_id)
    if not agent and _targon_is_active():
        resp, _ = _targon_get(f"/api/agents/{agent_id}")
        if resp and resp.status_code == 200:
            agent = resp.json()
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    challenge = (agent.get("config") or {}).get("challenge", "")
    if not challenge:
        return jsonify({"error": "Agent has no challenge configured"}), 400

    from mantis_model_iteration_tool.evaluator import CHALLENGES
    cfg = CHALLENGES.get(challenge)
    if not cfg:
        return jsonify({"error": f"Unknown challenge: {challenge}"}), 400

    ctype = cfg.challenge_type
    minfo = _get_metric_info(ctype)
    metric_key = minfo["primary"]
    higher_is_better = minfo["direction"] == "higher"

    data = _safe_json_body()
    req_iter = data.get("iteration")

    iters = agent.get("iterations", [])
    target_iter = None
    if req_iter is not None:
        target_iter = next((it for it in iters
                           if it.get("iteration") == int(req_iter)), None)
    else:
        best_score = None
        for it in iters:
            if it.get("has_error"):
                continue
            score = (it.get("metrics") or {}).get(metric_key)
            if score is None:
                continue
            if best_score is None or (higher_is_better and score > best_score) or \
               (not higher_is_better and score < best_score):
                best_score = score
                target_iter = it

    if target_iter is None:
        return jsonify({"error": "No successful iteration found to diagnose"}), 400

    iter_num = target_iter.get("iteration", 0)
    strategy_path = _resolve_strategy_file(agent_id, iter_num)
    if not strategy_path:
        return jsonify({"error": f"Strategy file not found: iteration_{iter_num}.py"}), 400

    _save_parity_state({"running": True, "agent_id": agent_id,
                        "iteration": iter_num, "status": "starting"})
    t = threading.Thread(
        target=_run_parity_diagnostic,
        args=(agent_id, strategy_path, challenge, iter_num),
        daemon=True)
    t.start()
    return jsonify({"status": "started", "iteration": iter_num})


@app.route("/api/agents/<agent_id>/parity-status", methods=["GET"])
def api_parity_status(agent_id):
    _validate_agent_id(agent_id)
    state = _load_parity_state()
    if state.get("agent_id") != agent_id:
        return jsonify({"running": False, "verdict": None, "checks": []})
    return jsonify(state)


# ── Strategy Sandbox ──────────────────────────────────────────────────────

SANDBOX_STATE_PATH = Path(__file__).parent / ".sandbox_state.json"


def _save_sandbox_state(state):
    SANDBOX_STATE_PATH.write_text(json.dumps(state, default=str))


def _load_sandbox_state():
    if SANDBOX_STATE_PATH.exists():
        try:
            return json.loads(SANDBOX_STATE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"running": False}


def _run_sandbox_eval(code, challenge_name, days_back):
    import traceback as _tb
    import tempfile
    try:
        _save_sandbox_state({"running": True, "status": "loading_strategy"})

        from mantis_model_iteration_tool.inferencer import load_strategy
        from mantis_model_iteration_tool.evaluator import evaluate, CHALLENGES

        if challenge_name not in CHALLENGES:
            _save_sandbox_state({"running": False, "error": f"Unknown challenge: {challenge_name}"})
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=str(Path(__file__).parent)) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        _save_sandbox_state({"running": True, "status": "loading_strategy"})
        try:
            feat, pred = load_strategy(tmp_path)
        except Exception as e:
            _save_sandbox_state({"running": False, "error": f"Strategy load failed: {e}"})
            return
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        _save_sandbox_state({"running": True, "status": "evaluating"})
        result = evaluate(challenge_name, feat, pred, days_back=days_back)

        _save_sandbox_state({"running": False, "result": result, "error": None,
                             "challenge": challenge_name})
    except Exception:
        _save_sandbox_state({"running": False, "error": _tb.format_exc()})


@app.route("/api/sandbox/run", methods=["POST"])
def api_sandbox_run():
    """Run a strategy code snippet through evaluation."""
    state = _load_sandbox_state()
    if state.get("running"):
        return jsonify({"status": "already_running"}), 409

    data = _safe_json_body()
    code = str(data.get("code", ""))
    challenge = str(data.get("challenge", ""))[:64]
    days_back = max(30, min(365, _safe_int(data.get("days_back"), 60)))

    if not code.strip():
        return jsonify({"error": "No code provided"}), 400
    if not challenge:
        return jsonify({"error": "No challenge specified"}), 400

    _save_sandbox_state({"running": True, "status": "starting"})
    t = threading.Thread(
        target=_run_sandbox_eval,
        args=(code, challenge, days_back),
        daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/sandbox/status", methods=["GET"])
def api_sandbox_status():
    return jsonify(_load_sandbox_state())


@app.route("/api/salience/batch", methods=["POST"])
def api_batch_salience():
    """Run salience estimation for multiple model×challenge combos.

    Body: {
      "jobs": [
        {"agent_id": "abc123", "iteration": 4, "challenge": "ETH-1H-BINARY"},
        {"agent_id": "abc123", "iteration": 4, "challenge": "ETH-LBFGS"},
        ...
      ],
      "days_back": 60,    // optional, default 60
      "holdout_days": 20  // optional, default 20
    }
    """
    bs = _load_batch_salience_state()
    if bs.get("running"):
        return jsonify({"error": "Batch salience already running", "state": bs}), 409

    data = _safe_json_body()
    jobs = data.get("jobs", [])
    if not jobs or not isinstance(jobs, list):
        return jsonify({"error": "Provide 'jobs' array with agent_id, iteration, challenge"}), 400

    days_back = max(60, min(365, _safe_int(data.get("days_back"), 60)))
    holdout_days = max(5, min(60, _safe_int(data.get("holdout_days"), 20)))
    cg_key = _get_coinglass_key()

    from mantis_model_iteration_tool.salience_estimator import datalog_exists
    dl_status = datalog_exists()
    if not dl_status["exists"]:
        return jsonify({"error": "Datalog not downloaded yet. Download it first (~30 GB)."}), 400

    validated = []
    for j in jobs:
        aid = str(j.get("agent_id", "")).strip()
        if not _SAFE_ID.match(aid):
            return jsonify({"error": f"Invalid agent_id: {aid}"}), 400
        itr = _safe_int(j.get("iteration"), -1)
        ch = str(j.get("challenge", "")).strip()
        if not aid or itr < 0 or not ch:
            return jsonify({"error": f"Invalid job entry: {j}"}), 400
        path = _resolve_strategy_file(aid, itr)
        if not path:
            return jsonify({"error": f"Strategy not found: agent={aid} iter={itr}. "
                            "Checked workspace, artifacts/models/, and Targon."}), 400
        validated.append({"agent_id": aid, "iteration": itr, "challenge": ch,
                          "strategy_path": path})

    _save_batch_salience_state({
        "running": True,
        "total": len(validated),
        "completed": 0,
        "current_job": None,
        "jobs": [{"agent_id": v["agent_id"], "iteration": v["iteration"],
                  "challenge": v["challenge"]} for v in validated],
        "results": [],
        "error": None,
    })

    def _run_batch():
        from mantis_model_iteration_tool.salience_estimator import estimate_salience
        import dataclasses

        results = []
        for idx, job in enumerate(validated):
            _save_batch_salience_state({
                "running": True,
                "total": len(validated),
                "completed": idx,
                "current_job": {"agent_id": job["agent_id"],
                                "iteration": job["iteration"],
                                "challenge": job["challenge"]},
                "jobs": [{"agent_id": v["agent_id"], "iteration": v["iteration"],
                          "challenge": v["challenge"]} for v in validated],
                "results": results,
                "error": None,
            })
            try:
                result = estimate_salience(
                    strategy_path=job["strategy_path"],
                    challenge_name=job["challenge"],
                    days_back=days_back,
                    holdout_days=holdout_days,
                    coinglass_key=cg_key,
                )
                entry = dataclasses.asdict(result)
                entry["agent_id"] = job["agent_id"]
                entry["iteration"] = job["iteration"]
                results.append(entry)
            except Exception as exc:
                logger.exception("Batch salience failed for %s/%s/%d",
                                 job["challenge"], job["agent_id"], job["iteration"])
                results.append({
                    "agent_id": job["agent_id"],
                    "iteration": job["iteration"],
                    "challenge_name": job["challenge"],
                    "error": str(exc)[:500],
                })

        _save_batch_salience_state({
            "running": False,
            "total": len(validated),
            "completed": len(validated),
            "current_job": None,
            "jobs": [{"agent_id": v["agent_id"], "iteration": v["iteration"],
                      "challenge": v["challenge"]} for v in validated],
            "results": results,
            "error": None,
        })

    t = threading.Thread(target=_run_batch, daemon=True)
    t.start()
    return jsonify({"status": "started", "total": len(validated)}), 201


@app.route("/api/salience/batch/status", methods=["GET"])
def api_batch_salience_status():
    return jsonify(_load_batch_salience_state())


@app.route("/api/salience/challenges", methods=["GET"])
def api_salience_challenges():
    """List challenges that support salience estimation."""
    from mantis_model_iteration_tool.salience_estimator import _CHALLENGE_TO_SUBNET
    return jsonify(list(_CHALLENGE_TO_SUBNET.keys()))


@app.route("/api/subnet/economics", methods=["GET"])
def api_subnet_economics():
    """Fetch live TAO price (CoinGecko) and SN123 emission data (on-chain).

    Emissions are in dTAO (the subnet's alpha token). Convert to TAO via
    alpha_price_tao and to USD via tao_usd.
    """
    import numpy as np

    result = {"tao_usd": None, "alpha_price_tao": None,
              "miner_emission_per_day_dtao": None,
              "miner_emission_per_day_tao": None,
              "tempo": 360, "burn_pct": 0.30,
              "young_threshold_blocks": 36000, "error": None}
    try:
        import requests as _rq
        r = _rq.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bittensor&vs_currencies=usd"
            "&include_24hr_change=true&include_market_cap=true",
            timeout=10,
        )
        if r.status_code == 200:
            cg = r.json().get("bittensor", {})
            result["tao_usd"] = cg.get("usd")
            result["tao_24h_change"] = cg.get("usd_24h_change")
            result["tao_market_cap"] = cg.get("usd_market_cap")
    except Exception as exc:
        logger.warning("CoinGecko fetch failed: %s", exc)

    try:
        import bittensor as bt
        sub = bt.Subtensor(network="finney")
        from mantis_model_iteration_tool.miner import NETUID
        alpha_price = sub.get_subnet_price(netuid=NETUID)
        alpha_tao = float(alpha_price)
        result["alpha_price_tao"] = alpha_tao
        result["dtao_per_tao"] = round(1.0 / alpha_tao, 1) if alpha_tao > 0 else None
        if result["tao_usd"]:
            result["alpha_price_usd"] = alpha_tao * result["tao_usd"]

        info = sub.get_subnet_info(netuid=NETUID)
        result["tempo"] = info.tempo

        mg = sub.metagraph(netuid=NETUID)
        emission = np.array(mg.emission)
        incentive = np.array(mg.incentive)
        dividends = np.array(mg.dividends)

        tempos_per_day = 86400.0 / (12.0 * info.tempo)

        miner_mask = incentive > 0
        val_only_mask = (dividends > 0) & ~miner_mask
        miner_emission_per_tempo = float(emission[miner_mask].sum())
        miner_daily_dtao = round(miner_emission_per_tempo * tempos_per_day, 1)
        result["miner_emission_per_day_dtao"] = miner_daily_dtao
        result["miner_emission_per_day_tao"] = round(miner_daily_dtao * alpha_tao, 2)

        total_daily_dtao = float(emission.sum()) * tempos_per_day
        result["total_emission_per_day_dtao"] = round(total_daily_dtao, 1)

        result["total_uids"] = info.subnetwork_n
        result["n_miners"] = int(miner_mask.sum())
        result["n_validators"] = int(val_only_mask.sum())
    except Exception as exc:
        logger.warning("Bittensor chain fetch failed: %s", exc)
        result["error"] = str(exc)[:300]

    return jsonify(_sanitize_for_json(result))


@app.route("/api/miner/chain-performance", methods=["GET"])
def api_miner_chain_performance():
    """Fetch the miner's live on-chain performance: incentive, rank, emissions."""
    import numpy as np
    mcfg = _get_miner_config()
    hotkey_ss58 = mcfg.get("hotkey_ss58", "")
    if not hotkey_ss58:
        return jsonify({"error": "No hotkey registered. Register a miner first."}), 400

    result = {"hotkey": hotkey_ss58, "error": None}
    try:
        import bittensor as bt
        sub = bt.Subtensor(network="finney")
        from mantis_model_iteration_tool.miner import NETUID
        mg = sub.metagraph(netuid=NETUID)

        hotkeys = list(mg.hotkeys)
        if hotkey_ss58 not in hotkeys:
            result["registered"] = False
            result["error"] = "Hotkey not found in metagraph. Not registered or deregistered."
            return jsonify(result)

        uid = hotkeys.index(hotkey_ss58)
        result["registered"] = True
        result["uid"] = uid

        incentive = np.array(mg.incentive)
        emission = np.array(mg.emission)
        stake = np.array(mg.stake)

        result["incentive"] = float(incentive[uid])
        result["emission_dtao"] = float(emission[uid])

        miner_mask = incentive > 0
        n_miners = int(miner_mask.sum())
        result["n_miners"] = n_miners

        if incentive[uid] > 0:
            rank = int((incentive > incentive[uid]).sum()) + 1
            result["rank"] = rank
            result["rank_pct"] = round(rank / n_miners * 100, 1) if n_miners > 0 else None
        else:
            result["rank"] = n_miners
            result["rank_pct"] = 100.0

        info = sub.get_subnet_info(netuid=NETUID)
        tempos_per_day = 86400.0 / (12.0 * info.tempo)
        result["daily_emission_dtao"] = round(float(emission[uid]) * tempos_per_day, 2)

        alpha_price = sub.get_subnet_price(netuid=NETUID)
        alpha_tao = float(alpha_price)
        result["alpha_price_tao"] = alpha_tao
        if alpha_tao > 0:
            result["daily_emission_tao"] = round(result["daily_emission_dtao"] * alpha_tao, 4)

        try:
            import requests as _rq
            r = _rq.get(
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bittensor&vs_currencies=usd", timeout=10)
            if r.status_code == 200:
                tao_usd = r.json().get("bittensor", {}).get("usd")
                if tao_usd and result.get("daily_emission_tao"):
                    result["daily_emission_usd"] = round(result["daily_emission_tao"] * tao_usd, 2)
                    result["tao_usd"] = tao_usd
        except Exception:
            pass

        result["stake"] = float(stake[uid])
        result["updated_at"] = datetime.now().isoformat()

    except Exception as exc:
        logger.warning("Chain performance fetch failed: %s", exc)
        result["error"] = str(exc)[:300]

    return jsonify(_sanitize_for_json(result))


# ── Miner config helpers ─────────────────────────────────────────────────────

def _get_miner_config():
    if MINER_CONFIG_PATH.exists():
        raw = MINER_CONFIG_PATH.read_text().strip()
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
    return {}


def _save_miner_config(cfg):
    MINER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    MINER_CONFIG_PATH.chmod(0o600)


def _list_minable_models():
    """List agent iterations that have strategy files and successful eval results.

    Checks both local workspace and artifacts directories for strategy files,
    and fetches metadata from Targon for remote-only agents.
    """
    _META_KEYS = {
        "windows", "feature_stats", "feature_analysis",
        "dev_length", "eval_holdout_minutes", "n_timesteps",
        "n_evaluated", "n_valid", "n_usable", "n_assets",
        "n_events", "n_scored", "n_fallback", "horizon",
        "challenge", "type",
    }
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    seen = set()

    agent_states = {}
    for d in sorted(AGENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        state = _safe_json_load(d / "log.json")
        if state:
            agent_states[d.name] = state

    if _targon_is_active():
        resp, err = _targon_get("/api/agents", params={"full": "1"})
        if not err and resp and resp.status_code == 200:
            try:
                for ra in resp.json():
                    aid = ra.get("id", "")
                    if aid and aid not in agent_states:
                        agent_states[aid] = ra
            except (ValueError, Exception):
                pass

    for agent_id, state in agent_states.items():
        config = state.get("config", {})
        challenge = config.get("challenge", "")
        ws = AGENTS_DIR / agent_id / "workspace"
        art = ARTIFACTS_MODELS_DIR / agent_id

        for it in state.get("iterations", []):
            idx = it.get("iteration", 0)
            if idx < 1 or it.get("has_error"):
                continue
            metrics = it.get("metrics", {})
            if "error" in metrics:
                continue
            fname = f"iteration_{idx}.py"
            strat_path = ws / fname
            if not strat_path.exists():
                strat_path = art / fname
            if not strat_path.exists() and _targon_is_active():
                resp, _ = _targon_get(
                    f"/api/agents/{agent_id}/code/{idx}")
                if resp and resp.status_code == 200 and resp.headers.get("content-type", "").startswith("text/"):
                    art.mkdir(parents=True, exist_ok=True)
                    strat_path = art / fname
                    strat_path.write_text(resp.text)
                    logger.info("Downloaded %s/%s from Targon", agent_id, fname)
            if not strat_path.exists():
                continue
            key = (agent_id, idx)
            if key in seen:
                continue
            seen.add(key)
            models.append({
                "agent_id": agent_id,
                "iteration": idx,
                "challenge": challenge,
                "goal": config.get("goal", "")[:100],
                "metrics": {k: v for k, v in metrics.items()
                            if k not in _META_KEYS
                            and isinstance(v, (int, float))},
                "host_path": str(strat_path),
            })
    return models


# ── Miner process lifecycle tracking ──────────────────────────────────────────

_miner_lifecycle = {
    "started_at": 0,
    "stopped_at": 0,
    "was_alive": False,
    "last_exit_reason": "",
    "last_stdout_snapshot": "",
}

_MINER_LIFECYCLE_PATH = MINER_DIR / ".miner_lifecycle.json"


def _persist_miner_lifecycle():
    MINER_DIR.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in _miner_lifecycle.items()
            if k != "last_stdout_snapshot"}
    try:
        _MINER_LIFECYCLE_PATH.write_text(json.dumps(save))
    except OSError:
        pass


def _load_miner_lifecycle():
    if _MINER_LIFECYCLE_PATH.exists():
        try:
            data = json.loads(_MINER_LIFECYCLE_PATH.read_text())
            _miner_lifecycle.update(data)
        except (OSError, json.JSONDecodeError):
            pass


_load_miner_lifecycle()


def _fetch_targon_stdout_snapshot():
    """Grab stdout from Targon for crash snapshot."""
    if not _targon_is_active():
        return ""
    try:
        import requests as _req
        tcfg = _get_targon_config()
        url = tcfg["server_url"].rstrip("/") + "/api/miner/stdout"
        headers = {}
        if tcfg.get("server_auth_key"):
            headers["Authorization"] = f"Bearer {tcfg['server_auth_key']}"
        resp = _req.get(url, headers=headers, timeout=10)
        return resp.json().get("content", "")[:10000]
    except Exception:
        return ""


# ── Miner API routes ─────────────────────────────────────────────────────────

@app.route("/api/miner/models", methods=["GET"])
def api_miner_models():
    return jsonify(_list_minable_models())


@app.route("/api/miner/config", methods=["GET"])
def api_miner_config():
    cfg = _get_miner_config()
    safe = {k: v for k, v in cfg.items() if k not in (
        "hotkey_mnemonic", "coldkey_mnemonic",
        "r2_secret_access_key",
    )}
    for secret_key in ("hotkey_mnemonic", "r2_secret_access_key"):
        if cfg.get(secret_key):
            safe[secret_key + "_set"] = True
    return jsonify(safe)


@app.route("/api/miner/config", methods=["POST"])
def api_miner_config_save():
    data = _safe_json_body()
    cfg = _get_miner_config()
    allowed = {
        "hotkey_mnemonic", "hotkey_ss58", "network",
        "r2_account_id", "r2_access_key_id", "r2_secret_access_key",
        "r2_bucket_name", "r2_public_base_url",
        "interval_seconds", "lock_seconds", "lookback",
        "selected_models",
    }
    for k in allowed:
        if k in data:
            cfg[k] = data[k]
    _save_miner_config(cfg)
    return jsonify({"status": "saved"})


@app.route("/api/miner/check-registration", methods=["POST"])
def api_miner_check_reg():
    data = _safe_json_body()
    hotkey_ss58 = str(data.get("hotkey_ss58", "")).strip()
    hotkey_mnemonic = str(data.get("hotkey_mnemonic", "")).strip()
    network = str(data.get("network", "finney")).strip()

    if _targon_is_active():
        resp, err = _targon_post("/api/miner/check-registration", json_body={
            "hotkey_ss58": hotkey_ss58,
            "hotkey_mnemonic": hotkey_mnemonic,
            "network": network,
        })
        if err:
            return jsonify({"error": err[:300]}), 502
        return jsonify(resp.json()), resp.status_code

    err_resp = _ensure_miner_image_or_500()
    if err_resp:
        return err_resp

    result = run_miner_check_registration(
        pkg_dir=_PKG_DIR_STR,
        hotkey_ss58=hotkey_ss58,
        hotkey_mnemonic=hotkey_mnemonic,
        network=network,
    )
    return jsonify(result)


@app.route("/api/miner/register", methods=["POST"])
def api_miner_register():
    data = _safe_json_body()
    coldkey_mnemonic = str(data.get("coldkey_mnemonic", "")).strip()
    hotkey_mnemonic = str(data.get("hotkey_mnemonic", "")).strip()
    network = str(data.get("network", "finney")).strip()

    if not coldkey_mnemonic or not hotkey_mnemonic:
        return jsonify({"error": "Both coldkey and hotkey mnemonics are required"}), 400

    if _targon_is_active():
        resp, err = _targon_post("/api/miner/register", json_body={
            "coldkey_mnemonic": coldkey_mnemonic,
            "hotkey_mnemonic": hotkey_mnemonic,
            "network": network,
        }, timeout=120)
        if err:
            return jsonify({"error": err[:300]}), 502
        return jsonify(resp.json()), resp.status_code

    err_resp = _ensure_miner_image_or_500()
    if err_resp:
        return err_resp

    result = run_miner_registration(
        pkg_dir=_PKG_DIR_STR,
        coldkey_mnemonic=coldkey_mnemonic,
        hotkey_mnemonic=hotkey_mnemonic,
        network=network,
        miner_dir=str(MINER_DIR),
    )
    return jsonify(result)


@app.route("/api/miner/start", methods=["POST"])
def api_miner_start():
    data = _safe_json_body()
    cfg = _get_miner_config()
    cfg.update({k: v for k, v in data.items() if k in (
        "hotkey_mnemonic", "hotkey_ss58", "network",
        "r2_account_id", "r2_access_key_id", "r2_secret_access_key",
        "r2_bucket_name", "r2_public_base_url",
        "interval_seconds", "lock_seconds", "lookback",
        "selected_models",
    )})
    _save_miner_config(cfg)

    selected = cfg.get("selected_models", [])
    if not selected:
        return jsonify({"error": "No models selected for mining"}), 400

    hotkey_mnemonic = cfg.get("hotkey_mnemonic", "")
    hotkey_ss58 = cfg.get("hotkey_ss58", "")
    if not hotkey_mnemonic and not hotkey_ss58:
        return jsonify({"error": "Hotkey mnemonic or SS58 address required"}), 400

    r2_missing = []
    for f in ("r2_account_id", "r2_access_key_id", "r2_secret_access_key", "r2_bucket_name"):
        if not cfg.get(f):
            r2_missing.append(f)
    if r2_missing:
        return jsonify({"error": f"Missing R2 config: {', '.join(r2_missing)}"}), 400

    if _targon_is_active():
        dep_resp, dep_err = _targon_post("/api/setup/install-deps", timeout=900)
        if dep_err:
            logger.warning("install-deps call failed: %s", dep_err)
        elif dep_resp and dep_resp.status_code >= 400:
            logger.warning("install-deps returned %s", dep_resp.status_code)

        upload_errors = []
        for m in selected:
            aid, itr = m["agent_id"], m["iteration"]
            fname = f"iteration_{itr}.py"
            code = None
            for loc in [
                AGENTS_DIR / aid / "workspace" / fname,
                ARTIFACTS_MODELS_DIR / aid / fname,
            ]:
                if loc.exists():
                    code = loc.read_text(errors="replace")
                    break
            if not code:
                upload_errors.append(f"Strategy file not found locally: {aid} iter {itr}")
                continue
            ur, ur_err = _targon_post("/api/miner/upload-strategy",
                                      json_body={"agent_id": aid, "iteration": itr, "code": code},
                                      timeout=30)
            if ur_err:
                upload_errors.append(f"Upload error {aid} iter {itr}: {ur_err}")
            elif ur.status_code >= 400:
                upload_errors.append(f"Upload failed for {aid} iter {itr}: {ur.text[:200]}")
            else:
                logger.info("Uploaded strategy %s iter %d to Targon", aid, itr)
        if upload_errors:
            return jsonify({"error": "; ".join(upload_errors)}), 400

        model_specs = []
        for m in selected:
            model_specs.append(
                f"{m['agent_id']}:{m['iteration']}:{m['challenge']}:"
                f"/data/agents/{m['agent_id']}/workspace/iteration_{m['iteration']}.py"
            )
        resp, err = _targon_post("/api/miner/start", json_body={
            "models": model_specs,
            "interval_seconds": int(cfg.get("interval_seconds", 60)),
            "lock_seconds": min(int(cfg.get("lock_seconds", 30)), 120),
            "lookback": int(cfg.get("lookback", 5000)),
            "network": cfg.get("network", "finney"),
            "hotkey_mnemonic": hotkey_mnemonic,
            "hotkey_ss58": hotkey_ss58,
            "r2_account_id": cfg.get("r2_account_id", ""),
            "r2_access_key_id": cfg.get("r2_access_key_id", ""),
            "r2_secret_access_key": cfg.get("r2_secret_access_key", ""),
            "r2_bucket_name": cfg.get("r2_bucket_name", ""),
            "r2_public_base_url": cfg.get("r2_public_base_url", ""),
            "coinglass_key": _get_coinglass_key() or "",
        }, timeout=30)
        if err:
            return jsonify({"error": err[:300]}), 502
        rj = resp.json()
        if resp.status_code < 300:
            _miner_lifecycle["started_at"] = _time.time()
            _miner_lifecycle["stopped_at"] = 0
            _miner_lifecycle["was_alive"] = True
            _miner_lifecycle["last_exit_reason"] = ""
            _miner_lifecycle["last_stdout_snapshot"] = ""
            _persist_miner_lifecycle()
        return jsonify(rj), resp.status_code

    err_resp = _ensure_miner_image_or_500()
    if err_resp:
        return err_resp

    if miner_running():
        return jsonify({"error": "Miner already running"}), 409

    MINER_DIR.mkdir(parents=True, exist_ok=True)
    (MINER_DIR / "MINER_STOP").unlink(missing_ok=True)
    (MINER_DIR / "MINER_FAILED").unlink(missing_ok=True)

    env_vars = {
        "HOTKEY_MNEMONIC": hotkey_mnemonic,
        "HOTKEY_SS58": hotkey_ss58,
        "R2_ACCOUNT_ID": cfg.get("r2_account_id", ""),
        "R2_ACCESS_KEY_ID": cfg.get("r2_access_key_id", ""),
        "R2_SECRET_ACCESS_KEY": cfg.get("r2_secret_access_key", ""),
        "R2_BUCKET_NAME": cfg.get("r2_bucket_name", ""),
        "R2_PUBLIC_BASE_URL": cfg.get("r2_public_base_url", ""),
        "COINGLASS_API_KEY": _get_coinglass_key() or "",
    }

    model_specs = []
    agent_dirs = []
    for m in selected:
        aid = m["agent_id"]
        itr = m["iteration"]
        strategy_fname = f"iteration_{itr}.py"

        ws_host = AGENTS_DIR / aid / "workspace"
        artifact_host = ARTIFACTS_MODELS_DIR / aid

        if (ws_host / strategy_fname).exists():
            host_dir = str(ws_host)
            container_dir = f"/agents/{aid}/workspace"
        elif (artifact_host / strategy_fname).exists():
            host_dir = str(artifact_host)
            container_dir = f"/artifacts/{aid}"
        else:
            return jsonify({
                "error": f"Model file not found for agent {aid} iteration {itr}. "
                         f"Checked workspace and artifacts/models/.",
            }), 400

        agent_dirs.append((host_dir, container_dir))
        model_specs.append({
            "agent_id": aid,
            "iteration": itr,
            "challenge": m["challenge"],
            "container_path": f"{container_dir}/{strategy_fname}",
        })

    import uuid
    miner_id = uuid.uuid4().hex[:8]

    try:
        launch_miner_container(
            miner_id=miner_id,
            miner_dir=str(MINER_DIR),
            pkg_dir=_PKG_DIR_STR,
            model_specs=model_specs,
            env_vars=env_vars,
            agent_dirs=agent_dirs,
            interval_seconds=int(cfg.get("interval_seconds", 60)),
            lock_seconds=max(10, min(120, int(cfg.get("lock_seconds", 30)))),
            lookback=int(cfg.get("lookback", 5000)),
            network=cfg.get("network", "finney"),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)[:500]}), 500

    mcfg = _get_miner_config()
    mcfg["miner_id"] = miner_id
    _save_miner_config(mcfg)

    _miner_lifecycle["started_at"] = _time.time()
    _miner_lifecycle["stopped_at"] = 0
    _miner_lifecycle["was_alive"] = True
    _miner_lifecycle["last_exit_reason"] = ""
    _miner_lifecycle["last_stdout_snapshot"] = ""
    _persist_miner_lifecycle()

    return jsonify({"status": "started", "miner_id": miner_id}), 201


@app.route("/api/miner/stop", methods=["POST"])
def api_miner_stop():
    if _targon_is_active():
        resp, err = _targon_post("/api/miner/stop", timeout=15)
        if err:
            return jsonify({"error": err[:300]}), 502
        return jsonify(resp.json()), resp.status_code

    mcfg = _get_miner_config()
    miner_id = mcfg.get("miner_id", "")

    MINER_DIR.mkdir(parents=True, exist_ok=True)
    (MINER_DIR / "MINER_STOP").touch()

    if miner_id:
        kill_miner_container(miner_id)

    _miner_lifecycle["stopped_at"] = _time.time()
    _miner_lifecycle["was_alive"] = False
    _miner_lifecycle["last_exit_reason"] = "user_stopped"
    _persist_miner_lifecycle()

    return jsonify({"status": "stopped"})


@app.route("/api/miner/status", methods=["GET"])
def api_miner_status():
    _load_miner_lifecycle()
    lc = _miner_lifecycle

    if _targon_is_active():
        resp, err = _targon_get("/api/miner/status", timeout=10)
        if resp:
            try:
                data = resp.json()
            except Exception:
                data = {"status": {}, "failed": None, "process_alive": False,
                        "error": "Invalid JSON from Targon"}
        else:
            data = {"status": {}, "failed": None, "process_alive": False,
                    "error": (err or "")[:300]}

        alive = data.get("process_alive", False)
        if lc["was_alive"] and not alive:
            lc["stopped_at"] = _time.time()
            lc["was_alive"] = False
            lc["last_stdout_snapshot"] = _fetch_targon_stdout_snapshot()
            lc["last_exit_reason"] = "crashed"
            _persist_miner_lifecycle()
        elif alive:
            lc["was_alive"] = True

        data["lifecycle"] = {
            "started_at": lc["started_at"],
            "stopped_at": lc["stopped_at"],
            "last_exit_reason": lc["last_exit_reason"],
            "had_process": lc["started_at"] > 0,
        }
        if lc["last_stdout_snapshot"] and not alive:
            data["crash_stdout"] = lc["last_stdout_snapshot"]
        return jsonify(data)

    status_path = MINER_DIR / "miner_status.json"
    failed_path = MINER_DIR / "MINER_FAILED"
    status_data = _safe_json_load(status_path, {})
    failed_data = _safe_json_load(failed_path) if failed_path.exists() else None

    mcfg = _get_miner_config()
    miner_id = mcfg.get("miner_id", "")
    alive = miner_running(miner_id) if miner_id else miner_running()

    if status_data.get("running") and not alive:
        status_data["running"] = False
        status_data["crashed"] = True
        if lc["was_alive"]:
            lc["stopped_at"] = _time.time()
            lc["was_alive"] = False
            lc["last_exit_reason"] = "crashed"
            _persist_miner_lifecycle()

    if alive and not lc["was_alive"]:
        lc["was_alive"] = True

    return jsonify({
        "status": status_data,
        "failed": failed_data,
        "process_alive": alive,
        "lifecycle": {
            "started_at": lc["started_at"],
            "stopped_at": lc["stopped_at"],
            "last_exit_reason": lc["last_exit_reason"],
            "had_process": lc["started_at"] > 0,
        },
    })


@app.route("/api/miner/diagnose", methods=["POST"])
def api_miner_diagnose():
    """Spawn a lightweight agent pre-seeded with miner context to diagnose issues."""
    ant_key = _get_anthropic_key()
    if not ant_key:
        return jsonify({"error": "Anthropic API key not configured"}), 400

    status_path = MINER_DIR / "miner_status.json"
    failed_path = MINER_DIR / "MINER_FAILED"
    stdout_path = MINER_DIR / "stdout.log"

    context_parts = ["# Miner Diagnostic Context\n"]

    status_data = _safe_json_load(status_path, {})
    if status_data:
        context_parts.append(f"## Current Status\n```json\n{json.dumps(status_data, indent=2)}\n```\n")

    if failed_path.exists():
        failed_text = failed_path.read_text(errors="replace").strip()[:2000]
        context_parts.append(f"## MINER_FAILED\n```\n{failed_text}\n```\n")

    if stdout_path.exists():
        try:
            fsize = stdout_path.stat().st_size
            with open(stdout_path, "r", errors="replace") as sf:
                if fsize > 30000:
                    sf.seek(fsize - 30000)
                    sf.readline()
                log_tail = sf.read()
        except OSError:
            log_tail = ""
        if log_tail:
            context_parts.append(f"## Recent Log (last ~30KB)\n```\n{log_tail[-8000:]}\n```\n")

    mcfg = _get_miner_config()
    safe_cfg = {k: v for k, v in mcfg.items() if k not in (
        "hotkey_mnemonic", "coldkey_mnemonic", "r2_secret_access_key"
    )}
    if safe_cfg:
        context_parts.append(f"## Miner Config (secrets redacted)\n```json\n{json.dumps(safe_cfg, indent=2)}\n```\n")

    context = "\n".join(context_parts)
    data = _safe_json_body()
    user_question = str(data.get("question", "")).strip()[:5000]
    if not user_question:
        user_question = "Analyze the miner's current state. Is it healthy? Are there any errors or warnings? What should I do next?"

    system_prompt = (
        "You are a Bittensor mining diagnostician for the MANTIS subnet (SN123). "
        "You have access to the miner's status, logs, and config. "
        "Diagnose issues, explain errors, suggest fixes, and assess overall health. "
        "Be direct and specific. Reference log lines and status fields when relevant."
    )
    user_msg = f"{context}\n\n## Question\n\n{user_question}"

    response = _anthropic_chat(ant_key, system_prompt, user_msg, max_tokens=4096)

    return jsonify({"response": response})


@app.route("/api/miner/stdout", methods=["GET"])
def api_miner_stdout():
    if _targon_is_active():
        import requests as _req
        tcfg = _get_targon_config()
        tail = request.args.get("tail", "500")
        url = tcfg["server_url"].rstrip("/") + f"/api/miner/stdout?lines={tail}"
        headers = {}
        if tcfg.get("server_auth_key"):
            headers["Authorization"] = f"Bearer {tcfg['server_auth_key']}"
        try:
            resp = _req.get(url, headers=headers, timeout=10)
            return jsonify(resp.json())
        except Exception as exc:
            return jsonify({"content": "", "lines": 0, "error": str(exc)[:300]})

    stdout_path = MINER_DIR / "stdout.log"
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


# ── Local artifacts API ───────────────────────────────────────────────────

@app.route("/api/artifacts/models", methods=["GET"])
def api_artifacts_models():
    """List all locally archived models."""
    result = []
    if not ARTIFACTS_MODELS_DIR.exists():
        return jsonify(result)
    for agent_dir in sorted(ARTIFACTS_MODELS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        for py_file in sorted(agent_dir.glob("iteration_*.py")):
            name = py_file.stem
            itr_str = name.replace("iteration_", "")
            try:
                itr = int(itr_str)
            except ValueError:
                continue
            meta_path = agent_dir / f"iteration_{itr}_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            result.append({
                "agent_id": agent_dir.name,
                "iteration": itr,
                "challenge": meta.get("challenge", ""),
                "timestamp": meta.get("timestamp", ""),
                "metrics": meta.get("metrics", {}),
                "path": str(py_file),
            })
    return jsonify(result)


# ── Wallet backup ─────────────────────────────────────────────────────────

def _backup_wallet(wallet: dict) -> Path:
    """Persist wallet mnemonics to artifacts/wallets/ as a timestamped JSON file.

    Returns the path to the backup file.  The file is chmod 600 so only
    the owner can read it.
    """
    import datetime as _dt

    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    hotkey_short = wallet.get("hotkey_ss58", "unknown")[:8]
    fname = f"wallet_{ts}_{hotkey_short}.json"
    path = ARTIFACTS_WALLETS_DIR / fname

    payload = {
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "coldkey_ss58": wallet.get("coldkey_ss58", ""),
        "coldkey_mnemonic": wallet.get("coldkey_mnemonic", ""),
        "hotkey_ss58": wallet.get("hotkey_ss58", ""),
        "hotkey_mnemonic": wallet.get("hotkey_mnemonic", ""),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    path.chmod(0o600)
    logger.info("Wallet backed up to %s", path)
    return path


# ── Auto-setup: generate wallet, wait for deposit, register, launch ───────

_auto_setup_lock = threading.Lock()
_auto_setup_state: dict = {}
_AUTO_SETUP_STATE_PATH = MINER_DIR / ".auto_setup_state.json"


def _persist_auto_setup_state():
    """Save current auto-setup state to disk (call while holding lock)."""
    try:
        MINER_DIR.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: v for k, v in _auto_setup_state.items()
            if not callable(v) and k != "last_check"
        }
        _AUTO_SETUP_STATE_PATH.write_text(json.dumps(serializable, indent=2))
        _AUTO_SETUP_STATE_PATH.chmod(0o600)
    except Exception:
        pass


def _load_auto_setup_state():
    """Restore auto-setup state from disk on startup.

    If the saved phase was waiting_for_deposit, restart the polling thread.
    """
    if _AUTO_SETUP_STATE_PATH.exists():
        try:
            data = json.loads(_AUTO_SETUP_STATE_PATH.read_text())
            if isinstance(data, dict) and data.get("phase"):
                data.pop("cancelled", None)
                _auto_setup_state.update(data)
                logger.info("Restored auto-setup state: phase=%s", data.get("phase"))
                if data["phase"] in ("generated", "waiting_for_deposit"):
                    _auto_setup_state["cancelled"] = False
                    _auto_setup_state["_needs_resume"] = True
        except Exception:
            pass


def _get_auto_setup_state():
    with _auto_setup_lock:
        return dict(_auto_setup_state)


def _set_auto_setup_state(**kw):
    with _auto_setup_lock:
        _auto_setup_state.update(kw)
        _persist_auto_setup_state()


def _auto_setup_poll_loop():
    """Background thread: poll balance until funded, then register + start."""
    state = _get_auto_setup_state()
    coldkey_ss58 = state.get("coldkey_ss58", "")
    network = state.get("network", "finney")
    poll_interval = 15

    logger.info("Auto-setup: polling balance for %s every %ds", coldkey_ss58, poll_interval)
    _set_auto_setup_state(phase="waiting_for_deposit")

    while True:
        if _get_auto_setup_state().get("cancelled"):
            _set_auto_setup_state(phase="cancelled")
            return

        try:
            ensure_miner_image(_PKG_DIR_STR)
            result = run_miner_check_balance(
                pkg_dir=_PKG_DIR_STR,
                ss58_address=coldkey_ss58,
                network=network,
            )
            balance = result.get("balance", 0.0)
            burn_cost = result.get("burn_cost", 0.0)
            _set_auto_setup_state(
                balance=balance, burn_cost=burn_cost,
                last_check=_time.monotonic(),
            )

            if result.get("sufficient") or (balance > 0 and balance >= burn_cost + 0.001):
                logger.info("Auto-setup: deposit detected! balance=%s TAO", balance)
                _set_auto_setup_state(phase="registering")
                break
        except Exception as exc:
            logger.warning("Auto-setup balance poll error: %s", exc)

        _time.sleep(poll_interval)

    state = _get_auto_setup_state()
    if state.get("cancelled"):
        _set_auto_setup_state(phase="cancelled")
        return

    coldkey_mnemonic = state.get("coldkey_mnemonic", "")
    hotkey_mnemonic = state.get("hotkey_mnemonic", "")

    try:
        reg_result = run_miner_registration(
            pkg_dir=_PKG_DIR_STR,
            coldkey_mnemonic=coldkey_mnemonic,
            hotkey_mnemonic=hotkey_mnemonic,
            network=network,
            miner_dir=str(MINER_DIR),
        )
    except Exception as exc:
        _set_auto_setup_state(phase="failed", error=f"Registration failed: {exc}")
        return

    if not reg_result.get("success"):
        _set_auto_setup_state(
            phase="failed",
            error=reg_result.get("error", "Registration returned failure"),
        )
        return

    resolved_ss58 = reg_result.get("hotkey_ss58", state.get("hotkey_ss58", ""))
    _set_auto_setup_state(
        phase="registered",
        uid=reg_result.get("uid", -1),
        hotkey_ss58=resolved_ss58,
    )

    cfg = _get_miner_config()
    cfg["hotkey_mnemonic"] = hotkey_mnemonic
    cfg["hotkey_ss58"] = resolved_ss58
    cfg["network"] = network
    _save_miner_config(cfg)

    _set_auto_setup_state(phase="ready")
    with _auto_setup_lock:
        _auto_setup_state.pop("coldkey_mnemonic", None)
        _persist_auto_setup_state()
    logger.info("Auto-setup complete: registered uid=%s, hotkey saved to config",
                reg_result.get("uid"))


_load_auto_setup_state()
if _auto_setup_state.pop("_needs_resume", False):
    threading.Thread(target=_auto_setup_poll_loop, daemon=True).start()
    logger.info("Resumed auto-setup deposit polling from saved state")


@app.route("/api/miner/auto-setup", methods=["POST"])
def api_miner_auto_setup():
    """Generate a wallet and start polling for deposit."""
    current = _get_auto_setup_state()
    if current.get("phase") in ("waiting_for_deposit", "registering"):
        return jsonify({"error": "Auto-setup already in progress", "state": current}), 409

    network = str(_safe_json_body().get("network", "finney")).strip()

    err_resp = _ensure_miner_image_or_500()
    if err_resp:
        return err_resp

    wallet = run_miner_generate_wallet(_PKG_DIR_STR)
    if wallet.get("error"):
        return jsonify({"error": wallet["error"]}), 500

    _backup_wallet(wallet)

    MINER_DIR.mkdir(parents=True, exist_ok=True)

    with _auto_setup_lock:
        _auto_setup_state.clear()
        _auto_setup_state.update({
            "phase": "generated",
            "coldkey_ss58": wallet["coldkey_ss58"],
            "coldkey_mnemonic": wallet["coldkey_mnemonic"],
            "hotkey_ss58": wallet["hotkey_ss58"],
            "hotkey_mnemonic": wallet["hotkey_mnemonic"],
            "network": network,
            "balance": 0.0,
            "burn_cost": 0.0,
            "cancelled": False,
        })
        _persist_auto_setup_state()

    threading.Thread(target=_auto_setup_poll_loop, daemon=True).start()

    return jsonify({
        "phase": "generated",
        "coldkey_ss58": wallet["coldkey_ss58"],
        "hotkey_ss58": wallet["hotkey_ss58"],
        "network": network,
        "coldkey_mnemonic": wallet["coldkey_mnemonic"],
        "hotkey_mnemonic": wallet["hotkey_mnemonic"],
    }), 201


@app.route("/api/miner/auto-setup/status", methods=["GET"])
def api_miner_auto_setup_status():
    """Poll the auto-setup state."""
    state = _get_auto_setup_state()
    safe = dict(state)
    safe.pop("last_check", None)
    safe["has_keys"] = bool(state.get("coldkey_mnemonic"))
    return jsonify(safe)


@app.route("/api/miner/auto-setup/register", methods=["POST"])
def api_miner_auto_setup_register():
    """Immediately trigger registration (skip deposit polling)."""
    state = _get_auto_setup_state()
    if state.get("phase") in ("registered", "ready"):
        return jsonify({"status": "already_registered"})

    coldkey_mnemonic = state.get("coldkey_mnemonic", "")
    hotkey_mnemonic = state.get("hotkey_mnemonic", "")
    network = state.get("network", "finney")

    if not coldkey_mnemonic or not hotkey_mnemonic:
        return jsonify({"error": "Wallet keys not available — regenerate"}), 400

    _set_auto_setup_state(cancelled=True, phase="registering")

    def _do_register():
        try:
            ensure_miner_image(_PKG_DIR_STR)
            reg_result = run_miner_registration(
                pkg_dir=_PKG_DIR_STR,
                coldkey_mnemonic=coldkey_mnemonic,
                hotkey_mnemonic=hotkey_mnemonic,
                network=network,
                miner_dir=str(MINER_DIR),
            )
        except Exception as exc:
            _set_auto_setup_state(phase="failed", error=f"Registration failed: {exc}")
            return

        if not reg_result.get("success"):
            _set_auto_setup_state(
                phase="failed",
                error=reg_result.get("error", "Registration returned failure"),
            )
            return

        resolved_ss58 = reg_result.get("hotkey_ss58", state.get("hotkey_ss58", ""))
        _set_auto_setup_state(
            phase="registered",
            uid=reg_result.get("uid", -1),
            hotkey_ss58=resolved_ss58,
        )

        cfg = _get_miner_config()
        cfg["hotkey_mnemonic"] = hotkey_mnemonic
        cfg["hotkey_ss58"] = resolved_ss58
        cfg["network"] = network
        _save_miner_config(cfg)

        _set_auto_setup_state(phase="ready")
        with _auto_setup_lock:
            _auto_setup_state.pop("coldkey_mnemonic", None)
            _persist_auto_setup_state()
        logger.info("Manual register complete: uid=%s", reg_result.get("uid"))

    threading.Thread(target=_do_register, daemon=True).start()
    return jsonify({"status": "registering"})


@app.route("/api/miner/auto-setup/cancel", methods=["POST"])
def api_miner_auto_setup_cancel():
    _set_auto_setup_state(cancelled=True, phase="cancelled")
    if _AUTO_SETUP_STATE_PATH.exists():
        _AUTO_SETUP_STATE_PATH.unlink(missing_ok=True)
    return jsonify({"status": "cancelling"})


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


def main():
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
        print(f"    python -m mantis_model_iteration_tool.gui --host 127.0.0.1")
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
            "mantis_model_iteration_tool.gui:app",
        ]
        _gunicorn_run()
    else:
        logger.warning("gunicorn not installed -- using Flask dev server (not for production)")
        signal.signal(signal.SIGTERM, _graceful_shutdown)
        signal.signal(signal.SIGINT, _graceful_shutdown)
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
