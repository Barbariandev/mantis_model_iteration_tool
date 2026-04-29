"""Shared helpers used by gui.py, targon_server.py, and agent_runner.py."""

import fcntl
import json
import math
import os
import tempfile
from pathlib import Path


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for valid JSON output."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def safe_json_load(path, default=None):
    """Load JSON from a file path, returning *default* on any failure."""
    path = Path(path)
    if not path.exists():
        return default
    raw = path.read_text(errors="replace").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return default


def atomic_json_write(path, data):
    """Write JSON atomically via tempfile + rename."""
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


def locked_json_update(path, mutate_fn, default=None):
    """Read-modify-write a JSON file under an exclusive flock."""
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
            if raw:
                corrupt = path.parent / (path.name + ".corrupt")
                try:
                    corrupt.write_text(raw)
                except OSError:
                    pass
            state = _make_default(default)
        if not isinstance(state, (dict, list)):
            state = _make_default(default)
        mutate_fn(state)
        atomic_json_write(path, state)
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


VALID_STATUSES = frozenset({
    "running", "stopped", "crashed", "completed", "paused",
    "starting", "max_iterations_reached", "error",
})


def sanitize_status(status):
    """Normalise agent status to a known value or 'unknown'."""
    return status if status in VALID_STATUSES else "unknown"


def chat_append(chat_path, role, text, msg_type="message"):
    """Append a message to a chat.json file under exclusive flock."""
    from datetime import datetime
    chat_path = Path(chat_path)

    def _append(msgs):
        entry = {"role": role, "text": text, "ts": datetime.now().isoformat()}
        if msg_type != "message":
            entry["type"] = msg_type
        msgs.append(entry)

    locked_json_update(chat_path, _append, default=[])


def build_agent_context(agents_dir, agent_id):
    """Build a markdown context string for an agent (shared by gui + targon)."""
    agent_dir = Path(agents_dir) / agent_id
    ws = agent_dir / "workspace"
    parts = []
    state = safe_json_load(agent_dir / "log.json", {})
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
            section = f"## Iteration {it['iteration']} Metrics\n{json.dumps(sanitize_for_json(display_m), indent=2, default=str)}"
            an = it.get("analysis")
            if an and isinstance(an, dict):
                section += f"\nAnalysis: {json.dumps(sanitize_for_json(an), indent=2, default=str)}"
            parts.append(section)
    return "\n\n".join(parts)


def attach_live_activity(state, agents_dir, agent_id):
    """Enrich an agent *state* dict with live notes, chat, partial results, and stdout tail."""
    agent_dir = Path(agents_dir) / agent_id
    ws = agent_dir / "workspace"

    for fname, key in [("notes.txt", "agent_notes"), ("STATE.md", "state_md")]:
        fpath = ws / fname
        try:
            if fpath.exists():
                raw = fpath.read_text(errors="replace").strip()
                if raw:
                    state[key] = raw[-2000:]
        except OSError:
            pass

    chat_path = agent_dir / "chat.json"
    msgs = safe_json_load(chat_path, [])
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
            it["analysis"] = safe_json_load(analysis_path)

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
            raw_result = safe_json_load(result_path, {})
            raw_result.pop("feature_stats", None)
            raw_result.pop("feature_analysis", None)
            analysis = safe_json_load(analysis_path) if analysis_path.exists() else None
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

    stdout_path = agent_dir / "stdout.log"
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


def get_agent_detail(agents_dir, agent_id, is_alive_fn):
    """Load agent state, check liveness, attach live activity, and return sanitised dict."""
    log_path = Path(agents_dir) / agent_id / "log.json"
    state = safe_json_load(log_path)
    if not state:
        return None
    if state.get("status") in ("running", "paused") and not is_alive_fn(agent_id):
        state["status"] = "crashed"
    state["status"] = sanitize_status(state.get("status", "unknown"))
    attach_live_activity(state, agents_dir, agent_id)
    return sanitize_for_json(state)


def _make_default(default):
    return list(default) if isinstance(default, list) else dict(default or {})
