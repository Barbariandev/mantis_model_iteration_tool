"""MANTIS remote agent management server.

Runs on Targon. Manages the full agent lifecycle: creates agents,
runs Claude Code CLI, executes walk-forward evaluation, and serves
agent state back to the GUI over HTTPS.

All agent operations (Claude Code, data fetching, evaluation) run
here — the user's machine only runs the lightweight Flask GUI.
"""

import hmac
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("mantis.server")

PKG_DIR = Path(__file__).resolve().parent

AGENTS_DIR = Path(os.environ.get("MANTIS_AGENTS_DIR", "/data/agents"))
DATA_DIR = Path(os.environ.get("MANTIS_DATA_DIR", "/data/cache"))
os.environ.setdefault("MANTIS_DATA_DIR", str(DATA_DIR))

SERVER_AUTH_KEY = os.environ.get("MANTIS_SERVER_AUTH_KEY", "")
MAX_AGENTS = int(os.environ.get("MANTIS_MAX_AGENTS", "5"))

SAFE_ID = re.compile(r"^[a-f0-9]{1,40}$")

app = FastAPI(title="MANTIS Agent Server", version="1.0.0")

_agent_procs_lock = threading.Lock()
_agent_procs: dict[str, subprocess.Popen] = {}
_ask_semaphore = threading.Semaphore(3)


# ── Auth ─────────────────────────────────────────────────────────────────────

async def _verify_auth(authorization: str = Header("")):
    if not SERVER_AUTH_KEY:
        raise HTTPException(
            status_code=503,
            detail="MANTIS_SERVER_AUTH_KEY is not configured",
        )
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token, SERVER_AUTH_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _validate_id(agent_id: str):
    if not SAFE_ID.match(agent_id):
        raise HTTPException(status_code=400, detail="Invalid agent ID")


# ── Helpers ──────────────────────────────────────────────────────────────────

try:
    from mantis_model_iteration_tool.utils import (
        sanitize_for_json as _sanitize_for_json,
        safe_json_load as _safe_json_load,
        sanitize_status as _sanitize_status,
        get_agent_detail as _get_agent_detail_shared,
    )
except ImportError:
    from utils import (
        sanitize_for_json as _sanitize_for_json,
        safe_json_load as _safe_json_load,
        sanitize_status as _sanitize_status,
        get_agent_detail as _get_agent_detail_shared,
    )


def _running_count():
    with _agent_procs_lock:
        alive = {k: p for k, p in _agent_procs.items() if p.poll() is None}
        _agent_procs.clear()
        _agent_procs.update(alive)
        return len(alive)


def _agent_is_alive(agent_id: str) -> bool:
    with _agent_procs_lock:
        p = _agent_procs.get(agent_id)
        if p and p.poll() is None:
            return True
        _agent_procs.pop(agent_id, None)
    return False


# ── Agent lifecycle ──────────────────────────────────────────────────────────

def _launch_agent(agent_id: str, config: dict):
    agent_dir = AGENTS_DIR / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    initial_state = {
        "id": agent_id,
        "status": "starting",
        "config": config,
        "iteration": 0,
        "iterations": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (agent_dir / "log.json").write_text(json.dumps(initial_state, indent=2))

    cmd = [
        sys.executable, "-m", "mantis_model_iteration_tool.agent_runner",
        "--challenge", config["challenge"],
        "--goal", config["goal"],
        "--min-iterations", str(config.get("min_iterations", 5)),
        "--max-iterations", str(config.get("max_iterations", 20)),
        "--model", config.get("model", "sonnet"),
        "--days-back", str(config.get("days_back", 60)),
        "--id", agent_id,
        "--agent-dir", str(agent_dir),
        "--data-dir", str(DATA_DIR),
    ]

    env = os.environ.copy()
    env["MANTIS_AGENT_DIR"] = str(agent_dir)
    env["MANTIS_DATA_DIR"] = str(DATA_DIR)

    stdout_log = agent_dir / "stdout.log"
    with open(stdout_log, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env=env, cwd=str(PKG_DIR.parent),
        )

    with _agent_procs_lock:
        _agent_procs[agent_id] = proc

    logger.info("Launched agent %s (pid=%d)", agent_id, proc.pid)


def _kill_agent(agent_id: str):
    with _agent_procs_lock:
        p = _agent_procs.pop(agent_id, None)
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    stop_path = AGENTS_DIR / agent_id / "STOP"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.touch()


# ── Agent state readers (mirrors gui.py logic) ──────────────────────────────

def _list_agents_summary():
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
        agents.append(summary)
    agents.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return _sanitize_for_json(agents)


def _get_agent_detail(agent_id: str):
    return _get_agent_detail_shared(AGENTS_DIR, agent_id, _agent_is_alive)


# ── Chat / ask helpers ───────────────────────────────────────────────────────

def _chat_append(agent_id, role, text, msg_type="message"):
    try:
        from mantis_model_iteration_tool.utils import chat_append
    except ImportError:
        from utils import chat_append
    chat_append(AGENTS_DIR / agent_id / "chat.json", role, text, msg_type)


def _build_agent_context(agent_id):
    try:
        from mantis_model_iteration_tool.utils import build_agent_context
    except ImportError:
        from utils import build_agent_context
    return build_agent_context(AGENTS_DIR, agent_id)


def _run_ask_claude(agent_id, question):
    if not _ask_semaphore.acquire(blocking=False):
        _chat_append(agent_id, "assistant",
                     "Too many concurrent questions. Try again in a moment.", "ask")
        return
    try:
        if not (AGENTS_DIR / agent_id).exists():
            return
        context = _build_agent_context(agent_id)
        system_prompt = (
            "You are a research assistant analyzing an autonomous prediction strategy agent. "
            "Answer the operator's question using the agent context below. Be concise and specific. "
            "Do not make up data -- only reference what is in the context."
        )
        user_msg = f"{context}\n\n## Operator Question\n\n{question}"
        ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not ant_key:
            _chat_append(agent_id, "assistant", "Error: Anthropic API key not configured.", "ask")
            return
        try:
            import requests
            resp = requests.post(
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
        if (AGENTS_DIR / agent_id).exists():
            _chat_append(agent_id, "assistant", response[:5000], "ask")
    finally:
        _ask_semaphore.release()


# ── JSON state update (safe file writes) ─────────────────────────────────────

try:
    from mantis_model_iteration_tool.utils import locked_json_update as _locked_json_update
except ImportError:
    from utils import locked_json_update as _locked_json_update


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "auth_required": True,
        "auth_configured": bool(SERVER_AUTH_KEY),
    }


@app.post("/api/coinglass-bundle", dependencies=[Depends(_verify_auth)])
async def upload_coinglass_bundle(request: Request):
    """Accept a tar.gz bundle of CoinGlass .npz files from the GUI.

    The GUI fetches CG data locally (where the API works) and uploads
    the pre-aligned numpy arrays here.  They're saved into every
    ``<days>d/coinglass/`` cache dir so agents find them during prefetch.
    """
    import tarfile, io
    body = await request.body()
    if len(body) > 200 * 1024 * 1024:
        raise HTTPException(413, "Bundle too large (>200 MB)")
    if len(body) < 10:
        raise HTTPException(400, "Empty bundle")
    try:
        buf = io.BytesIO(body)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            npz_names = [n for n in names if n.endswith(".npz")]
            if not npz_names:
                raise HTTPException(400, "No .npz files in bundle")
            for member in tar.getmembers():
                if not member.name.endswith(".npz"):
                    continue
                if member.name != os.path.basename(member.name):
                    continue
                for days in (60, 90, 120, 180, 365):
                    cg_dir = DATA_DIR / f"{days}d" / "coinglass"
                    cg_dir.mkdir(parents=True, exist_ok=True)
                    f = tar.extractfile(member)
                    if f:
                        dest = cg_dir / member.name
                        dest.write_bytes(f.read())
                        f.seek(0)
            assets = [n.replace(".npz", "") for n in npz_names]
            logger.info("CoinGlass bundle uploaded: %d assets (%s)",
                        len(assets), ", ".join(assets[:5]))
            return {"uploaded": len(assets), "assets": assets}
    except tarfile.TarError as exc:
        raise HTTPException(400, f"Invalid tar.gz: {exc}")


@app.get("/api/challenges", dependencies=[Depends(_verify_auth)])
async def list_challenges():
    from mantis_model_iteration_tool.evaluator import CHALLENGES
    result = {}
    for cname, cfg in CHALLENGES.items():
        result[cname] = {
            "type": cfg.challenge_type,
            "short": cname,
            "desc": "",
            "asset": "?",
            "metric": "?",
        }
    return result


class CreateAgentRequest(BaseModel):
    challenge: str = Field(..., max_length=64)
    goal: str = Field(..., max_length=10000)
    model: str = Field("sonnet", max_length=32)
    min_iterations: int = Field(5, ge=1, le=100)
    max_iterations: int = Field(20, ge=1, le=500)
    days_back: int = Field(60, ge=60, le=365)


@app.post("/api/agents", dependencies=[Depends(_verify_auth)])
async def create_agent(req: CreateAgentRequest):
    from mantis_model_iteration_tool.evaluator import CHALLENGES
    if req.challenge not in CHALLENGES:
        raise HTTPException(400, f"Unknown challenge: {req.challenge}")
    if req.model not in ("sonnet", "opus", "haiku"):
        req.model = "sonnet"

    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ant_key:
        raise HTTPException(400, "ANTHROPIC_API_KEY not configured on server")

    n_running = _running_count()
    if n_running >= MAX_AGENTS:
        raise HTTPException(429, f"Max {MAX_AGENTS} concurrent agents reached")

    import uuid
    agent_id = uuid.uuid4().hex[:12]
    config = {
        "challenge": req.challenge,
        "goal": req.goal,
        "min_iterations": req.min_iterations,
        "max_iterations": req.max_iterations,
        "model": req.model,
        "days_back": req.days_back,
    }
    _launch_agent(agent_id, config)
    return {"id": agent_id, "status": "starting"}


@app.get("/api/agents", dependencies=[Depends(_verify_auth)])
async def list_agents(full: str = "0"):
    if full == "1":
        agents = []
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        for d in sorted(AGENTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            detail = _get_agent_detail(d.name)
            if detail:
                agents.append(detail)
        agents.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        return _sanitize_for_json(agents)
    return _list_agents_summary()


@app.get("/api/agents/{agent_id}", dependencies=[Depends(_verify_auth)])
async def get_agent(agent_id: str):
    _validate_id(agent_id)
    state = _get_agent_detail(agent_id)
    if not state:
        raise HTTPException(404)
    return state


@app.post("/api/agents/{agent_id}/stop", dependencies=[Depends(_verify_auth)])
async def stop_agent(agent_id: str):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    (agent_dir / "STOP").touch()
    (agent_dir / "PAUSE").unlink(missing_ok=True)
    _kill_agent(agent_id)
    log_path = agent_dir / "log.json"
    if log_path.exists():
        def _mark(state):
            if state.get("status") == "running":
                state["status"] = "stopped"
        _locked_json_update(log_path, _mark)
    return {"status": "stopped"}


@app.post("/api/agents/{agent_id}/delete", dependencies=[Depends(_verify_auth)])
async def delete_agent(agent_id: str):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    _kill_agent(agent_id)
    shutil.rmtree(agent_dir, ignore_errors=True)
    return {"status": "deleted"}


@app.get("/api/agents/{agent_id}/code/{iteration}", dependencies=[Depends(_verify_auth)])
async def get_code(agent_id: str, iteration: int):
    _validate_id(agent_id)
    code_path = AGENTS_DIR / agent_id / "workspace" / f"iteration_{iteration}.py"
    if not code_path.exists():
        code_path = AGENTS_DIR / agent_id / f"iteration_{iteration}.py"
    if not code_path.exists():
        raise HTTPException(404)
    return PlainTextResponse(code_path.read_text(errors="replace"))


@app.get("/api/agents/{agent_id}/inbox", dependencies=[Depends(_verify_auth)])
async def get_inbox(agent_id: str):
    _validate_id(agent_id)
    inbox_path = AGENTS_DIR / agent_id / "workspace" / "INBOX.md"
    content = inbox_path.read_text(errors="replace") if inbox_path.exists() else ""
    return {"content": content}


@app.post("/api/agents/{agent_id}/inbox", dependencies=[Depends(_verify_auth)])
async def set_inbox(agent_id: str, request: Request):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    data = await request.json()
    content = str(data.get("content", ""))[:50000]
    workspace = agent_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "INBOX.md").write_text(content)
    return {"status": "ok"}


@app.post("/api/agents/{agent_id}/message", dependencies=[Depends(_verify_auth)])
async def message_agent(agent_id: str, request: Request):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    data = await request.json()
    text = str(data.get("text") or "").strip()[:10000]
    if not text:
        raise HTTPException(400, "text required")

    _chat_append(agent_id, "user", text, "message")

    workspace = agent_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    inbox_path = workspace / "INBOX.md"

    import fcntl
    import tempfile
    lock_path = workspace / "INBOX.md.lock"
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
    return {"status": "sent"}


@app.post("/api/agents/{agent_id}/ask", dependencies=[Depends(_verify_auth)])
async def ask_agent(agent_id: str, request: Request):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    data = await request.json()
    text = str(data.get("text") or "").strip()[:10000]
    if not text:
        raise HTTPException(400, "text required")
    _chat_append(agent_id, "user", text, "ask")
    threading.Thread(target=_run_ask_claude, args=(agent_id, text), daemon=True).start()
    return {"status": "processing"}


@app.get("/api/agents/{agent_id}/chat", dependencies=[Depends(_verify_auth)])
async def get_chat(agent_id: str):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    msgs = _safe_json_load(agent_dir / "chat.json", [])
    paused = (agent_dir / "PAUSE").exists()
    return {"messages": msgs, "paused": paused}


@app.post("/api/agents/{agent_id}/pause", dependencies=[Depends(_verify_auth)])
async def pause_agent(agent_id: str):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    (agent_dir / "PAUSE").touch()
    return {"paused": True}


@app.post("/api/agents/{agent_id}/resume", dependencies=[Depends(_verify_auth)])
async def resume_agent(agent_id: str):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    (agent_dir / "PAUSE").unlink(missing_ok=True)
    return {"paused": False}


@app.get("/api/agents/{agent_id}/goal", dependencies=[Depends(_verify_auth)])
async def get_goal(agent_id: str):
    _validate_id(agent_id)
    goal_path = AGENTS_DIR / agent_id / "workspace" / "GOAL.md"
    if not goal_path.exists():
        return {"content": ""}
    return {"content": goal_path.read_text(errors="replace")}


@app.post("/api/agents/{agent_id}/goal", dependencies=[Depends(_verify_auth)])
async def update_goal(agent_id: str, request: Request):
    _validate_id(agent_id)
    agent_dir = AGENTS_DIR / agent_id
    if not agent_dir.exists():
        raise HTTPException(404)
    data = await request.json()
    new_goal = str(data.get("goal") or "").strip()[:10000]
    if not new_goal:
        raise HTTPException(400, "goal required")

    log_path = agent_dir / "log.json"
    if log_path.exists():
        def _set(state):
            state.setdefault("config", {})["goal"] = new_goal
        _locked_json_update(log_path, _set)

    workspace = agent_dir / "workspace"
    if workspace.exists():
        (workspace / "GOAL.md").write_text(f"# Research Goal\n\n{new_goal}\n")
    return {"status": "updated"}


@app.get("/api/agents/{agent_id}/notes", dependencies=[Depends(_verify_auth)])
async def get_notes(agent_id: str):
    _validate_id(agent_id)
    notes_path = AGENTS_DIR / agent_id / "workspace" / "notes.txt"
    try:
        if notes_path.exists():
            return {"content": notes_path.read_text(errors="replace")}
    except OSError:
        pass
    return {"content": ""}


@app.get("/api/agents/{agent_id}/state", dependencies=[Depends(_verify_auth)])
async def get_state(agent_id: str):
    _validate_id(agent_id)
    state_path = AGENTS_DIR / agent_id / "workspace" / "STATE.md"
    if not state_path.exists():
        return {"content": ""}
    return {"content": state_path.read_text(errors="replace")}


@app.get("/api/agents/{agent_id}/stdout", dependencies=[Depends(_verify_auth)])
async def get_stdout(agent_id: str, tail: int = 200):
    _validate_id(agent_id)
    stdout_path = AGENTS_DIR / agent_id / "stdout.log"
    if not stdout_path.exists():
        return {"content": "", "lines": 0}
    tail = min(max(1, tail), 5000)
    tail_bytes = min(tail * 200, 2 * 1024 * 1024)
    try:
        fsize = stdout_path.stat().st_size
        with open(stdout_path, "r", errors="replace") as sf:
            if fsize > tail_bytes:
                sf.seek(fsize - tail_bytes)
                sf.readline()
            raw = sf.read()
    except OSError:
        return {"content": "", "lines": 0}
    lines = raw.strip().split("\n")
    out = lines[-tail:]
    return {"content": "\n".join(out), "lines": len(lines), "showing": len(out)}


@app.get("/api/agents/{agent_id}/iterations", dependencies=[Depends(_verify_auth)])
async def get_iterations(agent_id: str):
    _validate_id(agent_id)
    state = _get_agent_detail(agent_id)
    if not state:
        raise HTTPException(404)
    iterations = state.get("iterations", [])
    return {"iterations": iterations, "count": len(iterations)}


@app.get("/api/agents/{agent_id}/iterations/{iteration}/features",
         dependencies=[Depends(_verify_auth)])
async def get_features(agent_id: str, iteration: int):
    _validate_id(agent_id)
    state = _get_agent_detail(agent_id)
    if not state:
        raise HTTPException(404)
    for it in state.get("iterations", []):
        if it.get("iteration") == iteration:
            analysis = it.get("analysis") or {}
            feature_report = analysis.get("feature_report", analysis.get("features", []))
            return {"iteration": iteration, "feature_report": feature_report}
    raise HTTPException(404)


@app.get("/api/agents/{agent_id}/iterations/{iteration}/metrics",
         dependencies=[Depends(_verify_auth)])
async def get_metrics(agent_id: str, iteration: int):
    _validate_id(agent_id)
    state = _get_agent_detail(agent_id)
    if not state:
        raise HTTPException(404)
    for it in state.get("iterations", []):
        if it.get("iteration") == iteration:
            return {
                "iteration": iteration,
                "metrics": it.get("metrics", {}),
                "analysis": it.get("analysis"),
            }
    raise HTTPException(404)


MINER_DIR = Path(os.environ.get("MANTIS_MINER_DIR", "/data/miner"))

_miner_proc_lock = threading.Lock()
_miner_proc: dict[str, subprocess.Popen] = {}


class UploadStrategyRequest(BaseModel):
    agent_id: str = Field(..., max_length=40)
    iteration: int = Field(..., ge=0, le=100)
    code: str = Field(..., min_length=10, max_length=500_000)


@app.post("/api/miner/upload-strategy", dependencies=[Depends(_verify_auth)])
async def upload_strategy(req: UploadStrategyRequest):
    """Upload a strategy file so the miner can find it."""
    aid = req.agent_id
    if not SAFE_ID.match(aid):
        raise HTTPException(400, "Invalid agent ID")
    target_dir = AGENTS_DIR / aid / "workspace"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"iteration_{req.iteration}.py"
    target_file.write_text(req.code)
    logger.info("Uploaded strategy %s iter %d (%d bytes)",
                aid, req.iteration, len(req.code))
    return {"status": "ok", "path": str(target_file)}


@app.post("/api/setup/install-deps", dependencies=[Depends(_verify_auth)])
async def install_deps():
    """Trigger install_reqs.sh to install miner dependencies."""
    _ensure_miner_deps()
    venv_py = VENV_DIR / "bin" / "python"
    installed = venv_py.exists()
    pkgs = []
    if installed:
        r = subprocess.run(
            [str(venv_py), "-c",
             "import boto3; import bittensor; import timelock; print('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            pkgs = ["boto3", "bittensor", "timelock"]
        else:
            r2 = subprocess.run(
                [str(venv_py), "-c",
                 "import boto3; import bittensor; print('ok')"],
                capture_output=True, text=True, timeout=30,
            )
            if r2.returncode == 0:
                pkgs = ["boto3", "bittensor"]
    return {"installed": installed, "packages": pkgs}


@app.get("/api/setup/status", dependencies=[Depends(_verify_auth)])
async def setup_status():
    """Check if miner dependencies are installed."""
    venv_py = VENV_DIR / "bin" / "python"
    if not venv_py.exists():
        return {"deps_installed": False, "venv_exists": False, "packages": []}
    r = subprocess.run(
        [str(venv_py), "-c",
         "import boto3; import bittensor; import timelock; print('ok')"],
        capture_output=True, text=True, timeout=30,
    )
    ok = r.returncode == 0
    pkgs = ["boto3", "bittensor", "timelock"] if ok else []
    return {
        "deps_installed": ok,
        "venv_exists": True,
        "packages": pkgs,
        "error": r.stderr.strip()[:500] if not ok else "",
    }


PM2_APP_NAME = "mantis-miner"


def _pm2_available() -> bool:
    return shutil.which("pm2") is not None


def _ensure_pm2():
    """Install pm2 globally if not present."""
    if _pm2_available():
        return
    logger.info("Installing pm2...")
    subprocess.run(["npm", "install", "-g", "pm2"],
                   capture_output=True, timeout=120)


def _miner_is_alive() -> bool:
    if _pm2_available():
        try:
            r = subprocess.run(
                ["pm2", "jlist"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                import json as _json
                for proc in _json.loads(r.stdout):
                    if proc.get("name") == PM2_APP_NAME:
                        return proc.get("pm2_env", {}).get("status") == "online"
        except Exception:
            pass
    with _miner_proc_lock:
        p = _miner_proc.get("miner")
        if p and p.poll() is None:
            return True
        _miner_proc.pop("miner", None)
    return False


class StartMinerRequest(BaseModel):
    models: list[str] = Field(..., min_length=1)
    interval_seconds: int = Field(60, ge=30, le=300)
    lock_seconds: int = Field(30, ge=10, le=120)
    lookback: int = Field(5000, ge=500, le=15000)
    network: str = Field("finney", max_length=32)
    hotkey_mnemonic: str = Field("", max_length=2000)
    hotkey_path: str = Field("", max_length=500)
    wallet_name: str = Field("", max_length=128)
    hotkey_name: str = Field("", max_length=128)
    hotkey_ss58: str = Field("", max_length=128)
    r2_account_id: str = Field("", max_length=128)
    r2_access_key_id: str = Field("", max_length=256)
    r2_secret_access_key: str = Field("", max_length=256)
    r2_bucket_name: str = Field("", max_length=256)
    r2_public_base_url: str = Field("", max_length=512)
    coinglass_key: str = Field("", max_length=256)


@app.post("/api/miner/start", dependencies=[Depends(_verify_auth)])
async def start_miner(req: StartMinerRequest):
    if _miner_is_alive():
        raise HTTPException(409, "Miner already running")

    MINER_DIR.mkdir(parents=True, exist_ok=True)
    (MINER_DIR / "MINER_STOP").unlink(missing_ok=True)
    (MINER_DIR / "MINER_FAILED").unlink(missing_ok=True)

    miner_py = _miner_python()
    miner_args = [
        miner_py, "-m", "mantis_model_iteration_tool.miner",
        "--miner-dir", str(MINER_DIR),
        "--interval", str(req.interval_seconds),
        "--lock-seconds", str(req.lock_seconds),
        "--lookback", str(req.lookback),
        "--network", req.network,
    ]
    for model_spec in req.models:
        miner_args.extend(["--model", model_spec])

    env_vars = {}
    if req.hotkey_mnemonic:
        env_vars["HOTKEY_MNEMONIC"] = req.hotkey_mnemonic
    if req.hotkey_path:
        env_vars["HOTKEY_PATH"] = req.hotkey_path
    if req.wallet_name:
        env_vars["BT_WALLET_NAME"] = req.wallet_name
    if req.hotkey_name:
        env_vars["BT_HOTKEY_NAME"] = req.hotkey_name
    if req.hotkey_ss58:
        env_vars["HOTKEY_SS58"] = req.hotkey_ss58
    if req.r2_account_id:
        env_vars["R2_ACCOUNT_ID"] = req.r2_account_id
    if req.r2_access_key_id:
        env_vars["R2_ACCESS_KEY_ID"] = req.r2_access_key_id
    if req.r2_secret_access_key:
        env_vars["R2_SECRET_ACCESS_KEY"] = req.r2_secret_access_key
    if req.r2_bucket_name:
        env_vars["R2_BUCKET_NAME"] = req.r2_bucket_name
    if req.r2_public_base_url:
        env_vars["R2_PUBLIC_BASE_URL"] = req.r2_public_base_url
    if req.coinglass_key:
        env_vars["COINGLASS_API_KEY"] = req.coinglass_key

    venv_sp = _venv_site_packages()
    if venv_sp:
        env_vars["PYTHONPATH"] = venv_sp

    stdout_log = MINER_DIR / "stdout.log"

    # Try pm2 for proper process management
    _ensure_pm2()
    if _pm2_available():
        # Stop any existing pm2 miner process
        subprocess.run(["pm2", "delete", PM2_APP_NAME],
                       capture_output=True, timeout=15)

        # Write a launcher shell script with env vars baked in
        launcher = MINER_DIR / "_miner_launch.sh"
        lines = ["#!/usr/bin/env bash", "set -e"]
        for k, v in env_vars.items():
            lines.append(f"export {k}={shlex.quote(v)}")
        lines.append(f"cd {shlex.quote(str(PKG_DIR.parent))}")
        lines.append(f"exec {' '.join(shlex.quote(a) for a in miner_args)} 2>&1")
        launcher.write_text("\n".join(lines) + "\n")
        launcher.chmod(0o700)

        # Clear old log
        stdout_log.write_text("")

        r = subprocess.run(
            ["pm2", "start", str(launcher),
             "--name", PM2_APP_NAME,
             "--no-autorestart",
             "-o", str(stdout_log),
             "-e", str(stdout_log),
             "--merge-logs"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PKG_DIR.parent),
        )
        if r.returncode != 0:
            logger.error("pm2 start failed: %s %s", r.stdout[:500], r.stderr[:500])
            raise HTTPException(500, f"pm2 start failed: {r.stderr[:300]}")

        # Get PID from pm2
        pid = 0
        try:
            r2 = subprocess.run(
                ["pm2", "jlist"], capture_output=True, text=True, timeout=10)
            for proc in json.loads(r2.stdout):
                if proc.get("name") == PM2_APP_NAME:
                    pid = proc.get("pid", 0)
        except Exception:
            pass

        try:
            launcher.unlink()
        except OSError:
            pass
        logger.info("Launched miner via pm2 (pid=%d)", pid)
        return {"status": "started", "pid": pid}

    # Fallback: bare subprocess
    env = os.environ.copy()
    env.update(env_vars)
    with open(stdout_log, "w") as f:
        proc = subprocess.Popen(
            miner_args, stdout=f, stderr=subprocess.STDOUT,
            env=env, cwd=str(PKG_DIR.parent),
        )
    with _miner_proc_lock:
        _miner_proc["miner"] = proc
    logger.info("Launched miner (pid=%d, subprocess fallback)", proc.pid)
    return {"status": "started", "pid": proc.pid}


@app.post("/api/miner/stop", dependencies=[Depends(_verify_auth)])
async def stop_miner():
    MINER_DIR.mkdir(parents=True, exist_ok=True)
    (MINER_DIR / "MINER_STOP").touch()

    # Try pm2 first
    if _pm2_available():
        subprocess.run(["pm2", "stop", PM2_APP_NAME],
                       capture_output=True, timeout=15)
        subprocess.run(["pm2", "delete", PM2_APP_NAME],
                       capture_output=True, timeout=15)

    # Also clean up any bare subprocess
    with _miner_proc_lock:
        p = _miner_proc.pop("miner", None)
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()

    return {"status": "stopped"}


@app.get("/api/miner/status", dependencies=[Depends(_verify_auth)])
async def miner_status():
    status_path = MINER_DIR / "miner_status.json"
    failed_path = MINER_DIR / "MINER_FAILED"
    status_data = _safe_json_load(status_path, {})
    failed_data = None
    if failed_path.exists():
        failed_data = _safe_json_load(failed_path)

    alive = _miner_is_alive()
    if status_data.get("running") and not alive:
        status_data["running"] = False
        status_data["crashed"] = True

    return {
        "status": status_data,
        "failed": failed_data,
        "process_alive": alive,
    }


class CheckRegistrationRequest(BaseModel):
    hotkey_ss58: str = Field("", max_length=128)
    hotkey_mnemonic: str = Field("", max_length=2000)
    network: str = Field("finney", max_length=32)


@app.post("/api/miner/check-registration", dependencies=[Depends(_verify_auth)])
async def check_miner_registration(req: CheckRegistrationRequest):
    from mantis_model_iteration_tool.miner import check_registration, _resolve_hotkey

    hotkey_ss58 = req.hotkey_ss58
    if not hotkey_ss58 and req.hotkey_mnemonic:
        hotkey_ss58, _ = _resolve_hotkey(mnemonic=req.hotkey_mnemonic)
    if not hotkey_ss58:
        raise HTTPException(400, "Provide hotkey_ss58 or hotkey_mnemonic")

    result = check_registration(hotkey_ss58, network=req.network)
    return result


class RegisterMinerRequest(BaseModel):
    coldkey_mnemonic: str = Field(..., min_length=10, max_length=2000)
    hotkey_mnemonic: str = Field(..., min_length=10, max_length=2000)
    network: str = Field("finney", max_length=32)


@app.post("/api/miner/register", dependencies=[Depends(_verify_auth)])
async def register_miner_endpoint(req: RegisterMinerRequest):
    from mantis_model_iteration_tool.miner import register_miner

    result = register_miner(
        coldkey_mnemonic=req.coldkey_mnemonic,
        hotkey_mnemonic=req.hotkey_mnemonic,
        network=req.network,
    )
    if result.success:
        return result.to_dict()
    raise HTTPException(400, detail=result.to_dict())


@app.get("/api/miner/stdout", dependencies=[Depends(_verify_auth)])
async def miner_stdout(tail: int = 200):
    stdout_path = MINER_DIR / "stdout.log"
    if not stdout_path.exists():
        return {"content": "", "lines": 0}
    tail = min(max(1, tail), 5000)
    tail_bytes = min(tail * 200, 2 * 1024 * 1024)
    try:
        fsize = stdout_path.stat().st_size
        with open(stdout_path, "r", errors="replace") as sf:
            if fsize > tail_bytes:
                sf.seek(fsize - tail_bytes)
                sf.readline()
            raw = sf.read()
    except OSError:
        return {"content": "", "lines": 0}
    lines = raw.strip().split("\n")
    out = lines[-tail:]
    return {"content": "\n".join(out), "lines": len(lines), "showing": len(out)}


VENV_DIR = Path("/data/.venv")
INSTALL_REQS = PKG_DIR / "install_reqs.sh"


def _ensure_miner_deps():
    """Run install_reqs.sh to create venv and install miner dependencies.

    Also adds the venv site-packages to sys.path so in-process imports
    (e.g. check_registration, register_miner) can find boto3/bittensor.
    """
    if not INSTALL_REQS.exists():
        logger.warning("install_reqs.sh not found at %s", INSTALL_REQS)
        return

    # If venv exists, verify timelock is importable; if not, clear marker
    # to force reinstall
    venv_py = VENV_DIR / "bin" / "python"
    marker = VENV_DIR / ".installed"
    if venv_py.exists() and marker.exists():
        r = subprocess.run(
            [str(venv_py), "-c", "import timelock"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            logger.info("timelock not importable in venv, clearing marker to force reinstall")
            marker.unlink(missing_ok=True)

    INSTALL_REQS.chmod(0o755)
    logger.info("Running install_reqs.sh ...")
    r = subprocess.run(
        ["bash", str(INSTALL_REQS)],
        capture_output=True, text=True, timeout=900,
        cwd=str(PKG_DIR),
    )
    if r.returncode != 0:
        logger.error("install_reqs.sh failed (exit %d):\nSTDOUT:\n%s\nSTDERR:\n%s",
                      r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    else:
        for line in r.stdout.strip().splitlines():
            logger.info("  %s", line)

    _inject_venv_site_packages()


def _inject_venv_site_packages():
    """Add venv's site-packages to sys.path for in-process imports."""
    import glob as _glob
    pattern = str(VENV_DIR / "lib" / "python*" / "site-packages")
    for sp in _glob.glob(pattern):
        if sp not in sys.path:
            sys.path.insert(0, sp)
            logger.info("Added venv site-packages to sys.path: %s", sp)


def _miner_python() -> str:
    """Return the system Python — it can always find mantis_model_iteration_tool."""
    return sys.executable


def _venv_site_packages() -> str:
    """Return the venv's site-packages path (for PYTHONPATH injection)."""
    import glob as _glob
    pattern = str(VENV_DIR / "lib" / "python*" / "site-packages")
    for sp in _glob.glob(pattern):
        return sp
    return ""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    if not SERVER_AUTH_KEY:
        raise SystemExit(
            "Refusing to start without MANTIS_SERVER_AUTH_KEY. "
            "Generate one with: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MINER_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_miner_deps()
    uvicorn.run(app, host="0.0.0.0", port=port)
