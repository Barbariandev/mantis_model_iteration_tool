"""MANTIS Agentic Mining -- autonomous agent loop.

Each iteration is a claude -p call where the agent reads files, writes
strategy code, runs evaluation, and writes analysis. Agents run in
isolated workspaces inside Docker containers for persistence.
"""

import argparse
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

PLAYGROUND_DIR = Path(__file__).parent
WORKING_DIR = Path(__file__).parent.parent

_AGENTS_DIR_OVERRIDE = os.environ.get("MANTIS_AGENT_DIR")
_DATA_DIR_OVERRIDE = os.environ.get("MANTIS_DATA_DIR")
AGENTS_DIR = Path(_AGENTS_DIR_OVERRIDE) if _AGENTS_DIR_OVERRIDE else PLAYGROUND_DIR / "agents"

if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))


def _load_anthropic_key():
    return os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_TIMEOUT = int(os.environ.get("MANTIS_CLAUDE_TIMEOUT", "600"))
ITER_WALL_TIMEOUT = int(os.environ.get("MANTIS_ITER_WALL_TIMEOUT", "1800"))
IS_ROOT = os.getuid() == 0
EVAL_HOLDOUT_DAYS = 20
MIN_DAYS_BACK = 60

_stop_event = threading.Event()
_child_procs = set()
_child_procs_lock = threading.Lock()

PLAYGROUND_FILES = [
    "__init__.py", "data.py", "evaluator.py", "featurizer.py",
    "coinglass.py", "data_cache.py", "example_binary.py", "GUIDE.md",
]

CHALLENGE_META = {
    "binary": {
        "format": "[p_up, p_down] -- probabilities summing to ~1",
        "primary_metric": "mean_auc",
        "direction": "higher is better",
        "all_metrics": ["mean_auc"],
    },
    "hitfirst": {
        "format": "[p_up_first, p_down_first, p_neither] -- probabilities summing to ~1",
        "primary_metric": "direct_log_loss",
        "direction": "lower is better",
        "all_metrics": ["direct_log_loss", "up_auc", "dn_auc"],
    },
    "lbfgs": {
        "format": "[bucket_0..4, qpath_0..11] -- 5 bucket probs + 12 q-path probs",
        "primary_metric": "mean_balanced_accuracy",
        "direction": "higher is better",
        "all_metrics": ["mean_balanced_accuracy", "mean_lift"],
    },
    "breakout": {
        "format": "[asset_0_cont, asset_0_rev, ...] -- per-asset continuation/reversal probs",
        "primary_metric": "mean_auc",
        "direction": "higher is better",
        "all_metrics": ["mean_auc"],
    },
    "xsec_rank": {
        "format": "[asset_0_score, asset_1_score, ...] -- higher = outperform median",
        "primary_metric": "mean_spearman",
        "direction": "higher is better",
        "all_metrics": ["mean_spearman"],
    },
}


# ── Workspace setup ──────────────────────────────────────────────────────────

def _setup_workspace(agent_dir):
    """Create an isolated workspace with a copy of the playground framework.
    Framework files are refreshed every call so agents get bug fixes."""
    workspace = agent_dir / "workspace"
    pg_dest = workspace / "playground"
    first_time = not pg_dest.exists()
    pg_dest.mkdir(parents=True, exist_ok=True)

    for fname in PLAYGROUND_FILES:
        src = PLAYGROUND_DIR / fname
        if src.exists():
            shutil.copy2(src, pg_dest / fname)

    if first_time:
        subprocess.run(["git", "init", "-q", str(workspace)],
                       capture_output=True, timeout=10)
    return workspace


# ── Claude settings ──────────────────────────────────────────────────────────

def _write_claude_settings(workspace, model=None):
    """Write .claude/settings.local.json inside the workspace.
    API key is NOT written to disk -- it comes from the container env."""
    settings_dir = workspace / ".claude"
    settings_dir.mkdir(exist_ok=True)
    settings = {
        "model": model or CLAUDE_MODEL,
        "permissions": {
            "allow": [
                "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
                "Glob(*)", "Grep(*)", "WebFetch(*)", "WebSearch(*)",
                "TodoWrite(*)", "NotebookEdit(*)", "Task(*)",
            ],
            "deny": [],
        },
        "env": {
            "ANTHROPIC_AUTH_TOKEN": "",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    }
    (settings_dir / "settings.local.json").write_text(json.dumps(settings, indent=2))


def _claude_env():
    env = os.environ.copy()
    env.setdefault("ANTHROPIC_API_KEY", _load_anthropic_key())
    env["ANTHROPIC_AUTH_TOKEN"] = ""
    return env


def _claude_cmd(prompt):
    cmd = ["claude", "-p", prompt]
    if not IS_ROOT:
        cmd.append("--dangerously-skip-permissions")
    cmd.extend(["--output-format", "stream-json", "--verbose"])
    return cmd


# ── Challenge helpers ────────────────────────────────────────────────────────

def _challenge_info(challenge_name):
    from playground.evaluator import CHALLENGES
    cfg = CHALLENGES[challenge_name]
    meta = CHALLENGE_META[cfg.challenge_type]
    return cfg.embedding_dim, meta["format"], meta["primary_metric"], meta["direction"], meta["all_metrics"]


# ── Tool activity formatting ────────────────────────────────────────────────

_TOOL_LABELS = {
    "Bash": "running", "Read": "reading", "Write": "writing", "Edit": "editing",
    "Glob": "searching", "Grep": "locating", "WebFetch": "downloading",
    "WebSearch": "browsing", "TodoWrite": "planning", "Task": "executing",
}


def _format_tool_activity(tool_name, tool_input):
    label = _TOOL_LABELS.get(tool_name, tool_name)
    detail = ""
    if tool_name == "Bash":
        detail = (tool_input.get("command") or "")[:80]
    elif tool_name in ("Read", "Write", "Edit"):
        detail = (tool_input.get("file_path") or tool_input.get("path") or "")
        if detail:
            detail = detail.rsplit("/", 1)[-1]
    elif tool_name == "Glob":
        detail = (tool_input.get("pattern") or tool_input.get("glob") or "")[:60]
    elif tool_name == "Grep":
        detail = (tool_input.get("pattern") or tool_input.get("regex") or "")[:60]
    elif tool_name == "WebFetch":
        detail = (tool_input.get("url") or "")[:60]
    elif tool_name == "WebSearch":
        detail = (tool_input.get("query") or tool_input.get("search_term") or "")[:60]
    elif tool_name == "Task":
        detail = (tool_input.get("description") or "")[:60]
    if detail:
        return f"{label}: {detail}"
    return f"{label}..."


# ── Streaming JSON Claude runner ─────────────────────────────────────────────

class _StopRequested(Exception):
    pass


def _run_claude_once(cmd, env, cwd, on_activity=None, on_text=None,
                     wall_timeout=None):
    """Run Claude Code CLI and parse streaming JSON output.

    Returns (returncode, result_text, tokens_in, tokens_out, cost_usd, timed_out).
    on_text is called with each completed text block from Claude (reasoning/notes).
    wall_timeout: hard wall-clock limit in seconds (kills process regardless of activity).
    """
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    with _child_procs_lock:
        _child_procs.add(proc)

    result_text = ""
    complete_texts = []
    streaming_tokens = []
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    timed_out = False
    wall_start = time.monotonic()
    last_activity = time.monotonic()
    _wall_limit = wall_timeout or ITER_WALL_TIMEOUT

    import fcntl as _fcntl
    _fd = proc.stdout.fileno()
    _fl = _fcntl.fcntl(_fd, _fcntl.F_GETFL)
    _fcntl.fcntl(_fd, _fcntl.F_SETFL, _fl | os.O_NONBLOCK)
    _fd_err = proc.stderr.fileno()
    _fl_err = _fcntl.fcntl(_fd_err, _fcntl.F_GETFL)
    _fcntl.fcntl(_fd_err, _fcntl.F_SETFL, _fl_err | os.O_NONBLOCK)

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")
    _line_buf = ""
    _stderr_buf = ""

    try:
        while True:
            if _stop_event.is_set():
                proc.kill()
                raise _StopRequested()

            ready = sel.select(timeout=min(CLAUDE_TIMEOUT, 30))
            if not ready:
                elapsed_wall = time.monotonic() - wall_start
                if elapsed_wall > _wall_limit:
                    print(f"  wall-clock timeout: iteration exceeded {_wall_limit}s ({elapsed_wall:.0f}s elapsed), killing",
                          flush=True)
                    proc.kill()
                    timed_out = True
                    break
                if time.monotonic() - last_activity > CLAUDE_TIMEOUT:
                    print(f"  claude timeout: no output for {CLAUDE_TIMEOUT}s, killing", flush=True)
                    proc.kill()
                    timed_out = True
                    break
                if proc.poll() is not None:
                    break
                continue

            for key, _ in ready:
                if key.data == "stderr":
                    try:
                        err_chunk = proc.stderr.read(65536)
                        if err_chunk:
                            _stderr_buf += err_chunk
                            if len(_stderr_buf) > 1024 * 1024:
                                _stderr_buf = _stderr_buf[-256 * 1024:]
                    except (BlockingIOError, OSError):
                        pass
                    continue

            if time.monotonic() - wall_start > _wall_limit:
                print(f"  wall-clock timeout: iteration exceeded {_wall_limit}s, killing", flush=True)
                proc.kill()
                timed_out = True
                break

            try:
                chunk = proc.stdout.read(65536)
            except (BlockingIOError, OSError):
                if proc.poll() is not None:
                    break
                continue
            if not chunk:
                break
            _line_buf += chunk
            if len(_line_buf) > 10 * 1024 * 1024:
                _line_buf = _line_buf[-1024 * 1024:]
            while "\n" in _line_buf:
                line, _line_buf = _line_buf.split("\n", 1)
                last_activity = time.monotonic()
                if not line.strip():
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type", "")
                if etype == "assistant":
                    msg = evt.get("message", {})
                    for block in msg.get("content", []):
                        btype = block.get("type", "")
                        if btype == "text" and block.get("text"):
                            if evt.get("model_call_id"):
                                complete_texts.append(block["text"])
                                if on_text:
                                    on_text(block["text"])
                                streaming_tokens.clear()
                            else:
                                streaming_tokens.append(block["text"])
                        elif btype == "tool_use" and on_activity:
                            on_activity(_format_tool_activity(
                                block.get("name", ""), block.get("input", {})))
                elif etype == "item.completed":
                    item = evt.get("item", {})
                    if item.get("type") == "agent_message" and item.get("text"):
                        complete_texts.append(item["text"])
                        if on_text:
                            on_text(item["text"])
                        streaming_tokens.clear()
                elif etype == "result":
                    result_text = evt.get("result", "")
                    u = evt.get("usage", {})
                    if u:
                        tokens_in = u.get("input_tokens", 0)
                        tokens_out = u.get("output_tokens", 0)
                    cost_usd = evt.get("total_cost_usd", 0) or 0.0
    finally:
        sel.unregister(proc.stdout)
        try:
            sel.unregister(proc.stderr)
        except Exception:
            pass
        sel.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if _stderr_buf.strip():
            print(f"  claude stderr: {_stderr_buf.strip()[-2000:]}", flush=True)
        with _child_procs_lock:
            _child_procs.discard(proc)

    for leftover in _line_buf.split("\n"):
        leftover = leftover.strip()
        if not leftover:
            continue
        try:
            evt = json.loads(leftover)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "result":
            result_text = evt.get("result", "")
            u = evt.get("usage", {})
            if u:
                tokens_in = u.get("input_tokens", 0)
                tokens_out = u.get("output_tokens", 0)
            cost_usd = evt.get("total_cost_usd", 0) or 0.0

    if not result_text:
        if complete_texts:
            result_text = complete_texts[-1]
        elif streaming_tokens:
            result_text = "".join(streaming_tokens)

    return proc.returncode, result_text, tokens_in, tokens_out, cost_usd, timed_out


# ── Context file writers ─────────────────────────────────────────────────────

def _write_api_reference(workspace):
    """Write API_REFERENCE.md once for Claude to read."""
    path = workspace / "API_REFERENCE.md"
    if path.exists():
        return
    path.write_text("""# MANTIS Agentic Mining -- API Reference

## Featurizer Interface

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from playground.featurizer import Featurizer, Predictor
from playground.data import CausalView

class TechFeaturizer(Featurizer):
    warmup: int = 0           # minimum timesteps before first valid output
    compute_interval: int = 1 # compute every N timesteps

    def compute(self, view: CausalView) -> dict[str, np.ndarray]:
        # Return dict of named feature arrays (any shape)
        ...

class TechPredictor(Predictor):
    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        # Return embedding array of shape (embedding_dim,)
        ...
```

## CausalView API

- `view.t` -> int: current timestep index
- `view.assets` -> list[str]: loaded tickers
- `view.prices(asset)` -> np.ndarray shape (t+1,): close prices 0..t
- `view.ohlcv(asset)` -> np.ndarray shape (t+1, 5): open, high, low, close, volume columns
- `view.prices_matrix()` -> np.ndarray shape (t+1, N): all assets' close prices

### CoinGlass Derivatives Data (if loaded)

- `view.has_cg()` -> bool: whether CoinGlass data is available
- `view.cg_fields(asset)` -> list[str]: available field names for an asset
- `view.cg(asset, field)` -> np.ndarray shape (t+1,): causally aligned feature

Available fields (when CoinGlass API key is configured):
- `oi_1h`, `oi_1d` -- aggregated open interest close (hourly/daily)
- `funding_1h` -- funding rate close (hourly)
- `liq_long_1h` -- long liquidation USD (hourly)
- `liq_short_1h` -- short liquidation USD (hourly)
- `ls_ratio_1h` -- global long/short account ratio (hourly)

Each value is from the **last fully completed** candle. NaN where no completed candle exists.

All returned arrays are **copies**. No future data is accessible.

## Strategy File Requirements

Your strategy file MUST start with:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
Then import from the local `playground/` package:
```python
import numpy as np
from playground.featurizer import Featurizer, Predictor
from playground.data import CausalView
```

You must define:
- `TechFeaturizer(Featurizer)` with a `compute()` method
- `TechPredictor(Predictor)` with a `predict()` method

## Available Libraries

numpy, scipy, sklearn, pandas, and standard library modules.

## Dev / Eval Split

Data is split into a **dev** period and an **eval holdout** (last 20 days).
- You explore features and compute correlations using DEV data only.
- Walk-forward evaluation runs on ALL data; the test windows fall in the holdout period.
  The walk-forward metrics (the primary metric, etc.) ARE your real out-of-sample scores.
  These are the numbers you should trust, report, and use to judge what works vs what doesn't.
- `dev_emb_dim_N_corr` values are raw dev-period hints, NOT performance metrics. Ignore them
  when deciding if a strategy is good -- only the walk-forward holdout metrics matter.
- You must NEVER manually inspect, analyze, or compute statistics on the eval period.
- If you write custom analysis scripts, truncate data to exclude the last 28800 minutes.

## Custom Data (Advanced)

You can fetch data from **any external API** and make it available in the evaluator.
Place arrays in `custom_data/` in your workspace:

**Option A -- per-asset directory:**
```
custom_data/BTC/my_sentiment.npy    # np.save(path, arr)
custom_data/ETH/my_indicator.npy
```

**Option B -- per-asset npz:**
```
custom_data/BTC.npz    # np.savez(path, my_field1=arr1, my_field2=arr2)
```

Each array must be 1D float64, length matching the OHLCV data (1-minute resolution).
The eval script automatically merges these into the CoinGlass layer. Access via:
```python
view.cg(asset, "my_sentiment")  # your custom field, causally sliced
```

**CRITICAL: YOU must ensure causal alignment.** Each value at minute `t` must only use
data available at time `t`. Use last-completed-candle logic (e.g., for hourly data, the
value at minute 61 is from the candle that closed at minute 60, NOT the candle closing
at minute 120). Misaligned data = lookahead bias = fake performance.

The dashboard flags iterations using custom data with a prominent warning banner.
Only use this when the pre-cached data lacks fields you need.

## File Structure

Your workspace contains:
- `playground/` -- the framework source code (read-only reference)
- `API_REFERENCE.md` -- this file
- `GOAL.md` -- your research goal
- `STATE.md` -- your persistent memory (read + update every iteration)
- `HISTORY.md` -- previous iteration results
- `INSTRUCTIONS.md` -- what to do this iteration
- `INBOX.md` -- messages from the human operator (if any)
- `_eval_N.py` -- evaluation scripts (do NOT modify)
- `iteration_N.py` -- your strategy files (you write these)
- `_result_N.json` -- evaluation results (eval writes these)
- `iteration_N_analysis.json` -- your analysis (you write these)
- `custom_data/` -- optional: your custom data arrays (you create this)
""")


def _write_goal(workspace, goal):
    (workspace / "GOAL.md").write_text(f"# Research Goal\n\n{goal}\n")


def _write_history(workspace, challenge, prev_results):
    """Write HISTORY.md with previous iteration results."""
    _, _, metric_key, _, all_metrics = _challenge_info(challenge)
    parts = ["# Previous Iterations\n"]

    if not prev_results:
        parts.append("No iterations completed yet -- this is the first.\n")
    else:
        for r in prev_results:
            parts.append(f"\n## Iteration {r['iteration']}")
            if r.get("has_error"):
                parts.append(f"**ERROR**: {str(r['metrics'].get('error', 'unknown'))[:500]}")
                parts.append("")
                continue

            m = r["metrics"]
            parts.append("**Metrics:**")
            for k in all_metrics:
                v = m.get(k)
                if v is not None:
                    marker = " <-- PRIMARY" if k == metric_key else ""
                    parts.append(f"- {k}: {v:.4f}{marker}" if isinstance(v, float) else f"- {k}: {v}{marker}")

            windows = m.get("windows", [])
            if windows:
                parts.append(f"- walk-forward windows: {len(windows)}")

            analysis = r.get("analysis")
            if analysis and isinstance(analysis, dict):
                if analysis.get("hypothesis"):
                    parts.append(f"\n**Hypothesis:** {analysis['hypothesis'][:300]}")
                if analysis.get("what_changed"):
                    parts.append(f"**Changes:** {analysis['what_changed'][:300]}")
                if analysis.get("what_worked"):
                    parts.append(f"**Worked:** {analysis['what_worked'][:300]}")
                if analysis.get("what_failed"):
                    parts.append(f"**Failed:** {analysis['what_failed'][:300]}")
                feat_report = analysis.get("feature_report", analysis.get("features", []))
                if feat_report:
                    parts.append("\n**Feature Report:**")
                    for f in feat_report[:8]:
                        fname = f.get("name", "?")
                        desc = f.get("description", "")
                        pp = f.get("predictive_power", {})
                        pp_str = ""
                        if pp:
                            pp_str = f" | {pp.get('method', '?')}={pp.get('value', '?')}"
                            if pp.get("p_value") is not None:
                                pp_str += f" (p={pp['p_value']:.4g})" if isinstance(pp["p_value"], float) else ""
                        parts.append(f"- `{fname}`: {desc[:100]}{pp_str}")

            parts.append("")

    (workspace / "HISTORY.md").write_text("\n".join(parts))


def _write_eval_script(workspace, challenge, iteration, days_back, coinglass_key=None,
                       holdout_days=EVAL_HOLDOUT_DAYS, data_cache_dir=None):
    """Write the evaluation script that Claude will run.

    The script splits data into dev (first N-holdout) and eval (last holdout).
    Walk-forward windows span all data, but only holdout-period windows
    contribute to the reported headline metrics (mean_auc etc.).
    Feature analysis is the agent's responsibility (written in analysis JSON).
    """
    if not (isinstance(challenge, str) and challenge.replace("-", "").isalnum()):
        raise ValueError(f"Bad challenge: {challenge}")
    if not (isinstance(iteration, int) and 0 < iteration < 10000):
        raise ValueError(f"Bad iteration: {iteration}")
    if not (isinstance(days_back, int) and 0 < days_back < 3650):
        raise ValueError(f"Bad days_back: {days_back}")

    strategy_file = f"iteration_{iteration}.py"
    result_file = f"_result_{iteration}.json"
    eval_path = workspace / f"_eval_{iteration}.py"
    ws = str(workspace)

    if data_cache_dir:
        cache_abs = str(data_cache_dir).replace("\\", "\\\\").replace('"', '\\"')
        fetch_block = f'''# Load from prefetched cache (shared across all agents)
import pandas as _pd
_cache_dir = os.environ.get("MANTIS_DATA_CACHE", "{cache_abs}")
data_all = {{}}
for _asset in config.assets:
    _p = os.path.join(_cache_dir, "ohlcv", _asset + ".parquet")
    if os.path.exists(_p):
        data_all[_asset] = _pd.read_parquet(_p)
if not data_all:
    raise RuntimeError("Prefetched data cache empty -- run data_cache.py first")
coinglass_data = {{}}
_cg_dir = os.path.join(_cache_dir, "coinglass")
if os.path.isdir(_cg_dir):
    for _asset in config.assets:
        _cp = os.path.join(_cg_dir, _asset + ".npz")
        if os.path.exists(_cp):
            with np.load(_cp) as _npz:
                coinglass_data[_asset] = {{k: _npz[k] for k in _npz.files}}
if coinglass_data:
    full_provider = DataProvider(data_all, coinglass=coinglass_data)
else:
    full_provider = DataProvider(data_all)
print(f"Loaded {{len(data_all)}} assets from cache: {{_cache_dir}}")'''
    elif coinglass_key:
        fetch_block = f'''cg_key = os.environ.get("COINGLASS_API_KEY", "")
result_fetch = fetch_assets(assets=config.assets, interval="1m", days_back={days_back}, coinglass_api_key=cg_key)
data_all, coinglass_data = result_fetch
full_provider = DataProvider(data_all, coinglass=coinglass_data)'''
    else:
        fetch_block = f'''data_all, _ = fetch_assets(assets=config.assets, interval="1m", days_back={days_back})
full_provider = DataProvider(data_all)'''

    holdout_minutes = holdout_days * 1440

    eval_path.write_text(f'''import sys, os, json
_ws = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ws)
import numpy as np
from playground import evaluate, DataProvider, fetch_assets
from playground.evaluator import CHALLENGES, _generate_embeddings
import importlib.util

config = CHALLENGES["{challenge}"]

# ── Fetch all data ──
{fetch_block}

# ── Custom data loading ──
_custom_data_used = False
_custom_data_fields = []
_custom_dir = os.path.join(_ws, "custom_data")
if os.path.isdir(_custom_dir):
    if "coinglass_data" not in dir():
        coinglass_data = {{}}
    for _asset_dir in sorted(os.listdir(_custom_dir)):
        _asset_path = os.path.join(_custom_dir, _asset_dir)
        if os.path.isdir(_asset_path):
            if _asset_dir not in coinglass_data:
                coinglass_data[_asset_dir] = {{}}
            for _f in sorted(os.listdir(_asset_path)):
                _fp = os.path.join(_asset_path, _f)
                if _f.endswith(".npy"):
                    _field = _f[:-4]
                    coinglass_data[_asset_dir][_field] = np.load(_fp)
                    _custom_data_fields.append(f"{{_asset_dir}}/{{_field}}")
                    _custom_data_used = True
        elif _asset_dir.endswith(".npz"):
            _asset_name = _asset_dir[:-4]
            if _asset_name not in coinglass_data:
                coinglass_data[_asset_name] = {{}}
            with np.load(_asset_path) as _npz:
                for _k in _npz.files:
                    coinglass_data[_asset_name][_k] = _npz[_k]
                _custom_data_fields.append(f"{{_asset_name}}/{{_k}}")
                _custom_data_used = True
    if _custom_data_used:
        print(f"CUSTOM DATA LOADED: {{len(_custom_data_fields)}} fields: {{_custom_data_fields}}")
        full_provider = DataProvider(data_all, coinglass=coinglass_data)

# ── Dev / Eval split ──
# Last {holdout_days} days ({holdout_minutes} minutes) are held out for evaluation.
# Feature development uses DEV data ONLY.
# Walk-forward windows span all data but only holdout windows count for headline metrics.
holdout_minutes = {holdout_minutes}
total_len = full_provider.length
dev_len = max(0, total_len - holdout_minutes)
print(f"Total data: {{total_len}} minutes, Dev: {{dev_len}}, Eval holdout: {{total_len - dev_len}}")

dev_data = {{}}
for asset in data_all:
    dev_data[asset] = data_all[asset].iloc[:dev_len].copy()
if "coinglass_data" in dir() and coinglass_data:
    _dev_cg = {{}}
    for _ca, _cf in coinglass_data.items():
        _dev_cg[_ca] = {{k: v[:dev_len] for k, v in _cf.items()}}
    dev_provider = DataProvider(dev_data, coinglass=_dev_cg)
else:
    dev_provider = DataProvider(dev_data)

# ── Load strategy ──
strategy_path = os.path.join(_ws, "{strategy_file}")
spec = importlib.util.spec_from_file_location("strat", strategy_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

featurizer = mod.TechFeaturizer()
predictor = mod.TechPredictor()

# ── Walk-forward evaluation on FULL data, scored on HOLDOUT windows only ──
result = evaluate("{challenge}", featurizer, predictor, provider=full_provider,
                  holdout_start=dev_len)

result["dev_length"] = dev_len
result["eval_holdout_minutes"] = total_len - dev_len
result["custom_data_used"] = _custom_data_used
if _custom_data_used:
    result["custom_data_fields"] = _custom_data_fields

import math as _math
def _nan_safe(o):
    if isinstance(o, float) and (_math.isnan(o) or _math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {{k: _nan_safe(v) for k, v in o.items()}}
    if isinstance(o, (list, tuple)):
        return [_nan_safe(v) for v in o]
    return o
result = _nan_safe(result)
result_path = os.path.join(_ws, "{result_file}")
with open(result_path, "w") as f:
    json.dump(result, f, indent=2, default=str)

# Push results directly into log.json so the frontend picks them up immediately
import datetime as _dt, tempfile as _tempfile, fcntl as _fcntl
_log_path = os.path.join(_ws, "..", "log.json")
_lock_path = _log_path + ".lock"
if os.path.exists(_log_path):
    _lk_fd = os.open(_lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _fcntl.flock(_lk_fd, _fcntl.LOCK_EX)
        with open(_log_path, "r") as _lf: _raw_log = _lf.read().strip()
        try:
            _log = json.loads(_raw_log) if _raw_log else {{}}
        except (json.JSONDecodeError, ValueError):
            _log = {{}}
        _display_metrics = {{k: v for k, v in result.items() if k not in ("windows",)}}
        _entry = {{
            "iteration": {iteration},
            "timestamp": _dt.datetime.now().isoformat(),
            "metrics": _display_metrics,
            "analysis": None,
            "activity": [],
            "thoughts": [],
            "tokens": {{"input": 0, "output": 0, "cost": 0}},
            "elapsed_s": 0,
            "result_text": "",
            "code_path": "{strategy_file}" if os.path.exists(os.path.join(_ws, "{strategy_file}")) else None,
            "has_error": "error" in result,
            "done_signal": False,
            "timed_out": False,
            "custom_data_used": _custom_data_used,
            "custom_data_fields": _custom_data_fields if _custom_data_used else [],
            "_partial": True,
        }}
        _existing = [e for e in _log.get("iterations", []) if e.get("iteration") != {iteration}]
        _existing.append(_entry)
        _log["iterations"] = _existing
        _log["last_updated"] = _dt.datetime.now().isoformat()
        _log_dir = os.path.dirname(_log_path)
        _tfd, _tmp = _tempfile.mkstemp(dir=_log_dir, suffix=".tmp")
        with os.fdopen(_tfd, "w") as _wf:
            json.dump(_log, _wf, indent=2, default=str)
        os.replace(_tmp, _log_path)
        print(">> Results pushed to log.json for frontend")
    finally:
        _fcntl.flock(_lk_fd, _fcntl.LOCK_UN)
        os.close(_lk_fd)

print("=== EVALUATION RESULTS ===")
print(f"  dev_length: {{dev_len}} minutes ({{dev_len // 1440}} days)")
print(f"  eval_holdout: {{total_len - dev_len}} minutes ({{(total_len - dev_len) // 1440}} days)")
for k, v in sorted(result.items()):
    if k not in ("windows",):
        if isinstance(v, float):
            print(f"  {{k}}: {{v:.6f}}")
        else:
            print(f"  {{k}}: {{v}}")
print(f"\\nFull results written to: {result_file}")
''')
    return eval_path


def _write_instructions(workspace, challenge, iteration, min_iter,
                        days_back=60, holdout_days=EVAL_HOLDOUT_DAYS):
    """Write INSTRUCTIONS.md for the current iteration."""
    dim, fmt, metric_key, metric_dir, all_metrics = _challenge_info(challenge)
    ws = str(workspace)

    instructions = f"""# MANTIS Agentic Mining - Iteration {iteration}

## Challenge: {challenge}
- Embedding dimension: {dim}
- Embedding format: {fmt}
- Primary metric: {metric_key} ({metric_dir})
- All reported metrics: {', '.join(all_metrics)}

## Data Split: Dev / Eval

The dataset covers {days_back} days. The last **{holdout_days} days are held out** as evaluation data.

**How evaluation works:**
- The eval script `_eval_{iteration}.py` runs walk-forward evaluation on the FULL dataset.
  The walk-forward framework trains on past data and tests on future windows.
  The test windows fall into the holdout period that you never saw.
- The primary metric (`{metric_key}`) and all walk-forward metrics in `_result_{iteration}.json`
  are **out-of-sample scores from the holdout period**. These are the REAL performance numbers.
  Judge your strategy by THESE metrics. Report THESE numbers in your analysis.
- `dev_emb_dim_N_corr` values are raw embedding-vs-return correlations from the DEV period.
  They are weak exploratory hints at best. Do NOT treat them as real performance.
  A feature can show high dev correlation yet fail in walk-forward (spurious), or show
  low dev correlation yet help when combined with other features in walk-forward.

**CRITICAL RULES about the eval holdout:**
- You MUST NOT manually inspect, analyze, compute correlations on, or load the eval period.
- All your feature exploration, correlation analysis, sanity checks, and manual testing
  must use ONLY the first {days_back - holdout_days}+ days (the "dev" period).
- If you write any analysis scripts yourself, you MUST truncate the data
  to exclude the last {holdout_days * 1440} minutes before computing anything.
- The purpose: spurious correlations found in dev will NOT hold up in the walk-forward
  evaluation (which tests on the holdout). If you peek at eval data, you defeat this safeguard.
- The ONLY way to get holdout performance is by running `_eval_{iteration}.py` and reading
  the walk-forward metrics from `_result_{iteration}.json`. Trust those numbers.

## Your Workspace

Your workspace is: `{ws}/`
ALL files you read and write MUST be within this directory.
Do NOT write files outside this workspace.

## Persistent State (`STATE.md`)

`STATE.md` persists across ALL iterations. Use it to track what you've learned,
your best-performing approaches, key insights, and plans for future iterations.
Read it at the start of each iteration. Update it at the end with your latest findings.
This is your long-term memory between iterations.

## Operator Messages (`INBOX.md`)

If `INBOX.md` has content, the human operator has sent you a message. Read it carefully
and follow any instructions. The operator can change your goal, give hints, or ask
questions. Address their message in your notes and adapt your approach accordingly.

## Live Notes (`notes.txt`) -- YOUR FIRST ACTION

`notes.txt` is displayed on the frontend in real-time (polled every 3 seconds).
A human is watching your progress live. Writing to `notes.txt` is the FIRST thing
you do every iteration, BEFORE writing any code.

## Your Workflow

1. **Read context** (skim quickly -- do not over-read):
   - `GOAL.md`, `STATE.md`, `HISTORY.md`, `API_REFERENCE.md`
   - If `INBOX.md` has content, read it carefully and prioritize the operator's instructions

2. **IMMEDIATELY write `notes.txt`** (this MUST be your first file write):
   ```
   --- iteration {iteration} ---
   Hypothesis: [what signal you're trying to capture and why]
   Planned features:
     - feature_name: [what it measures] -> [why it predicts target]
     - feature_name: [what it measures] -> [why it predicts target]
     - ...
   Approach: [model type, key design decisions]
   What changed from last iteration: [or "first iteration"]
   ```
   This shows up on the dashboard within seconds. Do this BEFORE writing strategy code.

3. **Write your strategy** to `iteration_{iteration}.py`, then append to `notes.txt`:
   ```
   Writing strategy code... [brief description of what you built]
   ```
   - Must start with:
     ```python
     import sys, os
     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
     ```
   - Import from local playground:
     ```python
     import numpy as np
     from playground.featurizer import Featurizer, Predictor
     from playground.data import CausalView
     ```
   - Must define `TechFeaturizer(Featurizer)` with `compute()` method
   - Must define `TechPredictor(Predictor)` with `predict()` returning shape ({dim},)
   - You can use numpy, scipy, sklearn, pandas, and standard library
   - If CoinGlass data is available, use `view.cg(asset, field)` for derivatives features
     (check with `view.has_cg()` and `view.cg_fields(asset)` first)

   **Custom Data (optional):** You can fetch data from external APIs yourself and make it
   available to the evaluator. Place arrays in `{ws}/custom_data/` using one of:
   - Per-asset directory: `custom_data/<ASSET>/<field_name>.npy` -- each `.npy` is a 1D float64
     array of length matching the OHLCV data (1-minute resolution).
   - Per-asset npz: `custom_data/<ASSET>.npz` -- keys become field names.
   The eval script will merge these into the DataProvider's CoinGlass layer, accessible via
   `view.cg(asset, field)`. **YOU are responsible for causal alignment** -- each value at
   minute t must only reflect data available at time t (use last-completed-candle logic).
   Misalignment = lookahead bias. The dashboard will flag iterations using custom data with
   a prominent warning. Use this only when the pre-cached data lacks fields you need.

4. **Append to `notes.txt`**: "Running evaluation..." then **run eval**:
   ```
   cd {ws} && python3 _eval_{iteration}.py
   ```

5. **Read results** from `_result_{iteration}.json`:
   - If errors, append to `notes.txt` what went wrong, fix, re-run
   - Walk-forward metrics ({metric_key}, etc.) ARE your holdout scores -- ground truth
   - `dev_emb_dim_N_corr` are just rough dev-period hints, do NOT judge by them
   - Successful eval pushes results to frontend immediately
   - Append key results to `notes.txt`:
     ```
     Results: {metric_key} = [value], [other key metrics]
     Interpretation: [what this means, what worked, what to try next]
     ```

6. **Write analysis** to `iteration_{iteration}_analysis.json`:
   ```json
   {{
     "feature_report": [
       {{
         "name": "feature_dict_key",
         "description": "What this feature physically measures",
         "signal": "Why it should predict the target",
         "stats": {{
           "mean": 0.001,
           "std": 0.023,
           "min": -0.12,
           "max": 0.15
         }},
         "predictive_power": {{
           "method": "spearman vs fwd returns",
           "value": 0.05,
           "p_value": 0.02,
           "n_samples": 500,
           "interpretation": "Weak but statistically significant"
         }},
         "importance_rank": 1
       }}
     ],
     "inference_spec": {{
       "summary": "One-paragraph plain-English description of what the strategy does",
       "data_requirements": [
         {{
           "field": "close",
           "source": "Binance OHLCV",
           "granularity": "1m",
           "lookback": "200 candles",
           "description": "Closing prices for momentum calculation"
         }}
       ],
       "update_frequency": "Every 1 minute",
       "latency_sensitivity": "Low -- uses 10m+ lookback windows",
       "output_format": "2-element array [p_up, p_down] summing to 1.0",
       "dependencies": ["numpy", "scipy"]
     }},
     "hypothesis": "Core thesis of this strategy",
     "what_changed": "What changed from previous iteration (null if first)",
     "what_worked": "What showed promise (null if first)",
     "what_failed": "What didn't work (null if first)",
     "expected_outcome": "Expected next-iteration {metric_key} with reasoning",
     "log": ["Step-by-step reasoning notes..."],
     "done": false
   }}
   ```

   **Feature report guidelines:**
   - You MUST include a `feature_report` array describing every feature in your strategy.
   - For each feature, compute summary statistics (mean, std, min, max) on the DEV set.
   - Run your own predictive analysis: correlate features against forward returns or labels
     on the DEV set (first {days_back - holdout_days}+ days ONLY). Use scipy.stats for
     Spearman/Pearson correlation, AUC, or whatever test is appropriate for the challenge type.
   - Report your interpretation: is the feature working? Is it redundant with another?
   - The `stats` and `predictive_power` fields are displayed in the dashboard's Features tab.
     Be thorough -- this replaces any automated analysis.

   **Inference spec guidelines (REQUIRED):**
   - You MUST include an `inference_spec` object describing what the strategy needs for live deployment.
   - `data_requirements`: list every data feed the featurizer consumes at each timestamp.
     Include the field name, source (Binance OHLCV, CoinGlass, custom), granularity,
     lookback depth (how many candles/minutes of history), and a brief description.
   - `update_frequency`: how often the strategy should be called (e.g. "Every 1 minute").
   - `output_format`: what the predictor returns (shape, meaning of each element).
   - `dependencies`: Python packages required beyond numpy (e.g. scipy, sklearn).
   - `summary`: a clear plain-English paragraph explaining the strategy end-to-end.
   - This is displayed in the dashboard's Deploy tab so a human can wire it into production.

   Then EXIT immediately. The system starts the next iteration automatically.

7. **Update `STATE.md`** with your latest findings, best approach so far,
   key insights, and plans for next iteration. This file persists across iterations.

## Rules
- ALL file operations must be within `{ws}/`
- Do NOT modify `_eval_{iteration}.py` or any files in `playground/`
- NEVER look at, analyze, or compute statistics on the last {holdout_days} days of data
- Walk-forward metrics from eval are your REAL scores. Report them.
- Do NOT report dev correlations as performance metrics
- Set `"done": true` only after {min_iter}+ iterations AND diminishing returns
- Be specific in notes -- a human reads them live
- If an approach fails, try something fundamentally different next iteration
- **NOTES FIRST**: `notes.txt` must be the first file you write every iteration
- **SPEED**: After successful eval, write analysis JSON and EXIT. No lingering.
"""
    (workspace / "INSTRUCTIONS.md").write_text(instructions)


# ── Iteration execution ─────────────────────────────────────────────────────

def _run_iteration(agent_id, config, iteration, prev_results, state, log_path,
                    agent_dir=None):
    """Execute one Claude Code iteration with full tool access."""
    if agent_dir is None:
        agent_dir = AGENTS_DIR / agent_id
    workspace = _setup_workspace(agent_dir)
    challenge = config["challenge"]
    days_back = max(config.get("days_back", MIN_DAYS_BACK), MIN_DAYS_BACK)
    min_iter = config.get("min_iterations", 5)
    model = config.get("model", "sonnet")

    _write_claude_settings(workspace, model=model)
    _write_api_reference(workspace)
    _write_goal(workspace, config["goal"])
    _write_history(workspace, challenge, prev_results)
    coinglass_key = config.get("coinglass_api_key") or os.environ.get("COINGLASS_API_KEY")

    from playground.data_cache import is_cached, cache_dir_for
    data_cache_dir = str(cache_dir_for(days_back)) if is_cached(days_back) else None
    _write_eval_script(workspace, challenge, iteration, days_back,
                       coinglass_key=coinglass_key, data_cache_dir=data_cache_dir)
    _write_instructions(workspace, challenge, iteration, min_iter, days_back=days_back)

    ws = str(workspace)

    state_path = workspace / "STATE.md"
    if not state_path.exists():
        state_path.write_text("# Agent State\n\n_No state yet -- first iteration._\n")

    inbox_hint = ""
    inbox_path = workspace / "INBOX.md"
    inbox_content = ""
    if inbox_path.exists():
        inbox_content = inbox_path.read_text().strip()
    if inbox_content:
        inbox_hint = (
            f" IMPORTANT: Read INBOX.md -- it contains a message from the human operator. "
            f"Prioritize any instructions or guidance in INBOX.md for this iteration."
        )
        _append_chat(agent_dir, "system",
                     f"Message delivered to agent for iteration {iteration}")

    prompt = (
        f"Read INSTRUCTIONS.md in {ws}/ for iteration {iteration}. Read STATE.md for your "
        f"persistent memory. Skim GOAL.md and HISTORY.md quickly.{inbox_hint} Your FIRST "
        f"file write must be notes.txt with your hypothesis, planned features, and approach "
        f"-- a human watches the dashboard live. Then write strategy code, run eval, write "
        f"analysis JSON, update STATE.md with findings, and EXIT. "
        f"ALL file operations must be within {ws}/ only."
    )

    cmd = _claude_cmd(prompt)
    env = _claude_env()

    activities = []
    thoughts = []
    def _on_activity(msg):
        activities.append(msg)
        print(f"    {msg}", flush=True)
    def _on_text(text):
        snippet = text.strip()[:500]
        if snippet:
            thoughts.append(snippet)

    t0 = time.monotonic()
    print(f"[iter {iteration}] calling claude ({model})...", flush=True)

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        returncode, result_text, tokens_in, tokens_out, cost_usd, timed_out = _run_claude_once(
            cmd, env, cwd=workspace, on_activity=_on_activity, on_text=_on_text,
        )
        if returncode == 0 or attempt == max_retries or _stop_event.is_set():
            break
        print(f"[iter {iteration}] claude failed (rc={returncode}), retrying ({attempt}/{max_retries})...",
              flush=True)
        time.sleep(5)

    elapsed = time.monotonic() - t0

    print(f"[iter {iteration}] claude finished in {elapsed:.1f}s "
          f"(rc={returncode}, {tokens_in}in/{tokens_out}out, ${cost_usd:.4f})", flush=True)

    strategy_path = workspace / f"iteration_{iteration}.py"
    result_path = workspace / f"_result_{iteration}.json"
    analysis_path = workspace / f"iteration_{iteration}_analysis.json"

    metrics = {}
    custom_data_used = False
    custom_data_fields = []
    if result_path.exists():
        raw_text = result_path.read_text().strip()
        try:
            raw = json.loads(raw_text) if raw_text else {}
        except (json.JSONDecodeError, ValueError):
            raw = {"error": "malformed result JSON"}
        raw.pop("feature_stats", None)
        raw.pop("feature_analysis", None)
        custom_data_used = raw.pop("custom_data_used", False)
        custom_data_fields = raw.pop("custom_data_fields", [])
        metrics = raw
    elif timed_out:
        metrics = {"error": f"iteration killed: exceeded {ITER_WALL_TIMEOUT}s wall-clock limit"}
    else:
        metrics = {"error": "evaluation did not produce results"}

    analysis = None
    if analysis_path.exists():
        raw_a = analysis_path.read_text().strip()
        try:
            analysis = json.loads(raw_a) if raw_a else None
        except (json.JSONDecodeError, ValueError):
            analysis = None

    has_error = "error" in metrics
    done_signal = False
    if analysis and isinstance(analysis, dict):
        done_signal = analysis.get("done", False) and iteration >= min_iter

    entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "analysis": analysis,
        "activity": activities[-50:],
        "thoughts": thoughts[-20:],
        "tokens": {"input": tokens_in, "output": tokens_out, "cost": round(cost_usd, 4)},
        "elapsed_s": round(elapsed, 1),
        "result_text": result_text[:5000] if result_text else "",
        "code_path": f"iteration_{iteration}.py" if strategy_path.exists() else None,
        "has_error": has_error,
        "done_signal": done_signal,
        "timed_out": timed_out,
        "custom_data_used": custom_data_used,
        "custom_data_fields": custom_data_fields,
    }

    if inbox_content:
        inbox_path.write_text("")

    if not has_error:
        _, _, mk, _, _ = _challenge_info(challenge)
        v = metrics.get(mk)
        print(f"[iter {iteration}] {mk} = {v}", flush=True)
    else:
        print(f"[iter {iteration}] ERROR: {metrics.get('error', '')[:200]}", flush=True)

    return entry


# ── Signal handling ──────────────────────────────────────────────────────────

def _sigterm_handler(signum, frame):
    _stop_event.set()
    if _child_procs_lock.acquire(blocking=False):
        try:
            for proc in _child_procs:
                if proc.poll() is None:
                    proc.kill()
        finally:
            _child_procs_lock.release()


# ── State management ─────────────────────────────────────────────────────────

def _sanitize(obj):
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _save_state(state, log_path):
    import tempfile, fcntl
    state["last_updated"] = datetime.now().isoformat()
    data = json.dumps(_sanitize(state), indent=2, default=str)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = log_path.parent / (log_path.name + ".lock")
    lock_fd = -1
    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        fd, tmp = tempfile.mkstemp(dir=str(log_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, str(log_path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


# ── Chat helpers ─────────────────────────────────────────────────────────────

def _chat_path(agent_dir):
    return agent_dir / "chat.json"


def _load_chat(agent_dir):
    p = _chat_path(agent_dir)
    if p.exists():
        raw = p.read_text().strip()
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
    return []


def _append_chat(agent_dir, role, text):
    import tempfile, fcntl
    p = _chat_path(agent_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.parent / (p.name + ".lock")
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        msgs = []
        if p.exists():
            raw = p.read_text().strip()
            if raw:
                try:
                    msgs = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    msgs = []
        msgs.append({
            "role": role,
            "text": text,
            "ts": datetime.now().isoformat(),
        })
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(msgs, indent=2))
        os.replace(tmp, str(p))
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _check_pause(agent_dir):
    """Check if a PAUSE file exists (written by the GUI when user sends a message)."""
    return (agent_dir / "PAUSE").exists()


PAUSE_TIMEOUT_SECONDS = int(os.environ.get("MANTIS_PAUSE_TIMEOUT", "3600"))


def _wait_for_unpause(agent_id, agent_dir, state, log_path):
    """Block until PAUSE file is removed or timeout expires."""
    state["status"] = "paused"
    _save_state(state, log_path)
    print(f"[agent] paused -- waiting for resume (timeout {PAUSE_TIMEOUT_SECONDS}s)...", flush=True)
    waited = 0
    while (agent_dir / "PAUSE").exists():
        if _stop_event.is_set() or (agent_dir / "STOP").exists():
            return
        if waited >= PAUSE_TIMEOUT_SECONDS:
            print(f"[agent] pause timeout after {waited}s, auto-resuming", flush=True)
            try:
                (agent_dir / "PAUSE").unlink(missing_ok=True)
            except OSError:
                pass
            break
        time.sleep(1)
        waited += 1
    state["status"] = "running"
    _save_state(state, log_path)
    print(f"[agent] resumed", flush=True)


# ── Main agent loop ──────────────────────────────────────────────────────────

def run_agent(agent_id, config, agent_dir_override=None):
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    from playground.evaluator import CHALLENGES
    challenge = config.get("challenge", "")
    if challenge not in CHALLENGES:
        raise ValueError(f"Unknown challenge: {challenge!r}")

    agent_dir = Path(agent_dir_override) if agent_dir_override else AGENTS_DIR / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    log_path = agent_dir / "log.json"
    pid_path = agent_dir / "pid"
    stop_path = agent_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    pid_path.write_text(str(os.getpid()))

    if log_path.exists():
        raw_state = log_path.read_text().strip()
        try:
            state = json.loads(raw_state) if raw_state else {}
        except (json.JSONDecodeError, ValueError):
            state = {}
        state["iterations"] = [
            e for e in state.get("iterations", []) if not e.get("_partial")
        ]
    else:
        state = {
            "id": agent_id,
            "config": config,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "iterations": [],
        }

    state["status"] = "running"
    _save_state(state, log_path)

    days_back = max(config.get("days_back", MIN_DAYS_BACK), MIN_DAYS_BACK)
    from playground.data_cache import prefetch, is_cached
    if not is_cached(days_back):
        print(f"[prefetch] Fetching {days_back}d data for all assets...", flush=True)
        coinglass_key = os.environ.get("COINGLASS_API_KEY", "")
        prefetch(days_back, coinglass_api_key=coinglass_key)
        print("[prefetch] Done.", flush=True)
    from playground.data_cache import cache_dir_for
    os.environ["MANTIS_DATA_CACHE"] = str(cache_dir_for(days_back))

    max_iters = config.get("max_iterations", 20)

    def _should_stop():
        return _stop_event.is_set() or stop_path.exists()

    try:
        for i in range(len(state["iterations"]), max_iters):
            if _should_stop():
                state["status"] = "stopped"
                break

            if _check_pause(agent_dir):
                _wait_for_unpause(agent_id, agent_dir, state, log_path)
                if _should_stop():
                    state["status"] = "stopped"
                    break

            iteration = i + 1
            state["current_iteration"] = iteration
            _save_state(state, log_path)

            stopped = False
            entry = None
            try:
                entry = _run_iteration(
                    agent_id, config, iteration,
                    state["iterations"], state, log_path,
                    agent_dir=agent_dir,
                )
            except _StopRequested:
                stopped = True
            except Exception as exc:
                print(f"[iter {iteration}] FATAL: {exc}", flush=True)
                state["status"] = "crashed"
                _save_state(state, log_path)
                break

            if stopped or _should_stop():
                state["status"] = "stopped"
                _save_state(state, log_path)
                break

            state["iterations"].append(entry)
            _save_state(state, log_path)

            if _check_pause(agent_dir):
                _wait_for_unpause(agent_id, agent_dir, state, log_path)
                if _should_stop():
                    state["status"] = "stopped"
                    _save_state(state, log_path)
                    break

            if entry.get("done_signal"):
                state["status"] = "completed"
                print(f"[iter {iteration}] agent signalled DONE", flush=True)
                break
        else:
            if _should_stop():
                state["status"] = "stopped"
            else:
                state["status"] = "max_iterations_reached"

        _save_state(state, log_path)
    finally:
        pid_path.unlink(missing_ok=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MANTIS Agentic Mining agent runner")
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--days-back", type=int, default=60)
    parser.add_argument("--coinglass-key", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--agent-dir", default=None,
                        help="Override agent directory (used in Docker)")
    parser.add_argument("--data-dir", default=None,
                        help="Override data cache root (used in Docker)")
    args = parser.parse_args()

    if args.agent_dir:
        os.environ["MANTIS_AGENT_DIR"] = args.agent_dir
    if args.data_dir:
        os.environ["MANTIS_DATA_DIR"] = args.data_dir

    agent_id = args.id or str(uuid.uuid4())[:8]
    config = {
        "challenge": args.challenge,
        "goal": args.goal,
        "min_iterations": args.min_iterations,
        "max_iterations": args.max_iterations,
        "model": args.model,
        "days_back": args.days_back,
    }
    cg_key = args.coinglass_key or os.environ.get("COINGLASS_API_KEY")
    if cg_key:
        os.environ["COINGLASS_API_KEY"] = cg_key

    print(f"Agent {agent_id} | {args.challenge} | {args.goal}")
    print(f"Iterations: {args.min_iterations}-{args.max_iterations} | Model: {args.model}")
    print(flush=True)

    result = run_agent(agent_id, config, agent_dir_override=args.agent_dir)
    print(f"\nFinished: {result['status']} ({len(result['iterations'])} iterations)")
