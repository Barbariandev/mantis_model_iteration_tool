# MANTIS Agentic Mining Playground

Autonomous AI agents that research, code, and evaluate cryptocurrency prediction signal strategies. Each agent runs Claude Code inside a sandboxed Docker container, iteratively building features, training models, and evaluating them with walk-forward backtesting and strict causal data access.

## Prerequisites

- Python 3.10+
- Docker (running, current user must have `docker` permissions)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`)
- Anthropic API key
- Network access to Binance API (OHLCV data) and Anthropic API; CoinGlass API key optional

## Installation

```bash
pip install flask gunicorn numpy pandas scipy scikit-learn requests pyarrow
```

Or from the parent directory's `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Starting the server

```bash
# From the parent directory of model_iteration_tool/
python -m model_iteration_tool.gui
```

The server starts at `http://127.0.0.1:8420`. Open it in a browser for the full GUI.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MANTIS_HOST` | `127.0.0.1` | Bind address |
| `MANTIS_PORT` | `8420` | Bind port |
| `MANTIS_WORKERS` | `2` | Gunicorn worker count |
| `MANTIS_AUTH_TOKEN` | _(none)_ | Set to enable token auth (required for non-localhost access) |
| `MANTIS_MAX_AGENTS` | `5` | Max concurrent agent containers |
| `MANTIS_CLAUDE_TIMEOUT` | `600` | Seconds before killing a Claude Code invocation |
| `MANTIS_RATE_LIMIT_RPM` | `120` | API rate limit per IP (0 to disable) |
| `MANTIS_LOG_LEVEL` | `INFO` | Python logging level |
| `MANTIS_DATA_DIR` | `model_iteration_tool/.data` | Where cached market data is stored |
| `MANTIS_CONTAINER_MEM_PCT` | `15` | Per-container memory as % of system RAM (clamped 4-64 GB) |
| `MANTIS_PAUSE_TIMEOUT` | `3600` | Seconds before a paused agent auto-resumes |

### Authentication

**The server refuses to start on `0.0.0.0` without `MANTIS_AUTH_TOKEN` set.** This prevents accidental public exposure that would let anyone create agents and burn your Anthropic API credits.

Without `MANTIS_AUTH_TOKEN`, access is restricted to localhost. To expose publicly:

```bash
export MANTIS_AUTH_TOKEN="your-secret-token"
python -m model_iteration_tool.gui --host 0.0.0.0

# Browser: http://your-host:8420/?token=your-secret-token
# curl:    curl -H "Authorization: Bearer your-secret-token" http://your-host:8420/api/agents
```

The token is accepted via Bearer header, `mantis_token` cookie (set automatically on first browser visit with `?token=`), or `?token=` query parameter.

## Quick start

1. Open `http://127.0.0.1:8420` in a browser
2. Go to **Settings** and paste your Anthropic API key
3. Click **Fetch Data** on the dashboard to prefetch OHLCV data
4. Pick a challenge, write a research goal, and click **Launch**
5. Watch the agent iterate in real time

Or via CLI:

```bash
# Set API key
curl -X POST localhost:8420/api/anthropic-key \
  -H "Content-Type: application/json" \
  -d '{"key": "sk-ant-api03-..."}'

# Prefetch data
curl -X POST localhost:8420/api/data-cache/prefetch \
  -H "Content-Type: application/json" \
  -d '{"days_back": 60}'

# Launch an agent
curl -X POST localhost:8420/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "challenge": "ETH-1H-BINARY",
    "goal": "Explore momentum and mean-reversion signals using RSI, Bollinger bands, and volume-weighted features",
    "max_iterations": 10
  }'
```

## Challenges

| Name | What you predict | Metric | Asset(s) |
|------|-----------------|--------|----------|
| `ETH-1H-BINARY` | ETH up/down in 1 hour | AUC | ETH |
| `ETH-HITFIRST-100M` | Which barrier ETH hits first | Log Loss | ETH |
| `ETH-LBFGS` | ETH return bucket (5-class) | Balanced Accuracy | ETH |
| `BTC-LBFGS-6H` | BTC 6h return bucket | Balanced Accuracy | BTC |
| `MULTI-BREAKOUT` | Range breakout direction | AUC | 29 assets |
| `XSEC-RANK` | Cross-sectional outperformance | Spearman | 29 assets |

## API reference

See **[API_guide.md](API_guide.md)** for the full HTTP API reference with curl examples for every endpoint.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (always public) |
| `GET` | `/api/challenges` | List available challenges |
| `GET` | `/api/agents` | List all agents (supports ETag) |
| `POST` | `/api/agents` | Create and launch an agent |
| `GET` | `/api/agents/<id>` | Get agent detail with all iterations |
| `POST` | `/api/agents/<id>/stop` | Stop agent and kill container |
| `POST` | `/api/agents/<id>/delete` | Delete agent and all data |
| `POST` | `/api/agents/<id>/message` | Send message to agent's INBOX.md |
| `GET` | `/api/agents/<id>/code/<n>` | Get iteration N source code |
| `POST` | `/api/data-cache/prefetch` | Prefetch market data |

## SDK usage (without the GUI)

You can use the evaluation framework directly in Python:

```python
from model_iteration_tool import Featurizer, Predictor, evaluate
import numpy as np

class MyFeaturizer(Featurizer):
    warmup = 200
    compute_interval = 1

    def compute(self, view):
        prices = view.prices("ETH")
        log_r = np.diff(np.log(prices[-100:]))
        return {"momentum": np.array([log_r.mean()]), "vol": np.array([log_r.std()])}

class MyPredictor(Predictor):
    def predict(self, features):
        p_up = 0.5 + 0.5 * np.tanh(features["momentum"][0] * 500)
        return np.array([p_up, 1.0 - p_up])

result = evaluate("ETH-1H-BINARY", MyFeaturizer(), MyPredictor(), days_back=30)
print(f"AUC: {result['mean_auc']:.4f}")
```

See [GUIDE.md](GUIDE.md) for the full SDK reference (CausalView API, embedding formats, all challenge types).

## Running tests

```bash
python model_iteration_tool/test_model_iteration_tool.py
```

## Architecture

```
model_iteration_tool/
    gui.py              Web GUI + REST API (Flask/Gunicorn)
    sandbox.py          Docker container lifecycle
    agent_runner.py     Agent iteration loop (runs inside container)
    evaluator.py        Walk-forward evaluation, label generation
    data.py             Binance OHLCV fetcher, CausalView, DataProvider
    data_cache.py       Data prefetch and caching
    coinglass.py        CoinGlass derivatives data fetcher
    featurizer.py       Featurizer/Predictor base classes
    Dockerfile          Agent container image
    templates/          Jinja2 templates (SPA dashboard)
    test_model_iteration_tool.py  Test suite
    example_binary.py   Example strategy
    GUIDE.md            SDK reference
    API_guide.md        HTTP API reference
```

Each agent runs in an isolated Docker container with:
- Memory limit (15% of system RAM by default, clamped 4-64 GB), CPU limit (2 cores), PID limit (512)
- `--cap-drop=ALL`, `--no-new-privileges`
- Read-only access to framework code, read-write to its own workspace
- Cloud metadata endpoints blocked
- No access to other agents' data or host filesystem

## License

See parent repository.
