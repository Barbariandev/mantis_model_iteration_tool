# MANTIS Model Iteration Tool

Autonomous agents for cryptocurrency signal research. This tool runs Claude Code in isolated Docker workspaces, asks it to implement strategy iterations, and evaluates each attempt with causal data access and walk-forward backtesting.

This repository is intended for developers who want to run, inspect, extend, or deploy the MANTIS model iteration loop.

## What It Does

- Launches local or remote agents that write `Featurizer` and `Predictor` strategies.
- Evaluates strategies on crypto forecasting challenges without leaking future data.
- Tracks iteration notes, code, metrics, costs, feature reports, and walk-forward windows in a web dashboard.
- Supports local Docker execution and optional Targon remote execution.

## Requirements

- Python 3.10 or newer.
- Docker, running and available to your user.
- Node.js/npm for installing the Claude Code CLI in agent images.
- Anthropic API key for agent execution.
- Network access to Binance market data; CoinGlass is optional.

## Install

```bash
git clone <your-fork-or-repo-url>
cd mantis_model_iteration_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

For miner or Targon extras:

```bash
pip install -e ".[miner]"
pip install -e ".[targon]"
```

The legacy `requirements.txt` is kept for container/deployment flows. New developer environments should prefer the editable install above.

## Run The GUI

```bash
python -m mantis_model_iteration_tool.gui
# or
mantis-gui
```

Open `http://127.0.0.1:8420`.

To expose the GUI beyond localhost, set a strong token:

```bash
export MANTIS_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m mantis_model_iteration_tool.gui --host 0.0.0.0
```

The GUI refuses non-localhost binding without `MANTIS_AUTH_TOKEN`.

## Security Defaults

Do not commit local runtime files. This repo ignores agent workspaces, cache data, local configs, API keys, miner state, generated deployment files, and wallet artifacts.

Important auth variables:

| Service | Variable | Required When |
| --- | --- | --- |
| GUI | `MANTIS_AUTH_TOKEN` | Any non-localhost bind |
| Targon agent server | `MANTIS_SERVER_AUTH_KEY` | Always for protected remote APIs |
| Remote eval server | `MANTIS_EVAL_API_KEY` | Always for eval/cache APIs |

The remote eval service imports and executes submitted strategy code by design. Run it only behind authentication and container/resource isolation.

If an API key was ever stored in this directory before launch, rotate it before publishing.

## Quick Start

1. Start the GUI.
2. Add your Anthropic API key in Settings.
3. Fetch data from the dashboard.
4. Pick a challenge and write a specific research goal.
5. Launch an agent and monitor the Overview, Iterations, Features, and Notes tabs.

Example SDK usage:

```python
import numpy as np

from mantis_model_iteration_tool import Featurizer, Predictor, evaluate


class MyFeaturizer(Featurizer):
    warmup = 200
    compute_interval = 1

    def compute(self, view):
        prices = view.prices("ETH")
        returns = np.diff(np.log(prices[-100:]))
        return {
            "momentum": np.array([returns.mean()]),
            "volatility": np.array([returns.std()]),
        }


class MyPredictor(Predictor):
    def predict(self, features):
        p_up = 0.5 + 0.5 * np.tanh(features["momentum"][0] * 500)
        return np.array([p_up, 1.0 - p_up])


result = evaluate("ETH-1H-BINARY", MyFeaturizer(), MyPredictor(), days_back=60)
print(result)
```

You can also run the included example:

```bash
python -m mantis_model_iteration_tool.example_binary
```

## Challenges

| Name | Prediction Target | Primary Metric |
| --- | --- | --- |
| `ETH-1H-BINARY` | ETH up/down in 1 hour | AUC |
| `ETH-HITFIRST-100M` | Which ETH volatility barrier hits first | Log loss |
| `ETH-LBFGS` | ETH return bucket | Balanced accuracy |
| `BTC-LBFGS-6H` | BTC 6-hour return bucket | Balanced accuracy |
| `MULTI-BREAKOUT` | Multi-asset breakout direction | AUC |
| `XSEC-RANK` | Cross-sectional outperformance rank | Spearman |
| `FUNDING-XSEC` | Cross-sectional funding-rate changes | Spearman |

## Developer Workflow

```bash
python -m pytest
ruff check .
```

Use `GUIDE.md` for SDK details and `API_guide.md` for GUI HTTP endpoints.

## Architecture

```text
mantis_model_iteration_tool/
    gui.py                  Flask/Gunicorn dashboard and local REST API
    sandbox.py              Docker container lifecycle for local agents
    agent_runner.py         Agent iteration loop run inside each workspace
    evaluator.py            Challenge definitions and walk-forward evaluation
    data.py                 Binance OHLCV fetcher, CausalView, DataProvider
    data_cache.py           Market data prefetch/cache helpers
    featurizer.py           Featurizer/Predictor base classes
    targon_server.py        Remote FastAPI agent-management server
    targon_eval/            Remote evaluation service/deployment helpers
    templates/              Dashboard templates
    test_model_iteration_tool.py
    test_auth.py
    example_binary.py
```

Each local agent runs in a Docker container with memory/CPU/PID limits, dropped capabilities, no new privileges, read-only framework code, and a dedicated writable workspace.

## Contributing

See `CONTRIBUTING.md`.

## Security

See `SECURITY.md` for supported reporting guidance and operational expectations.

## License

MIT. See `LICENSE`.
