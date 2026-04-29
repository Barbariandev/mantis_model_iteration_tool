# MANTIS Agent Server — Targon Integration

## Quick Start (from the GUI)

1. Open MANTIS dashboard (http://localhost:8420)
2. Go to **Settings → Targon**
3. Make sure your **Anthropic API key** is set (needed on the remote server)
4. Enter your **Targon API key** (get one at https://targon.com/dashboard)
5. Click **Deploy**
6. Once deployed, all agents run entirely on Targon — Claude Code, evaluation, everything

No local Docker containers needed. No published Docker images needed.
The Targon SDK packages and ships your code automatically.

## Architecture

```
Your Machine                          Targon Server (CVM)
┌──────────────────┐                  ┌──────────────────────────┐
│  Flask GUI :8420 │   HTTPS + Auth   │  FastAPI Agent Server    │
│                  │ ───────────────> │    ├── agent_runner.py   │
│  Settings        │                  │    │   ├── Claude Code   │
│  Dashboard       │ <─────────────── │    │   ├── _eval_N.py   │
│  Agent views     │   JSON state     │    │   └── walk-forward  │
│                  │                  │    ├── OHLCV data cache  │
│  (thin proxy)    │                  │    └── Agent workspaces  │
└──────────────────┘                  └──────────────────────────┘
```

**Local mode (fallback):** Agents run in Docker containers on your machine.
Everything works as before — no Targon account needed.

**Targon mode:** The GUI proxies all agent requests to the remote server.
Your machine only serves the HTML dashboard.

## What Runs Where

| Component | Local Mode | Targon Mode |
|-----------|-----------|-------------|
| Flask GUI | Your machine | Your machine |
| Agent processes | Docker (your machine) | Targon server |
| Claude Code CLI | Docker (your machine) | Targon server |
| Data fetching | Docker (your machine) | Targon server |
| Walk-forward eval | Docker (your machine) | Targon server |
| Agent state | Local filesystem | Targon filesystem (via HTTP) |

## Security

### Communication
- All GUI ↔ Targon traffic goes over **HTTPS** (provided by Targon)
- Every request carries a **Bearer token** (auto-generated, 256-bit)
- The token is stored in `.targon_config` (chmod 600) and never exposed to the browser

### Credentials on Targon
- **Anthropic API key**: baked into `Image.env()` during deployment, encrypted at rest by Targon
- **CoinGlass API key**: same treatment (optional)
- **Server auth key**: same treatment; `MANTIS_SERVER_AUTH_KEY` is mandatory for the agent server
- **Eval API key**: `MANTIS_EVAL_API_KEY` is mandatory when running the standalone eval service
- Credentials are sent once during deployment, not with every request
- Changing credentials requires redeployment

### No shell injection
- All subprocess calls use list arguments (never `shell=True`)
- Agent IDs are validated against `^[a-f0-9]{1,40}$`
- Challenge names are validated against the known set
- All user inputs are length-limited and sanitized

### Local site
- CSRF protection on all state-changing requests
- Rate limiting (120 RPM default)
- Content-Security-Policy headers
- Auth token required for non-localhost binding

## How Deployment Works

1. GUI generates a temporary `targon_app.py` with credentials in `Image.env()`
2. The SDK's `add_local_dir` ships the `mantis_model_iteration_tool` package
3. `apt_install` adds Node.js; `run_commands` installs Claude Code CLI
4. `pip_install` adds numpy, pandas, scipy, scikit-learn, etc.
5. The SDK builds the container image on Targon's infrastructure
6. Targon provisions compute and starts the web server
7. GUI stores the server URL and auth key in `.targon_config`
8. All subsequent agent operations are proxied to the server

## Manual Deployment (without the GUI)

```bash
pip install targon-sdk
targon setup
cd /path/to/mantis_model_iteration_tool
targon deploy targon_eval/targon_app.py --name mantis-server
```

Note: manual deployment requires setting environment variables
(`ANTHROPIC_API_KEY`, `MANTIS_SERVER_AUTH_KEY`) separately. The standalone
evaluation service also requires `MANTIS_EVAL_API_KEY`; it imports and runs
submitted strategy code and must not be exposed without auth and isolation.

## API Reference

### GUI → Targon Server (proxied routes)

All `/api/agents/*` and `/api/challenges` routes are proxied to Targon
when the server is deployed. The GUI adds the auth header automatically.

### GUI-only Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/targon` | Get Targon config status |
| POST | `/api/targon/key` | Set Targon API key |
| DELETE | `/api/targon/key` | Remove Targon API key |
| POST | `/api/targon/deploy` | Deploy agent server |
| GET | `/api/targon/deploy/status` | Check deployment progress |
| POST | `/api/targon/teardown` | Remove agent server |
| GET | `/api/targon/health` | Health check server |

### Targon Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health |
| GET | `/api/challenges` | List challenges |
| POST | `/api/agents` | Create agent |
| GET | `/api/agents` | List agents |
| GET | `/api/agents/{id}` | Get agent detail |
| POST | `/api/agents/{id}/stop` | Stop agent |
| POST | `/api/agents/{id}/delete` | Delete agent |
| POST | `/api/agents/{id}/pause` | Pause agent |
| POST | `/api/agents/{id}/resume` | Resume agent |
| POST | `/api/agents/{id}/message` | Send message |
| POST | `/api/agents/{id}/ask` | Ask Claude about agent |
| GET | `/api/agents/{id}/chat` | Get chat history |
| GET | `/api/agents/{id}/code/{n}` | Get strategy code |
| GET | `/api/agents/{id}/stdout` | Get stdout log |
| GET | `/api/agents/{id}/notes` | Get notes |
| GET | `/api/agents/{id}/state` | Get STATE.md |
| GET/POST | `/api/agents/{id}/goal` | Get/set goal |
| GET | `/api/agents/{id}/iterations` | Get iterations |
