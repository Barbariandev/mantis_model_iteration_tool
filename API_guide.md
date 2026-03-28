# MANTIS HTTP API Reference

All endpoints return JSON. State-changing requests (`POST`, `DELETE`) require `Content-Type: application/json`.

## Authentication

If `MANTIS_AUTH_TOKEN` is set, every request (except `/health`) must include the token via one of:

- **Bearer header**: `Authorization: Bearer <token>`
- **Cookie**: `mantis_token=<token>` (set automatically on first browser visit with `?token=`)
- **Query parameter**: `?token=<token>`

Without `MANTIS_AUTH_TOKEN`, access is restricted to localhost (`127.0.0.1` / `::1`).

---

## Health

### `GET /health`

Always public (no auth required). Returns server health and agent count.

```bash
curl http://localhost:8420/health
```

```json
{
  "status": "healthy",
  "agents_running": 2,
  "max_agents": 5,
  "uptime_s": 3621.4
}
```

---

## Challenges

### `GET /api/challenges`

List all available challenge types with their metrics and descriptions.

```bash
curl http://localhost:8420/api/challenges
```

```json
{
  "ETH-1H-BINARY": {
    "type": "binary",
    "short": "ETH 1h direction",
    "desc": "Predict whether ETH goes up or down in the next hour",
    "asset": "ETH",
    "metric": "AUC",
    "metric_info": {
      "primary": "mean_auc",
      "all": ["mean_auc"],
      "direction": "higher",
      "label": "AUC"
    }
  }
}
```

---

## Agents

### `GET /api/agents`

List all agents. Returns lightweight summaries by default. Supports ETag caching.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `full` | query | `0` | Set to `1` for full agent state including iterations |

```bash
# Summary list
curl http://localhost:8420/api/agents

# Full detail (includes all iterations, notes, chat)
curl http://localhost:8420/api/agents?full=1
```

Response (summary mode):

```json
[
  {
    "id": "a1b2c3d4e5f6",
    "status": "running",
    "config": {
      "challenge": "ETH-1H-BINARY",
      "goal": "Explore momentum signals",
      "min_iterations": 5,
      "max_iterations": 20,
      "model": "sonnet",
      "days_back": 60
    },
    "created_at": "2026-03-26T12:00:00Z",
    "iteration_count": 3,
    "paused": false,
    "last_metrics": {"mean_auc": 0.542}
  }
]
```

Supports ETag: pass `If-None-Match` header to get `304 Not Modified` when nothing changed.

---

### `GET /api/agents/<id>`

Get full detail for a single agent, including all iterations, live activity, notes, and chat.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6
```

Returns `404` if agent does not exist.

---

### `POST /api/agents`

Create and launch a new agent.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `challenge` | string | yes | — | Challenge name (e.g. `ETH-1H-BINARY`) |
| `goal` | string | yes | — | Research goal (max 10,000 chars) |
| `model` | string | no | `sonnet` | Claude model: `sonnet`, `opus`, or `haiku` |
| `min_iterations` | int | no | `5` | Minimum iterations before agent can signal done (1-100) |
| `max_iterations` | int | no | `20` | Hard stop after this many iterations (1-500) |
| `days_back` | int | no | `60` | Days of historical data to use (60-365) |

```bash
curl -X POST http://localhost:8420/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "challenge": "ETH-1H-BINARY",
    "goal": "Explore momentum and mean-reversion signals using RSI and Bollinger bands",
    "max_iterations": 10
  }'
```

```json
{"id": "a1b2c3d4e5f6", "status": "starting"}
```

Returns `400` if challenge is unknown or goal is missing.
Returns `429` if max concurrent agents reached.

---

### `POST /api/agents/<id>/stop`

Stop a running agent and kill its Docker container.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/stop
```

```json
{"status": "stopped"}
```

---

### `POST /api/agents/<id>/delete`

Delete an agent, kill its container, and remove all data.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/delete
```

```json
{"status": "deleted"}
```

---

### `POST /api/agents/<id>/pause`

Pause an agent between iterations. The agent finishes its current iteration then waits.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/pause
```

```json
{"paused": true}
```

---

### `POST /api/agents/<id>/resume`

Resume a paused agent.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/resume
```

```json
{"paused": false}
```

---

### `POST /api/agents/<id>/goal`

Update a running agent's research goal. The agent sees the new goal on its next iteration.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/goal \
  -H "Content-Type: application/json" \
  -d '{"goal": "Focus on volume-weighted momentum features"}'
```

```json
{"status": "updated"}
```

---

### `GET /api/agents/<id>/goal`

Get the agent's current goal.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/goal
```

```json
{"content": "# Research Goal\n\nFocus on volume-weighted momentum features\n"}
```

---

## Agent Communication

### `POST /api/agents/<id>/message`

Send a message to the agent's INBOX.md. The agent reads this on its next iteration.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/message \
  -H "Content-Type: application/json" \
  -d '{"text": "Try adding RSI divergence as a feature"}'
```

```json
{"status": "sent"}
```

---

### `GET /api/agents/<id>/inbox`

Read the agent's current INBOX.md contents.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/inbox
```

```json
{"content": "[14:30] Operator: Try adding RSI divergence as a feature"}
```

---

### `POST /api/agents/<id>/inbox`

Replace the agent's INBOX.md content entirely.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/inbox \
  -H "Content-Type: application/json" \
  -d '{"content": "New instructions here"}'
```

```json
{"status": "ok"}
```

---

### `GET /api/agents/<id>/chat`

Get the chat history (operator messages + Ask Claude responses).

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/chat
```

```json
{
  "messages": [
    {"role": "user", "text": "How is AUC trending?", "ts": "2026-03-26T14:30:00", "type": "ask"},
    {"role": "assistant", "text": "AUC improved from 0.51 to 0.54 over...", "ts": "2026-03-26T14:30:05", "type": "ask"}
  ],
  "paused": false
}
```

---

### `POST /api/agents/<id>/ask`

Ask Claude a question about the agent's current state. Response appears in the chat asynchronously.

```bash
curl -X POST http://localhost:8420/api/agents/a1b2c3d4e5f6/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What features are contributing most to AUC?"}'
```

```json
{"status": "processing"}
```

Poll `GET /api/agents/<id>/chat` to see the response when ready.

---

## Agent Observability

### `GET /api/agents/<id>/notes`

Get the agent's `notes.txt` (maintained by the agent itself).

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/notes
```

```json
{"content": "Iteration 3: RSI feature improved AUC by 0.02..."}
```

---

### `GET /api/agents/<id>/state`

Get the agent's `STATE.md` (structured state the agent maintains).

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/state
```

```json
{"content": "# Current State\n\nBest AUC: 0.542 (iteration 3)\n..."}
```

---

### `GET /api/agents/<id>/stdout`

Get the agent's raw stdout log.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `tail` | query | `200` | Number of lines from the end (1-5000) |

```bash
curl "http://localhost:8420/api/agents/a1b2c3d4e5f6/stdout?tail=50"
```

```json
{
  "content": "[iter 3] calling claude (sonnet)...\n    reading: notes.txt\n...",
  "lines": 450,
  "showing": 50
}
```

---

### `GET /api/agents/<id>/iterations`

Get all iteration results for an agent.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/iterations
```

```json
{
  "count": 3,
  "iterations": [
    {
      "iteration": 1,
      "timestamp": "2026-03-26T12:05:00",
      "metrics": {"mean_auc": 0.512},
      "analysis": {
        "feature_report": [...],
        "inference_spec": {
          "summary": "Uses 10m and 60m momentum with Bollinger band width...",
          "data_requirements": [{"field": "close", "source": "Binance OHLCV", ...}],
          "update_frequency": "Every 1 minute",
          "output_format": "2-element array [p_up, p_down]",
          "dependencies": ["numpy"]
        },
        "hypothesis": "...",
        "done": false
      },
      "has_error": false,
      "elapsed_s": 45.2,
      "tokens": {"input": 12000, "output": 3500, "cost": 0.0234},
      "done_signal": false,
      "timed_out": false
    }
  ]
}
```

---

### `GET /api/agents/<id>/code/<iteration>`

Get the Python source code from a specific iteration.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/code/3
```

Returns `text/plain` with the source code. Returns `404` if the iteration has no code file.

---

### `GET /api/agents/<id>/iterations/<iteration>/features`

Get the agent's feature report for a specific iteration. The agent writes this as part of its analysis.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/iterations/3/features
```

```json
{
  "iteration": 3,
  "feature_report": [
    {
      "name": "momentum_10m",
      "description": "10-minute log return",
      "signal": "Mean-reversion indicator",
      "importance_rank": 1,
      "stats": {"mean": 0.001, "std": 0.023, "min": -0.12, "max": 0.15},
      "predictive_power": {
        "method": "spearman vs fwd returns",
        "value": 0.05,
        "p_value": 0.02,
        "n_samples": 500,
        "interpretation": "Weak but statistically significant"
      }
    }
  ]
}
```

---

The `feature_report` is written by the agent, not computed automatically. If the agent hasn't written a feature report for an iteration, the array will be empty.

---

### `GET /api/agents/<id>/iterations/<iteration>/metrics`

Get metrics and analysis for a specific iteration.

```bash
curl http://localhost:8420/api/agents/a1b2c3d4e5f6/iterations/3/metrics
```

```json
{
  "iteration": 3,
  "metrics": {"mean_auc": 0.542, "windows": [...]},
  "analysis": {"summary": "...", "done": false}
}
```

---

## API Keys

### `GET /api/anthropic-key`

Check if Anthropic API key is configured. Never returns the full key.

```bash
curl http://localhost:8420/api/anthropic-key
```

```json
{"set": true, "masked": "sk-ant-a...xY9z"}
```

---

### `POST /api/anthropic-key`

Set the Anthropic API key. Must start with `sk-ant-`.

```bash
curl -X POST http://localhost:8420/api/anthropic-key \
  -H "Content-Type: application/json" \
  -d '{"key": "sk-ant-api03-..."}'
```

```json
{"set": true, "masked": "sk-ant-a...xY9z"}
```

---

### `DELETE /api/anthropic-key`

Remove the stored Anthropic API key.

```bash
curl -X DELETE http://localhost:8420/api/anthropic-key
```

```json
{"set": false}
```

---

### `GET /api/coinglass-key`

Check if CoinGlass API key is configured.

```bash
curl http://localhost:8420/api/coinglass-key
```

```json
{"set": true, "masked": "abc123...xY9z"}
```

---

### `POST /api/coinglass-key`

Set the CoinGlass API key. Alphanumeric, hyphens, underscores, and dots only.

```bash
curl -X POST http://localhost:8420/api/coinglass-key \
  -H "Content-Type: application/json" \
  -d '{"key": "your-coinglass-key"}'
```

```json
{"set": true, "masked": "your-c...s-key"}
```

---

### `DELETE /api/coinglass-key`

Remove the stored CoinGlass API key.

```bash
curl -X DELETE http://localhost:8420/api/coinglass-key
```

```json
{"set": false}
```

---

## Data Cache

### `GET /api/data-cache`

Check cache status for all standard lookback windows.

```bash
curl http://localhost:8420/api/data-cache
```

```json
{
  "60": {"cached": true, "assets": 29, "has_coinglass": true, "age_hours": 2.3},
  "90": {"cached": false},
  "120": {"cached": false},
  "180": {"cached": false}
}
```

---

### `POST /api/data-cache/prefetch`

Start background data fetch. Fetches OHLCV from Binance and optionally CoinGlass derivatives data.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `days_back` | int | `60` | Days of history to fetch (min 60, max 365) |
| `coinglass_key` | string | _(stored key)_ | CoinGlass API key (auto-saved if provided) |

```bash
curl -X POST http://localhost:8420/api/data-cache/prefetch \
  -H "Content-Type: application/json" \
  -d '{"days_back": 60}'
```

```json
{"status": "started", "days_back": 60}
```

Returns `409` if a fetch is already running.

---

### `GET /api/data-cache/status`

Check progress of a running prefetch.

```bash
curl http://localhost:8420/api/data-cache/status
```

```json
{
  "running": true,
  "progress": "Fetching OHLCV: 15/29 assets",
  "step": 15,
  "total_steps": 29,
  "error": null,
  "finished_at": null
}
```

---

### `POST /api/data-cache/delete`

Delete cached data for a specific lookback window.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `days_back` | int | `60` | Which cache to delete |

```bash
curl -X POST http://localhost:8420/api/data-cache/delete \
  -H "Content-Type: application/json" \
  -d '{"days_back": 60}'
```

```json
{"status": "deleted", "days_back": 60}
```

Returns `409` if a prefetch is currently running.

---

## Error Responses

All errors return JSON with an `error` field:

```json
{"error": "description of what went wrong"}
```

| Status | Meaning |
|--------|---------|
| `400` | Bad request (missing/invalid parameters) |
| `401` | Unauthorized (auth token required) |
| `403` | Forbidden (wrong token or CSRF violation) |
| `404` | Not found (agent or resource doesn't exist) |
| `429` | Rate limited or max agents reached |
| `409` | Conflict (e.g. deleting cache while fetch is running) |

## Rate Limiting

By default, each IP is limited to 120 requests per minute (sliding window). Configure with `MANTIS_RATE_LIMIT_RPM` environment variable. Set to `0` to disable.
