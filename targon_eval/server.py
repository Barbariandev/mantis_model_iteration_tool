"""MANTIS remote evaluation server for Targon deployment.

Accepts strategy code + challenge config via HTTP, runs walk-forward
evaluation, and returns metrics. Designed to be deployed as a Targon
serverless app or any container platform.
"""

import hmac
import importlib.util
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PKG_DIR.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("mantis.targon")

app = FastAPI(title="MANTIS Evaluation Service", version="1.0.0")

EVAL_API_KEY = os.environ.get("MANTIS_EVAL_API_KEY", "")


async def _verify_key(authorization: str = Header("")):
    if not EVAL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="MANTIS_EVAL_API_KEY is not configured",
        )
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token, EVAL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Data cache ───────────────────────────────────────────────────────────────

_data_lock = threading.Lock()
_data_cache: dict[int, dict] = {}


def _get_or_fetch_data(challenge_name: str, days_back: int,
                       coinglass_api_key: str | None = None):
    """Return (data_all, coinglass_data, provider) for the given config.

    Fetches from Binance on first call per days_back and caches in memory.
    """
    from mantis_model_iteration_tool.evaluator import CHALLENGES
    from mantis_model_iteration_tool.data import DataProvider, fetch_assets

    config = CHALLENGES[challenge_name]

    cache_key = days_back
    with _data_lock:
        cached = _data_cache.get(cache_key)

    if cached and cached.get("assets") and set(config.assets).issubset(set(cached["assets"])):
        data_all = cached["data_all"]
        cg_data = cached.get("coinglass_data", {})
        if cg_data:
            return data_all, cg_data, DataProvider(data_all, coinglass=cg_data)
        return data_all, {}, DataProvider(data_all)

    logger.info("Fetching %dd data for assets=%s ...", days_back, config.assets)
    t0 = time.monotonic()

    all_assets = set()
    for _, cfg in CHALLENGES.items():
        all_assets.update(cfg.assets)
    all_assets = sorted(all_assets)

    if coinglass_api_key:
        data_all, cg_data = fetch_assets(
            assets=all_assets, interval="1m", days_back=days_back,
            coinglass_api_key=coinglass_api_key)
    else:
        data_all, _ = fetch_assets(
            assets=all_assets, interval="1m", days_back=days_back)
        cg_data = {}

    elapsed = time.monotonic() - t0
    logger.info("Data fetch completed in %.1fs (%d assets)", elapsed, len(data_all))

    with _data_lock:
        _data_cache[cache_key] = {
            "data_all": data_all,
            "coinglass_data": cg_data,
            "assets": list(data_all.keys()),
            "fetched_at": time.time(),
        }

    if cg_data:
        return data_all, cg_data, DataProvider(data_all, coinglass=cg_data)
    return data_all, {}, DataProvider(data_all)


# ── Request/response models ──────────────────────────────────────────────────

class EvalRequest(BaseModel):
    strategy_code: str = Field(..., description="Python source code for the strategy")
    challenge: str = Field(..., description="Challenge name (e.g., ETH-1H-BINARY)")
    days_back: int = Field(60, ge=60, le=365)
    holdout_days: int = Field(20, ge=5, le=60)
    coinglass_api_key: str | None = Field(None, description="Optional CoinGlass API key")


class EvalResponse(BaseModel):
    metrics: dict
    error: str | None = None
    elapsed_s: float = 0.0
    custom_data_used: bool = False


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "auth_required": True,
        "auth_configured": bool(EVAL_API_KEY),
    }


@app.post("/evaluate", response_model=EvalResponse, dependencies=[Depends(_verify_key)])
async def evaluate(req: EvalRequest):
    from mantis_model_iteration_tool.evaluator import CHALLENGES, evaluate as run_eval
    from mantis_model_iteration_tool.data import DataProvider

    if req.challenge not in CHALLENGES:
        raise HTTPException(400, f"Unknown challenge: {req.challenge}")

    t0 = time.monotonic()

    try:
        data_all, cg_data, full_provider = _get_or_fetch_data(
            req.challenge, req.days_back, req.coinglass_api_key)
    except Exception as exc:
        logger.exception("Data fetch failed")
        return EvalResponse(metrics={}, error=f"Data fetch failed: {str(exc)[:500]}")

    holdout_minutes = req.holdout_days * 1440
    total_len = full_provider.length
    dev_len = max(0, total_len - holdout_minutes)

    with tempfile.TemporaryDirectory(prefix="mantis_eval_") as tmpdir:
        strategy_path = os.path.join(tmpdir, "strategy.py")
        with open(strategy_path, "w") as f:
            f.write(req.strategy_code)

        try:
            spec = importlib.util.spec_from_file_location("strategy", strategy_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            return EvalResponse(
                metrics={"error": f"Strategy import failed: {str(exc)[:500]}"},
                error=f"Strategy import failed: {str(exc)[:500]}",
                elapsed_s=round(time.monotonic() - t0, 2),
            )

        if not hasattr(mod, "TechFeaturizer") or not hasattr(mod, "TechPredictor"):
            return EvalResponse(
                metrics={"error": "Strategy must define TechFeaturizer and TechPredictor"},
                error="Missing TechFeaturizer or TechPredictor",
                elapsed_s=round(time.monotonic() - t0, 2),
            )

        try:
            featurizer = mod.TechFeaturizer()
            predictor = mod.TechPredictor()
        except Exception as exc:
            return EvalResponse(
                metrics={"error": f"Strategy instantiation failed: {str(exc)[:500]}"},
                error=str(exc)[:500],
                elapsed_s=round(time.monotonic() - t0, 2),
            )

        try:
            result = run_eval(
                req.challenge, featurizer, predictor,
                provider=full_provider, holdout_start=dev_len)
        except Exception as exc:
            logger.exception("Evaluation failed")
            return EvalResponse(
                metrics={"error": f"Evaluation failed: {str(exc)[:500]}"},
                error=str(exc)[:500],
                elapsed_s=round(time.monotonic() - t0, 2),
            )

    result["dev_length"] = dev_len
    result["eval_holdout_minutes"] = total_len - dev_len

    def _nan_safe(o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: _nan_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_nan_safe(v) for v in o]
        return o

    result = _nan_safe(result)
    elapsed = round(time.monotonic() - t0, 2)

    return EvalResponse(
        metrics=result,
        elapsed_s=elapsed,
    )


@app.post("/cache/prefetch", dependencies=[Depends(_verify_key)])
async def prefetch_cache(days_back: int = 60, coinglass_api_key: str | None = None):
    """Pre-fetch and cache data for a given lookback period."""
    from mantis_model_iteration_tool.evaluator import CHALLENGES
    first_challenge = next(iter(CHALLENGES))
    try:
        _get_or_fetch_data(first_challenge, days_back, coinglass_api_key)
        return {"status": "cached", "days_back": days_back}
    except Exception as exc:
        raise HTTPException(500, f"Prefetch failed: {str(exc)[:500]}")


@app.get("/cache/status", dependencies=[Depends(_verify_key)])
async def cache_status():
    with _data_lock:
        entries = {}
        for days, cached in _data_cache.items():
            entries[str(days)] = {
                "assets": len(cached.get("assets", [])),
                "has_coinglass": bool(cached.get("coinglass_data")),
                "fetched_at": cached.get("fetched_at"),
            }
    return {"cached": entries}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    if not EVAL_API_KEY:
        raise SystemExit(
            "Refusing to start without MANTIS_EVAL_API_KEY. "
            "Generate one with: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    uvicorn.run(app, host="0.0.0.0", port=port)
