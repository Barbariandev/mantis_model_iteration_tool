"""Live inference engine for MANTIS mining.

Loads trained Featurizer + Predictor from an agent's iteration_N.py,
fetches live market data, and produces embeddings formatted for the
subnet's challenge protocol.

The critical invariant is that feature computation uses the EXACT same
CausalView / DataProvider / Featurizer.compute() path as evaluation,
so there is zero train-serve skew.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mantis_model_iteration_tool.data import (
    BREAKOUT_ASSETS,
    CausalView,
    DataProvider,
    TICKER_TO_SYMBOL,
    fetch_assets,
    fetch_klines,
)
from mantis_model_iteration_tool.encryption import NAME_TO_TICKER, SUBNET_CHALLENGES

logger = logging.getLogger(__name__)

WARMUP_MINUTES = 5000


@dataclass
class ModelSlot:
    """One loaded model assigned to one or more challenges."""

    agent_id: str
    iteration: int
    challenge_name: str
    strategy_path: str
    featurizer: Any = None
    predictor: Any = None
    loaded: bool = False
    error: str = ""


def load_strategy(strategy_path: str) -> tuple[Any, Any]:
    """Dynamically load TechFeaturizer + TechPredictor from an iteration file.

    This mirrors exactly how the eval script loads strategies.
    """
    path = Path(strategy_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    spec = importlib.util.spec_from_file_location("strat", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    featurizer_cls = getattr(mod, "TechFeaturizer", None)
    predictor_cls = getattr(mod, "TechPredictor", None)
    if featurizer_cls is None or predictor_cls is None:
        raise AttributeError(
            f"Strategy {path.name} must define TechFeaturizer and TechPredictor classes"
        )
    return featurizer_cls(), predictor_cls()


def load_model_slot(slot: ModelSlot) -> ModelSlot:
    """Load the featurizer and predictor for a model slot."""
    try:
        slot.featurizer, slot.predictor = load_strategy(slot.strategy_path)
        slot.loaded = True
        slot.error = ""
        logger.info(
            "Loaded strategy: agent=%s iter=%d challenge=%s warmup=%d",
            slot.agent_id, slot.iteration, slot.challenge_name,
            slot.featurizer.warmup,
        )
    except Exception as exc:
        slot.loaded = False
        slot.error = str(exc)
        logger.error("Failed to load strategy %s: %s", slot.strategy_path, exc)
    return slot


def _collect_required_assets(slots: list[ModelSlot]) -> list[str]:
    """Determine the full set of assets needed across all model slots."""
    from mantis_model_iteration_tool.evaluator import CHALLENGES as EVAL_CHALLENGES

    needed: set[str] = set()
    for slot in slots:
        if not slot.loaded:
            continue
        cfg = EVAL_CHALLENGES.get(slot.challenge_name)
        if cfg:
            needed.update(cfg.assets)

    for sc in SUBNET_CHALLENGES:
        if sc.get("assets"):
            needed.update(sc["assets"])

    available = set(TICKER_TO_SYMBOL.keys())
    return sorted(needed & available)


def fetch_live_data(
    assets: list[str],
    lookback_minutes: int = WARMUP_MINUTES,
    coinglass_key: str = "",
) -> DataProvider:
    """Fetch recent OHLCV data from Binance for live inference.

    Returns a DataProvider with the last `lookback_minutes` of 1m candles.
    Uses the same fetch_assets path as evaluation to ensure consistency.
    """
    days_back = max(2, (lookback_minutes // 1440) + 1)
    import tempfile, shutil
    _tmpdir = tempfile.mkdtemp(prefix="mantis_live_")
    try:
        data, cg_data = fetch_assets(
            assets=assets,
            interval="1m",
            days_back=days_back,
            cache_dir=_tmpdir,
            coinglass_api_key=coinglass_key or None,
        )
    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)
    if not data:
        raise RuntimeError("No market data fetched from Binance")

    for asset in list(data.keys()):
        df = data[asset]
        if len(df) > lookback_minutes:
            data[asset] = df.iloc[-lookback_minutes:].reset_index(drop=True)

    trimmed_cg: dict = {}
    if cg_data:
        min_len = min(len(data[a]) for a in data)
        for asset, fields in cg_data.items():
            if asset in data:
                trimmed_cg[asset] = {
                    k: v[-min_len:] for k, v in fields.items()
                }

    provider = DataProvider(data, coinglass=trimmed_cg)
    logger.info(
        "Live data: %d assets, %d candles, cg=%s",
        len(data), provider.length, bool(trimmed_cg),
    )
    return provider


def run_inference_single(
    slot: ModelSlot,
    provider: DataProvider,
) -> np.ndarray:
    """Run a single model's featurizer + predictor on the latest candle.

    Returns the raw embedding vector. This uses the exact same code path
    as _generate_embeddings in evaluator.py but only computes the final
    timestep.
    """
    if not slot.loaded:
        raise RuntimeError(f"Model slot not loaded: {slot.error}")

    feat = slot.featurizer
    pred = slot.predictor

    t_latest = provider.length - 1
    warmup = feat.warmup

    if t_latest < warmup:
        raise RuntimeError(
            f"Not enough data for warmup: have {t_latest + 1} candles, "
            f"need {warmup + 1}"
        )

    feat.setup(provider.assets, provider.length)

    ci = max(1, feat.compute_interval)

    # Eval computes features at: warmup, warmup+ci, warmup+2*ci, ...
    # We must start on the same grid so cache_t aligns at t_latest.
    last_grid_t = warmup + ((t_latest - warmup) // ci) * ci
    start_t = max(warmup, last_grid_t)

    cache = None
    cache_t = -999999
    latest_emb = None

    for t in range(start_t, t_latest + 1):
        if cache is None or (t - cache_t) >= ci:
            view = provider.view(t)
            cache = feat.compute(view)
            cache_t = t
        latest_emb = pred.predict(cache)

    return np.asarray(latest_emb, dtype=np.float32).ravel()


def _format_binary(emb: np.ndarray) -> list[float]:
    """Format a dim-2 binary embedding as [-1, 1] range features."""
    if len(emb) < 2:
        return [0.0, 0.0]
    p_up = float(np.clip(emb[0], 0, 1))
    p_dn = float(np.clip(emb[1], 0, 1))
    feat_up = p_up * 2 - 1
    feat_dn = p_dn * 2 - 1
    return [feat_up, feat_dn]


def _format_hitfirst(emb: np.ndarray) -> list[float]:
    """Format a dim-3 hitfirst embedding as probabilities summing to 1."""
    if len(emb) < 3:
        return [1 / 3, 1 / 3, 1 / 3]
    probs = np.clip(emb[:3], 1e-6, 1 - 1e-6)
    probs = probs / probs.sum()
    return probs.tolist()


def _format_lbfgs(emb: np.ndarray) -> list[float]:
    """Format a dim-17 LBFGS embedding.

    p[0:5] must be valid probabilities summing to 1.
    q[5:17] must be in (0, 1).
    """
    if len(emb) < 17:
        out = np.full(17, 0.0)
        out[:min(len(emb), 17)] = emb[:17]
        emb = out
    p = np.clip(emb[:5], 1e-6, 1 - 1e-6)
    p = p / p.sum()
    q = np.clip(emb[5:17], 1e-6, 1 - 1e-6)
    return np.concatenate([p, q]).tolist()


def _format_breakout(emb: np.ndarray, assets: list[str]) -> dict[str, list[float]]:
    """Format a multi-breakout embedding as {asset: [p_cont, p_rev]}."""
    result = {}
    for i, asset in enumerate(assets):
        if i * 2 + 1 < len(emb):
            p_cont = float(np.clip(emb[i * 2], 1e-6, 1 - 1e-6))
            p_rev = float(np.clip(emb[i * 2 + 1], 1e-6, 1 - 1e-6))
        else:
            p_cont, p_rev = 0.5, 0.5
        result[asset] = [p_cont, p_rev]
    return result


def _format_xsec(emb: np.ndarray, assets: list[str]) -> dict[str, float]:
    """Format an xsec-rank embedding as {asset: score}."""
    result = {}
    for i, asset in enumerate(assets):
        if i < len(emb):
            result[asset] = float(np.clip(emb[i], -1, 1))
        else:
            result[asset] = 0.0
    return result


def format_embedding_for_subnet(
    challenge_name: str,
    emb: np.ndarray,
) -> tuple[str, Any]:
    """Convert a raw embedding vector to the subnet's expected format.

    Returns (ticker, formatted_value) ready for the payload dict.
    """
    from mantis_model_iteration_tool.evaluator import CHALLENGES as EVAL_CHALLENGES

    ticker = NAME_TO_TICKER.get(challenge_name)
    if ticker is None:
        raise ValueError(f"Unknown challenge: {challenge_name}")

    cfg = EVAL_CHALLENGES.get(challenge_name)
    if cfg is None:
        raise ValueError(f"No eval config for challenge: {challenge_name}")

    ctype = cfg.challenge_type

    if ctype == "binary":
        return ticker, _format_binary(emb)
    elif ctype == "hitfirst":
        return ticker, _format_hitfirst(emb)
    elif ctype == "lbfgs":
        return ticker, _format_lbfgs(emb)
    elif ctype == "breakout":
        sc = next((c for c in SUBNET_CHALLENGES if c["name"] == challenge_name), None)
        assets = sc["assets"] if sc and sc.get("assets") else cfg.assets
        return ticker, _format_breakout(emb, assets)
    elif ctype == "xsec_rank":
        sc = next((c for c in SUBNET_CHALLENGES if c["name"] == challenge_name), None)
        assets = sc["assets"] if sc and sc.get("assets") else cfg.assets
        return ticker, _format_xsec(emb, assets)
    elif ctype == "funding_xsec":
        sc = next((c for c in SUBNET_CHALLENGES if c["name"] == challenge_name), None)
        assets = sc["assets"] if sc and sc.get("assets") else cfg.assets
        return ticker, _format_xsec(emb, assets)
    else:
        return ticker, emb.tolist()


def run_all_inference(
    slots: list[ModelSlot],
    provider: DataProvider,
) -> dict[str, Any]:
    """Run inference for all loaded model slots and return a combined
    embedding dict keyed by subnet ticker, ready for encryption."""
    embeddings: dict[str, Any] = {}

    for slot in slots:
        if not slot.loaded:
            logger.warning(
                "Skipping unloaded slot: agent=%s challenge=%s error=%s",
                slot.agent_id, slot.challenge_name, slot.error,
            )
            continue
        try:
            raw_emb = run_inference_single(slot, provider)
            ticker, formatted = format_embedding_for_subnet(
                slot.challenge_name, raw_emb,
            )
            embeddings[ticker] = formatted
            logger.debug(
                "Inference OK: %s -> %s (dim=%d)",
                slot.challenge_name, ticker, len(raw_emb),
            )
        except Exception as exc:
            logger.error(
                "Inference failed for %s (agent=%s iter=%d): %s",
                slot.challenge_name, slot.agent_id, slot.iteration, exc,
            )

    return embeddings
