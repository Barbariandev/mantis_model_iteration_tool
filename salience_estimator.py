"""Estimate salience of dashboard agents against the production datalog.

Downloads the ~30 GB datalog (SN123 validator SQLite), loads the agent's
strategy, generates subnet-format embeddings for the holdout period, injects
them into the real miner history, and runs the production salience scoring.

RESOURCE REQUIREMENTS
---------------------
* Disk:   ~32 GB free (datalog + working space)
* RAM:    ~32 GB (SQLite + numpy matrices for all miners)
* CPU:    Strong multi-core recommended
* Time:   ~1 hour for a full run
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time as _time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _open_datalog(db_path):
    """Open the datalog SQLite in read-only mode with WAL and large mmap."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA mmap_size=4294967296")
    return conn


def _hotkeys_for_ticker(cursor, ticker):
    """Collect unique miner hotkeys for a challenge ticker."""
    hks = set()
    for (hks_json,) in cursor.execute(
        "SELECT hotkeys FROM challenge_data WHERE ticker = ? AND hotkeys != '[]'",
        (ticker,),
    ):
        for hk in json.loads(hks_json):
            hks.add(hk)
    return hks

DATALOG_URL = "https://pub-879ad825983e43529792665f4f510cd6.r2.dev/datalog.db"
DATALOG_DIR = Path(__file__).resolve().parent.parent / ".storage"
DATALOG_PATH = DATALOG_DIR / "mantis_datalog.db"

AGENT_HOTKEY = "__dashboard_agent__"

_CHALLENGE_TO_SUBNET = {
    "ETH-1H-BINARY": {"ticker": "ETH", "dim": 2, "loss_func": "binary",
                       "blocks_ahead": 300, "weight": 1.0},
    "ETH-HITFIRST-100M": {"ticker": "ETHHITFIRST", "dim": 3, "loss_func": "hitfirst",
                           "blocks_ahead": 500, "weight": 2.5},
    "ETH-LBFGS": {"ticker": "ETHLBFGS", "dim": 17, "loss_func": "lbfgs",
                   "blocks_ahead": 300, "weight": 3.5},
    "BTC-LBFGS-6H": {"ticker": "BTCLBFGS", "dim": 17, "loss_func": "lbfgs",
                      "blocks_ahead": 1800, "weight": 2.875},
    "CADUSD-1H-BINARY": {"ticker": "CADUSD", "dim": 2, "loss_func": "binary",
                          "blocks_ahead": 300, "weight": 1.0},
    "NZDUSD-1H-BINARY": {"ticker": "NZDUSD", "dim": 2, "loss_func": "binary",
                          "blocks_ahead": 300, "weight": 1.0},
    "CHFUSD-1H-BINARY": {"ticker": "CHFUSD", "dim": 2, "loss_func": "binary",
                          "blocks_ahead": 300, "weight": 1.0},
    "XAGUSD-1H-BINARY": {"ticker": "XAGUSD", "dim": 2, "loss_func": "binary",
                          "blocks_ahead": 300, "weight": 1.0},
    "MULTI-BREAKOUT": {"ticker": "MULTIBREAKOUT", "dim": 2,
                        "loss_func": "range_breakout_multi",
                        "blocks_ahead": 0, "weight": 5.0},
    "XSEC-RANK": {"ticker": "MULTIXSEC", "dim": 1,
                   "loss_func": "xsec_rank",
                   "blocks_ahead": 1200, "weight": 3.0},
    "FUNDING-XSEC": {"ticker": "FUNDINGXSEC", "dim": 1,
                      "loss_func": "funding_xsec",
                      "blocks_ahead": 2400, "weight": 4.0},
}

TOTAL_CHALLENGE_WEIGHT = sum(v["weight"] for v in _CHALLENGE_TO_SUBNET.values())


@dataclass
class SalienceResult:
    agent_salience: float = 0.0
    agent_rank: int = 0
    total_miners: int = 0
    percentile: float = 0.0
    challenge_name: str = ""
    challenge_weight: float = 0.0
    estimated_global_share: float = 0.0
    top_miners: list = field(default_factory=list)
    all_salience: dict = field(default_factory=dict)
    error: str = ""
    elapsed_s: float = 0.0
    caveat: str = ""


# ── Datalog download ─────────────────────────────────────────────────────────

def download_datalog(
    dest: str | Path | None = None,
    progress_cb: Callable[[dict], None] | None = None,
) -> str:
    """Stream-download the datalog archive.  Returns path on success."""
    import requests

    dest = Path(dest or DATALOG_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".db.downloading")

    def _emit(msg, pct=0.0, done=False, error=None):
        if progress_cb:
            progress_cb({"message": msg, "pct": round(pct, 1),
                         "done": done, "error": error})

    _emit("Connecting to datalog archive...")

    try:
        r = requests.get(DATALOG_URL, stream=True, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        _emit(f"Download failed: {exc}", error=str(exc), done=True)
        raise

    total = int(r.headers.get("content-length", 0))
    total_gb = total / (1024 ** 3) if total else 0
    downloaded = 0
    t0 = _time.monotonic()

    _emit(f"Downloading {total_gb:.1f} GB datalog...", pct=0.0)

    with open(str(tmp), "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    elapsed = _time.monotonic() - t0
                    speed_mb = (downloaded / (1024 ** 2)) / max(elapsed, 0.1)
                    eta = ((total - downloaded) / (1024 ** 2)) / max(speed_mb, 0.01)
                    _emit(
                        f"Downloading: {downloaded / (1024**3):.1f} / "
                        f"{total_gb:.1f} GB  ({speed_mb:.1f} MB/s, "
                        f"ETA {int(eta // 60)}m{int(eta % 60)}s)",
                        pct=pct,
                    )

    os.replace(str(tmp), str(dest))
    elapsed = _time.monotonic() - t0
    _emit(f"Download complete ({elapsed / 60:.0f}min)", pct=100, done=True)
    logger.info("Datalog downloaded to %s (%.1f GB, %.0fs)", dest,
                downloaded / (1024 ** 3), elapsed)
    return str(dest)


def datalog_exists(path: str | Path | None = None) -> dict:
    """Check if the datalog is already on disk.  Returns status dict."""
    p = Path(path or DATALOG_PATH)
    if not p.exists():
        return {"exists": False, "size_gb": 0, "path": str(p)}
    sz = p.stat().st_size
    return {"exists": True, "size_gb": round(sz / (1024 ** 3), 1), "path": str(p)}


# ── Training data loading (read-only SQLite, no bittensor dep) ───────────────

SAMPLE_EVERY = 5


def _unpack_embeddings(blob: bytes, dim: int) -> dict[str, np.ndarray]:
    if not blob:
        return {}
    try:
        sep = blob.index(b"\x00")
    except ValueError:
        return {}
    hk_list = json.loads(blob[:sep].decode())
    raw = blob[sep + 1:]
    n = len(hk_list)
    if n == 0 or len(raw) != n * dim * 2:
        return {}
    vecs = np.frombuffer(raw, dtype=np.float16).reshape(n, dim)
    return {hk: vecs[i].copy() for i, hk in enumerate(hk_list)}


def _load_binary_training(
    db_path: str, ticker: str, dim: int, blocks_ahead: int,
    max_block: int | None = None,
) -> tuple | None:
    """Load binary challenge training data from the datalog DB.

    Returns ((X, hk2idx), y, sidx_array) or None.
    """
    conn = _open_datalog(db_path)
    c = conn.cursor()

    ahead = blocks_ahead // SAMPLE_EVERY

    prices_by_sidx: dict[int, float] = {}
    for sidx, price in c.execute(
        "SELECT sidx, price FROM challenge_data WHERE ticker = ?", (ticker,),
    ):
        if price is not None:
            prices_by_sidx[int(sidx)] = float(price)

    hks = _hotkeys_for_ticker(c, ticker)
    if not hks:
        conn.close()
        return None

    sorted_hks = sorted(hks)
    hk2idx = {hk: i for i, hk in enumerate(sorted_hks)}

    X_list, y_list, sidx_list = [], [], []

    for sidx, price, emb_blob in c.execute(
        "SELECT sidx, price, embeddings FROM challenge_data "
        "WHERE ticker = ? ORDER BY sidx", (ticker,),
    ):
        sidx = int(sidx)
        block = sidx * SAMPLE_EVERY
        if max_block and block > max_block:
            break

        price_now = float(price) if price is not None else None
        price_fut = prices_by_sidx.get(sidx + ahead)
        if price_now is None or price_fut is None:
            continue
        if price_now <= 0 or price_fut <= 0:
            continue

        emb = _unpack_embeddings(emb_blob, dim) if emb_blob else {}
        if not emb:
            continue

        mat = np.zeros((len(sorted_hks), dim), dtype=np.float16)
        for hk, vec in emb.items():
            idx = hk2idx.get(hk)
            if idx is not None:
                mat[idx] = np.asarray(vec, dtype=np.float16)

        X_list.append(mat.flatten())
        y_list.append((price_fut - price_now) / price_now)
        sidx_list.append(sidx)

    conn.close()
    if not X_list:
        return None

    return (
        (np.array(X_list, dtype=np.float16), hk2idx),
        np.array(y_list, dtype=np.float32),
        np.array(sidx_list, dtype=np.int64),
    )


def _load_lbfgs_training(
    db_path: str, ticker: str, dim: int, blocks_ahead: int,
    max_block: int | None = None,
) -> dict | None:
    """Load LBFGS/hitfirst challenge training data from the datalog DB.

    Returns {"hist": (X, hk2idx), "price": prices, "sidx": sidx_arr,
             "blocks_ahead": blocks_ahead} or None.
    """
    conn = _open_datalog(db_path)
    c = conn.cursor()

    hks = _hotkeys_for_ticker(c, ticker)
    if not hks:
        conn.close()
        return None

    sorted_hks = sorted(hks)
    hk2idx = {hk: i for i, hk in enumerate(sorted_hks)}

    rows, prices, sidx_list = [], [], []

    for sidx, price, emb_blob in c.execute(
        "SELECT sidx, price, embeddings FROM challenge_data "
        "WHERE ticker = ? ORDER BY sidx", (ticker,),
    ):
        block = int(sidx) * SAMPLE_EVERY
        if max_block and block > max_block:
            break
        if price is None:
            continue
        pf = float(price)
        if not np.isfinite(pf) or pf <= 0:
            continue

        emb = _unpack_embeddings(emb_blob, dim) if emb_blob else {}
        row = np.zeros((len(sorted_hks), dim), dtype=np.float32)
        for hk, vec in emb.items():
            idx = hk2idx.get(hk)
            if idx is not None:
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape == (dim,):
                    row[idx] = arr
                elif arr.size == dim:
                    row[idx] = arr.reshape(dim)

        rows.append(row.reshape(-1))
        prices.append(pf)
        sidx_list.append(int(sidx))

    conn.close()
    if not rows:
        return None

    return {
        "hist": (np.stack(rows, axis=0), hk2idx),
        "price": np.asarray(prices, dtype=np.float64),
        "sidx": np.asarray(sidx_list, dtype=np.int64),
        "blocks_ahead": blocks_ahead,
    }


def _load_xsec_training(
    db_path: str, dim: int, blocks_ahead: int,
    max_block: int | None = None,
) -> dict | None:
    """Load XSEC-RANK training data from the datalog DB.

    Returns {"hist": (X, hk2idx), "prices_multi": prices_multi,
             "blocks_ahead": blocks_ahead} or None.
    """
    _ensure_repo_importable()
    import config as subnet_config

    ticker = "MULTIXSEC"
    n_assets = len(subnet_config.BREAKOUT_ASSETS)
    storage_dim = dim * n_assets

    conn = _open_datalog(db_path)
    c = conn.cursor()

    hks = _hotkeys_for_ticker(c, ticker)
    if not hks:
        conn.close()
        return None

    sorted_hks = sorted(hks)
    hk2idx = {hk: i for i, hk in enumerate(sorted_hks)}

    rows, prices_list, sidx_list = [], [], []
    for sidx, price_data, emb_blob in c.execute(
        "SELECT sidx, price_data, embeddings FROM challenge_data "
        "WHERE ticker = ? ORDER BY sidx", (ticker,),
    ):
        block = int(sidx) * SAMPLE_EVERY
        if max_block and block > max_block:
            break
        if not price_data:
            continue
        pd_dict = json.loads(price_data)
        price_vec = [float(pd_dict.get(a, 0.0)) for a in subnet_config.BREAKOUT_ASSETS]
        if not any(p > 0 for p in price_vec):
            continue

        emb = _unpack_embeddings(emb_blob, storage_dim) if emb_blob else {}
        row = np.zeros((len(sorted_hks), storage_dim), dtype=np.float32)
        for hk, vec in emb.items():
            idx = hk2idx.get(hk)
            if idx is not None:
                arr = np.asarray(vec, dtype=np.float32)
                if arr.size == storage_dim:
                    row[idx] = arr.reshape(storage_dim)
        rows.append(row.reshape(-1))
        prices_list.append(price_vec)
        sidx_list.append(int(sidx))

    conn.close()
    if not rows:
        return None

    return {
        "hist": (np.stack(rows, axis=0), hk2idx),
        "prices_multi": np.array(prices_list, dtype=np.float64),
        "sidx": np.asarray(sidx_list, dtype=np.int64),
        "blocks_ahead": blocks_ahead,
    }


def _load_funding_xsec_training(
    db_path: str, dim: int, blocks_ahead: int,
    max_block: int | None = None,
) -> dict | None:
    """Load FUNDING-XSEC training data from the datalog DB.

    Same structure as xsec_rank but uses FUNDINGXSEC ticker and
    FUNDING_ASSETS. Returns {"hist": (X, hk2idx), "funding_multi": funding_multi,
    "blocks_ahead": blocks_ahead} or None.
    """
    _ensure_repo_importable()
    import config as subnet_config

    ticker = "FUNDINGXSEC"
    n_assets = len(subnet_config.FUNDING_ASSETS)
    storage_dim = dim * n_assets

    conn = _open_datalog(db_path)
    c = conn.cursor()

    hks = _hotkeys_for_ticker(c, ticker)
    if not hks:
        conn.close()
        return None

    sorted_hks = sorted(hks)
    hk2idx = {hk: i for i, hk in enumerate(sorted_hks)}

    rows, funding_list, sidx_list = [], [], []
    for sidx, price_data, emb_blob in c.execute(
        "SELECT sidx, price_data, embeddings FROM challenge_data "
        "WHERE ticker = ? ORDER BY sidx", (ticker,),
    ):
        block = int(sidx) * SAMPLE_EVERY
        if max_block and block > max_block:
            break
        if not price_data:
            continue
        pd_dict = json.loads(price_data)
        funding_vec = [float(pd_dict.get(a, 0.0)) for a in subnet_config.FUNDING_ASSETS]

        emb = _unpack_embeddings(emb_blob, storage_dim) if emb_blob else {}
        row = np.zeros((len(sorted_hks), storage_dim), dtype=np.float32)
        for hk, vec in emb.items():
            idx = hk2idx.get(hk)
            if idx is not None:
                arr = np.asarray(vec, dtype=np.float32)
                if arr.size == storage_dim:
                    row[idx] = arr.reshape(storage_dim)
        rows.append(row.reshape(-1))
        funding_list.append(funding_vec)
        sidx_list.append(int(sidx))

    conn.close()
    if not rows:
        return None

    return {
        "hist": (np.stack(rows, axis=0), hk2idx),
        "funding_multi": np.array(funding_list, dtype=np.float64),
        "sidx": np.asarray(sidx_list, dtype=np.int64),
        "blocks_ahead": blocks_ahead,
    }


def _load_breakout_training(
    db_path: str,
    max_block: int | None = None,
) -> list | None:
    """Load MULTI-BREAKOUT training data (CompletedBreakoutSample list) from datalog.

    Mirrors ledger.py's DataLog._load_breakout_from_db but works directly
    on the sqlite file without the full DataLog class.
    """
    from collections import deque

    _ensure_repo_importable()
    import config as subnet_config
    from range_breakout import CompletedBreakoutSample

    mb = subnet_config.CHALLENGE_MAP.get("MULTIBREAKOUT")
    if not mb:
        return None

    assets = mb["assets"]
    n_assets = len(assets)
    asset_indices = {a: i for i, a in enumerate(assets)}
    storage_dim = 2 * n_assets
    lookback_sidxs = mb.get("range_lookback_blocks", 28800) // SAMPLE_EVERY
    barrier_frac = mb.get("barrier_pct", 25.0) / 100.0
    min_range_frac = mb.get("min_range_pct", 1.0) / 100.0
    max_pending_blocks = 43200

    conn = _open_datalog(db_path)
    c = conn.cursor()

    # Find where MULTIXSEC data starts
    row = c.execute(
        "SELECT MIN(sidx) FROM challenge_data "
        "WHERE ticker='MULTIXSEC' AND price_data IS NOT NULL"
    ).fetchone()
    xsec_start = int(row[0]) if row and row[0] else None

    # Phase 1: pre-MULTIXSEC samples from breakout_state table
    tables = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}

    pre_xsec: list = []
    if "breakout_state" in tables and xsec_start is not None:
        for asset, state_json in c.execute(
            "SELECT asset, state_json FROM breakout_state"
        ):
            aidx = asset_indices.get(asset)
            if aidx is None:
                continue
            estart = aidx * 2
            for cd in json.loads(state_json).get("completed", []):
                if cd["trigger_sidx"] >= xsec_start:
                    continue
                if max_block and cd["resolution_block"] > max_block:
                    continue
                emb_raw = cd.get("embeddings", {})
                per_asset = {}
                for hk, v in emb_raw.items():
                    arr = np.array(v, dtype=np.float16)
                    if arr.shape[0] >= estart + 2:
                        per_asset[hk] = arr[estart:estart + 2]
                    elif arr.shape == (2,):
                        per_asset[hk] = arr
                sample = CompletedBreakoutSample(
                    trigger_sidx=cd["trigger_sidx"],
                    trigger_block=cd["trigger_block"],
                    resolution_block=cd["resolution_block"],
                    direction=cd["direction"],
                    label=cd["label"],
                    embeddings=per_asset,
                )
                sample.asset_name = asset
                pre_xsec.append(sample)

    if xsec_start is None:
        conn.close()
        return pre_xsec or None

    # Phase 2: load MULTIXSEC prices
    sidxs_list, price_rows = [], []
    for sidx, price_data in c.execute(
        "SELECT sidx, price_data FROM challenge_data "
        "WHERE ticker='MULTIXSEC' AND price_data IS NOT NULL ORDER BY sidx",
    ):
        sidx = int(sidx)
        if max_block and sidx * SAMPLE_EVERY > max_block:
            break
        pd_dict = json.loads(price_data)
        sidxs_list.append(sidx)
        price_rows.append([float(pd_dict.get(a, 0.0)) for a in assets])

    if not sidxs_list:
        conn.close()
        return pre_xsec or None

    sidx_arr = np.array(sidxs_list, dtype=np.int64)
    price_mat = np.array(price_rows, dtype=np.float64)
    T = len(sidxs_list)

    # Phase 3: batch-load MULTIBREAKOUT embedding blobs
    emb_blobs: dict = {}
    for sidx, blob in c.execute(
        "SELECT sidx, embeddings FROM challenge_data "
        "WHERE ticker='MULTIBREAKOUT' AND sidx>=? AND embeddings IS NOT NULL",
        (xsec_start,),
    ):
        emb_blobs[int(sidx)] = blob

    emb_decoded: dict = {}

    def _slice_emb(trigger_sidx: int, ai: int) -> dict:
        if trigger_sidx not in emb_decoded:
            blob = emb_blobs.get(trigger_sidx)
            emb_decoded[trigger_sidx] = (
                _unpack_embeddings(blob, storage_dim) if blob else {}
            )
        full = emb_decoded[trigger_sidx]
        if not full:
            return {}
        s = ai * 2
        out = {}
        for hk, vec in full.items():
            if vec.shape[0] >= s + 2:
                out[hk] = vec[s:s + 2]
        return out

    # Phase 4: vectorised replay per asset (matches ledger.py exactly)
    post_xsec: list = []
    for ai in range(n_assets):
        prices = price_mat[:, ai]

        # Rolling min/max via monotone deques.
        # Key: compute range bounds BEFORE adding current price so that
        # the current price can exceed the previous window's extremes.
        rng_lo = np.full(T, np.nan)
        rng_hi = np.full(T, np.nan)
        rng_cnt = np.zeros(T, dtype=np.int32)
        lo_q: deque = deque()  # (price, index) monotone increasing
        hi_q: deque = deque()  # (price, index) monotone decreasing
        valid_in_window = 0
        win_head = 0

        for t in range(T):
            cur_sidx = int(sidx_arr[t])
            win_lo = cur_sidx - lookback_sidxs

            while win_head < t and sidx_arr[win_head] < win_lo:
                if prices[win_head] > 0:
                    valid_in_window -= 1
                win_head += 1
            while lo_q and lo_q[0][1] < win_head:
                lo_q.popleft()
            while hi_q and hi_q[0][1] < win_head:
                hi_q.popleft()

            if lo_q:
                rng_lo[t] = lo_q[0][0]
            if hi_q:
                rng_hi[t] = hi_q[0][0]
            rng_cnt[t] = valid_in_window

            if prices[t] > 0:
                pv = prices[t]
                while lo_q and lo_q[-1][0] >= pv:
                    lo_q.pop()
                lo_q.append((pv, t))
                while hi_q and hi_q[-1][0] <= pv:
                    hi_q.pop()
                hi_q.append((pv, t))
                valid_in_window += 1

        pending_hi = None
        pending_lo = None
        half_lookback = lookback_sidxs // 2

        for t in range(T):
            p = prices[t]
            cur_sidx = int(sidx_arr[t])
            cur_block = cur_sidx * SAMPLE_EVERY
            if not (np.isfinite(p) and p > 0):
                continue

            if pending_hi is not None:
                tsidx, tblk, d, cont_b, rev_b = pending_hi
                if cur_block - tblk > max_pending_blocks:
                    pending_hi = None
                elif p >= cont_b:
                    embs = _slice_emb(tsidx, ai)
                    sample = CompletedBreakoutSample(
                        trigger_sidx=tsidx, trigger_block=tblk,
                        resolution_block=cur_block,
                        direction=1, label=1, embeddings=embs,
                    )
                    sample.asset_name = assets[ai]
                    post_xsec.append(sample)
                    pending_hi = None
                elif p <= rev_b:
                    embs = _slice_emb(tsidx, ai)
                    sample = CompletedBreakoutSample(
                        trigger_sidx=tsidx, trigger_block=tblk,
                        resolution_block=cur_block,
                        direction=1, label=0, embeddings=embs,
                    )
                    sample.asset_name = assets[ai]
                    post_xsec.append(sample)
                    pending_hi = None

            if pending_lo is not None:
                tsidx, tblk, d, cont_b, rev_b = pending_lo
                if cur_block - tblk > max_pending_blocks:
                    pending_lo = None
                elif p <= cont_b:
                    embs = _slice_emb(tsidx, ai)
                    sample = CompletedBreakoutSample(
                        trigger_sidx=tsidx, trigger_block=tblk,
                        resolution_block=cur_block,
                        direction=-1, label=1, embeddings=embs,
                    )
                    sample.asset_name = assets[ai]
                    post_xsec.append(sample)
                    pending_lo = None
                elif p >= rev_b:
                    embs = _slice_emb(tsidx, ai)
                    sample = CompletedBreakoutSample(
                        trigger_sidx=tsidx, trigger_block=tblk,
                        resolution_block=cur_block,
                        direction=-1, label=0, embeddings=embs,
                    )
                    sample.asset_name = assets[ai]
                    post_xsec.append(sample)
                    pending_lo = None

            if rng_cnt[t] < half_lookback:
                continue
            lo, hi = rng_lo[t], rng_hi[t]
            if np.isnan(lo) or np.isnan(hi):
                continue
            rw = hi - lo
            mid = (hi + lo) / 2.0
            if mid <= 0 or rw / mid < min_range_frac:
                continue
            bd = rw * barrier_frac

            if p > hi and pending_hi is None:
                pending_hi = (cur_sidx, cur_block, 1, p + bd, p - bd)
            if p < lo and pending_lo is None:
                pending_lo = (cur_sidx, cur_block, -1, p - bd, p + bd)

    conn.close()
    all_samples = pre_xsec + post_xsec
    all_samples.sort(key=lambda s: s.trigger_sidx)
    logger.info("Loaded %d breakout samples (%d pre-xsec, %d post-xsec)",
                len(all_samples), len(pre_xsec), len(post_xsec))
    return all_samples if all_samples else None


# ── Agent embedding generation ───────────────────────────────────────────────

def _generate_agent_embeddings(
    strategy_path: str,
    challenge_name: str,
    days_back: int = 60,
    coinglass_key: str | None = None,
    progress_cb: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run the agent's strategy to produce (embeddings, minute_epoch_ms).

    embeddings: (T, dim) array of raw predictions
    minute_epoch_ms: (T,) array of UTC epoch-ms for each minute
    """
    from mantis_model_iteration_tool.evaluator import CHALLENGES, _generate_embeddings
    from mantis_model_iteration_tool.data import DataProvider, fetch_assets
    from mantis_model_iteration_tool.inferencer import load_strategy

    if progress_cb:
        progress_cb({"message": "Loading agent strategy...", "pct": 10})

    cfg = CHALLENGES.get(challenge_name)
    if cfg is None:
        raise ValueError(f"Unknown challenge: {challenge_name}")

    featurizer, predictor = load_strategy(strategy_path)

    if progress_cb:
        progress_cb({"message": f"Loading {days_back}d OHLCV data...", "pct": 15})

    data_all, cg_data = None, None
    try:
        from mantis_model_iteration_tool.data_cache import load_cached, is_cached
        if is_cached(days_back):
            data_all, cg_data = load_cached(days_back, assets=cfg.assets)
            if data_all:
                logger.info("Loaded %d assets from prefetch cache", len(data_all))
    except Exception as exc:
        logger.warning("Cache load failed, falling back to fetch: %s", exc)

    if not data_all:
        if progress_cb:
            progress_cb({"message": f"Fetching {days_back}d OHLCV data (no cache)...", "pct": 15})
        data_all, cg_data = fetch_assets(
            assets=cfg.assets, interval="1m", days_back=days_back,
            coinglass_api_key=coinglass_key,
        )

    if not data_all:
        raise RuntimeError("No OHLCV data fetched")

    provider = DataProvider(data_all, coinglass=cg_data or None)

    if progress_cb:
        progress_cb({"message": "Generating embeddings (walk-forward)...", "pct": 25})

    raw_embeddings, warmup = _generate_embeddings(
        featurizer, predictor, provider,
    )
    T = provider.length
    dim = cfg.embedding_dim
    embeddings = np.zeros((T, dim), dtype=np.float32)
    n_emb = raw_embeddings.shape[0]
    emb_dim = min(raw_embeddings.shape[1], dim) if raw_embeddings.ndim == 2 else 0
    if n_emb > 0 and emb_dim > 0:
        embeddings[warmup:warmup + n_emb, :emb_dim] = raw_embeddings[:, :emb_dim]
    logger.info("Generated %d embeddings (warmup=%d, dim=%d) for %s",
                n_emb, warmup, emb_dim, challenge_name)

    import pandas as pd
    ref_asset = cfg.assets[0]
    ref_df = data_all[ref_asset].iloc[:provider.length]
    if "timestamp" in ref_df.columns:
        _epoch = pd.Timestamp("1970-01-01", tz="UTC")
        _ts = pd.to_datetime(ref_df["timestamp"], utc=True)
        minute_ms = ((_ts - _epoch).dt.total_seconds() * 1000).astype(np.int64).values
    else:
        now_ms = int(_time.time() * 1000)
        now_ms -= now_ms % 60_000
        start_ms = now_ms - days_back * 86_400_000
        minute_ms = np.arange(provider.length, dtype=np.int64) * 60_000 + start_ms

    return embeddings, minute_ms


# ── Injection & salience computation ─────────────────────────────────────────

def _ensure_repo_importable():
    """Make repo root's salience modules importable, mocking heavy deps."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            mock = types.ModuleType("torch")
            mock.cuda = types.SimpleNamespace(is_available=lambda: False)
            mock.manual_seed = lambda x: None
            mock.set_num_threads = lambda x: None
            mock.set_num_interop_threads = lambda x: None
            mock.use_deterministic_algorithms = lambda x: None
            mock.backends = types.SimpleNamespace(
                cudnn=types.SimpleNamespace(benchmark=False)
            )
            sys.modules["torch"] = mock

    if "bittensor" not in sys.modules:
        try:
            import bittensor  # noqa: F401
        except ImportError:
            sys.modules["bittensor"] = types.ModuleType("bittensor")

    if "timelock" not in sys.modules:
        try:
            import timelock  # noqa: F401
        except ImportError:
            sys.modules["timelock"] = types.ModuleType("timelock")


def _build_agent_sidx_map(agent_embeddings, agent_minute_ms):
    """Map minute-ms timestamps to sidx-keyed embeddings."""
    from mantis_model_iteration_tool.block_sync import ohlcv_minute_to_sidx
    return {ohlcv_minute_to_sidx(int(ms)): agent_embeddings[i]
            for i, ms in enumerate(agent_minute_ms)}


def _inject_columns(X, sidx_arr, agent_by_sidx, holdout_start_sidx, col_dim, dtype, label):
    """Core injection: build agent columns and splice into X matrix.

    Returns (X_new, matched_count, skipped_insample_count).
    """
    X = np.asarray(X, dtype=dtype)
    T = X.shape[0]
    agent_col = np.zeros((T, col_dim), dtype=dtype)
    matched, skipped = 0, 0
    for row_idx, sidx in enumerate(sidx_arr):
        sidx_int = int(sidx)
        if sidx_int < holdout_start_sidx:
            skipped += 1
            continue
        emb = agent_by_sidx.get(sidx_int)
        if emb is not None:
            arr = np.asarray(emb, dtype=dtype)
            fill = min(arr.size, col_dim)
            agent_col[row_idx, :fill] = arr[:fill]
            matched += 1
    logger.info("%s injection: %d/%d holdout rows matched (%d in-sample excluded)",
                label, matched, T - skipped, skipped)
    return np.hstack([X, agent_col]), matched


def _extend_hk2idx(hk2idx):
    hk2idx_new = dict(hk2idx)
    hk2idx_new[AGENT_HOTKEY] = len(hk2idx)
    return hk2idx_new


def _inject_agent_into_binary(training_data, agent_embeddings, agent_minute_ms,
                               holdout_start_sidx, dim):
    (X, hk2idx), y, sidx_arr = training_data
    agent_map = _build_agent_sidx_map(agent_embeddings, agent_minute_ms)
    X_new, _ = _inject_columns(X, sidx_arr, agent_map, holdout_start_sidx,
                                dim, np.float16, "Binary")
    return (X_new, _extend_hk2idx(hk2idx)), y


def _inject_agent_into_lbfgs(training_data, agent_embeddings, agent_minute_ms,
                              holdout_start_sidx, dim):
    X, hk2idx = training_data["hist"]
    agent_map = _build_agent_sidx_map(agent_embeddings, agent_minute_ms)
    X_new, _ = _inject_columns(X, training_data["sidx"], agent_map,
                                holdout_start_sidx, dim, np.float32, "LBFGS")
    return {
        "hist": (X_new, _extend_hk2idx(hk2idx)),
        "price": training_data["price"],
        "sidx": training_data["sidx"],
        "blocks_ahead": training_data["blocks_ahead"],
    }


def _inject_agent_into_xsec(training_data, agent_embeddings, agent_minute_ms,
                             holdout_start_sidx, dim):
    _ensure_repo_importable()
    import config as subnet_config
    storage_dim = dim * len(subnet_config.BREAKOUT_ASSETS)

    X, hk2idx = training_data["hist"]
    agent_map = _build_agent_sidx_map(agent_embeddings, agent_minute_ms)
    X_new, _ = _inject_columns(X, training_data["sidx"], agent_map,
                                holdout_start_sidx, storage_dim, np.float32, "XSEC")
    return {
        "hist": (X_new, _extend_hk2idx(hk2idx)),
        "prices_multi": training_data["prices_multi"],
        "sidx": training_data["sidx"],
        "blocks_ahead": training_data["blocks_ahead"],
    }


def _inject_agent_into_breakout(
    completed_samples: list,
    agent_embeddings: np.ndarray,
    agent_minute_ms: np.ndarray,
    holdout_start_sidx: int = 0,
) -> list:
    """Inject agent embeddings into breakout completed samples."""
    from mantis_model_iteration_tool.data import BREAKOUT_ASSETS

    agent_asset_idx = {a: i for i, a in enumerate(BREAKOUT_ASSETS)}
    agent_by_sidx = _build_agent_sidx_map(agent_embeddings, agent_minute_ms)

    matched, skipped_insample, skipped_no_asset = 0, 0, 0
    for sample in completed_samples:
        if sample.trigger_sidx < holdout_start_sidx:
            skipped_insample += 1
            continue
        asset = getattr(sample, "asset_name", None)
        if asset is None:
            skipped_no_asset += 1
            continue
        aidx = agent_asset_idx.get(asset)
        if aidx is None:
            continue

        emb = agent_by_sidx.get(sample.trigger_sidx)
        if emb is not None:
            start = aidx * 2
            if emb.shape[0] >= start + 2:
                arr = np.asarray(emb[start:start + 2], dtype=np.float16)
                sample.embeddings[AGENT_HOTKEY] = arr
                matched += 1

    logger.info("Breakout injection: %d holdout samples matched "
                "(%d in-sample excluded, %d without asset tag)",
                matched, skipped_insample, skipped_no_asset)
    return completed_samples


def _run_salience(
    loss_func: str,
    ticker: str,
    training_data,
    progress_cb: Callable | None = None,
) -> dict[str, float]:
    """Run the appropriate production salience function."""
    _ensure_repo_importable()

    if loss_func == "binary":
        from model import salience_binary_prediction
        hist, y = training_data
        if progress_cb:
            progress_cb({"message": f"Running binary salience ({ticker})..."})
        return salience_binary_prediction(hist, y, ticker)

    elif loss_func == "lbfgs":
        from bucket_forecast import compute_lbfgs_salience, compute_q_path_salience
        import config as subnet_config
        hist = training_data["hist"]
        price = training_data["price"]
        ba = training_data["blocks_ahead"]
        se = int(subnet_config.SAMPLE_EVERY)

        if progress_cb:
            progress_cb({"message": f"Running LBFGS cls salience ({ticker})..."})
        s_cls = compute_lbfgs_salience(hist, price, blocks_ahead=ba, sample_every=se)

        if progress_cb:
            progress_cb({"message": f"Running LBFGS q-path salience ({ticker})..."})
        s_q = compute_q_path_salience(hist, price, blocks_ahead=ba, sample_every=se)

        keys = sorted(set(s_cls.keys()) | set(s_q.keys()))
        s = {}
        for hk in keys:
            v = 0.75 * float(s_cls.get(hk, 0.0)) + 0.25 * float(s_q.get(hk, 0.0))
            if v > 0.0:
                s[hk] = v
        tot = sum(s.values())
        return {k: v / tot for k, v in s.items()} if tot > 0 else {}

    elif loss_func == "hitfirst":
        from hitfirst import compute_hitfirst_salience
        import config as subnet_config
        hist = training_data["hist"]
        price = training_data["price"]
        ba = training_data["blocks_ahead"]
        se = int(subnet_config.SAMPLE_EVERY)
        if progress_cb:
            progress_cb({"message": f"Running hitfirst salience ({ticker})..."})
        return compute_hitfirst_salience(hist, price, blocks_ahead=ba, sample_every=se)

    elif loss_func == "range_breakout_multi":
        from range_breakout import compute_multi_breakout_salience, _assign_episodes
        n_episodes = int(_assign_episodes(training_data).max()) + 1
        # Production uses min_episodes=5, but our datalog snapshot may
        # have fewer temporal episodes, and the holdout window has even
        # fewer.  Use min(n_episodes, 2) so holdout-only agents can qualify.
        min_ep = min(n_episodes, 2)
        if n_episodes < 5:
            logger.warning("Only %d breakout episodes available (production "
                           "requires 5); using min_episodes=%d for estimation",
                           n_episodes, min_ep)
        if progress_cb:
            progress_cb({"message": f"Running breakout salience "
                         f"({len(training_data)} samples, "
                         f"{n_episodes} episodes)..."})
        result = compute_multi_breakout_salience(
            training_data, min_episodes=min_ep,
        )
        logger.info("Breakout salience result: %d miners with salience, "
                     "agent=%s", len(result),
                     "present" if AGENT_HOTKEY in result else "absent")
        return result

    elif loss_func == "xsec_rank":
        from xsec_rank import compute_xsec_rank_salience
        import config as subnet_config
        hist = training_data["hist"]
        prices_multi = training_data["prices_multi"]
        ba = training_data["blocks_ahead"]
        se = int(subnet_config.SAMPLE_EVERY)
        if progress_cb:
            progress_cb({"message": "Running XSEC-RANK salience..."})
        return compute_xsec_rank_salience(hist, prices_multi, blocks_ahead=ba, sample_every=se)

    elif loss_func == "funding_xsec":
        from funding_xsec import compute_funding_xsec_salience
        import config as subnet_config
        hist = training_data["hist"]
        funding_multi = training_data["funding_multi"]
        ba = training_data["blocks_ahead"]
        se = int(subnet_config.SAMPLE_EVERY)
        if progress_cb:
            progress_cb({"message": "Running FUNDING-XSEC salience..."})
        return compute_funding_xsec_salience(hist, funding_multi, blocks_ahead=ba, sample_every=se)

    return {}


# ── Main estimation entry point ──────────────────────────────────────────────

def estimate_salience(
    strategy_path: str,
    challenge_name: str,
    datalog_path: str | None = None,
    days_back: int = 60,
    holdout_days: int = 20,
    coinglass_key: str | None = None,
    progress_cb: Callable[[dict], None] | None = None,
) -> SalienceResult:
    """Estimate an agent's salience against the production datalog.

    Only injects agent embeddings for the holdout period (the last
    ``holdout_days`` days of the agent's OHLCV data window) so the
    estimate is entirely out-of-sample for the agent's trained model.
    """
    t0 = _time.monotonic()
    db_path = str(datalog_path or DATALOG_PATH)

    spec = _CHALLENGE_TO_SUBNET.get(challenge_name)
    if not spec:
        return SalienceResult(error=f"Unsupported challenge: {challenge_name}")

    ticker = spec["ticker"]
    dim = spec["dim"]
    loss_func = spec["loss_func"]
    blocks_ahead = spec["blocks_ahead"]
    weight = spec["weight"]

    def _p(msg, pct=None):
        if progress_cb:
            d = {"message": msg}
            if pct is not None:
                d["pct"] = round(pct, 1)
            progress_cb(d)

    try:
        # 1. Generate agent embeddings
        _p("Generating agent embeddings...", 5)
        emb_result = _generate_agent_embeddings(
            strategy_path, challenge_name, days_back, coinglass_key,
            progress_cb=progress_cb,
        )
        if emb_result is None:
            return SalienceResult(error="Failed to generate embeddings")
        agent_embeddings, agent_minute_ms = emb_result

        # 2. Determine holdout boundary
        holdout_start_minute = len(agent_minute_ms) - holdout_days * 1440
        if holdout_start_minute < 0:
            holdout_start_minute = 0
        holdout_start_ms = int(agent_minute_ms[holdout_start_minute])
        from mantis_model_iteration_tool.block_sync import ohlcv_minute_to_sidx
        holdout_start_sidx = ohlcv_minute_to_sidx(holdout_start_ms)

        n_holdout_emb = int(np.any(agent_embeddings[holdout_start_minute:] != 0, axis=1).sum())
        _p(f"Agent has {n_holdout_emb} non-zero holdout embeddings "
           f"(sidx >= {holdout_start_sidx})", 30)

        # 3. Load training data from datalog
        _p(f"Loading {ticker} training data from datalog (this may take minutes)...", 35)

        if loss_func == "binary":
            raw_data = _load_binary_training(db_path, ticker, dim, blocks_ahead)
            if raw_data is None:
                return SalienceResult(error=f"No training data for {ticker}")
            (X, hk2idx), y, sidx_arr = raw_data
            n_miners = len(hk2idx)
            n_rows = len(y)
            _p(f"Loaded {n_rows} timesteps, {n_miners} miners", 50)

            # 4. Inject agent embeddings (holdout only)
            _p("Injecting agent embeddings into miner matrix...", 55)
            modified = _inject_agent_into_binary(
                raw_data, agent_embeddings, agent_minute_ms,
                holdout_start_sidx, dim,
            )

            # 5. Run salience
            _p("Running salience computation (this takes a while)...", 60)
            salience = _run_salience(loss_func, ticker, modified, progress_cb)

        elif loss_func in ("lbfgs", "hitfirst"):
            raw_data = _load_lbfgs_training(db_path, ticker, dim, blocks_ahead)
            if raw_data is None:
                return SalienceResult(error=f"No training data for {ticker}")
            X, hk2idx = raw_data["hist"]
            n_miners = len(hk2idx)
            n_rows = X.shape[0]
            _p(f"Loaded {n_rows} timesteps, {n_miners} miners", 50)

            _p("Injecting agent embeddings into miner matrix...", 55)
            modified = _inject_agent_into_lbfgs(
                raw_data, agent_embeddings, agent_minute_ms,
                holdout_start_sidx, dim,
            )

            _p("Running salience computation (this takes a while)...", 60)
            salience = _run_salience(loss_func, ticker, modified, progress_cb)

        elif loss_func == "range_breakout_multi":
            raw_data = _load_breakout_training(db_path)
            if raw_data is None:
                return SalienceResult(error="No breakout training data in datalog")
            _p(f"Loaded {len(raw_data)} breakout samples", 50)

            _p("Injecting agent embeddings into breakout samples...", 55)
            modified = _inject_agent_into_breakout(
                raw_data, agent_embeddings, agent_minute_ms,
                holdout_start_sidx,
            )

            _p("Running breakout salience computation...", 60)
            salience = _run_salience(loss_func, ticker, modified, progress_cb)

        elif loss_func == "xsec_rank":
            raw_data = _load_xsec_training(db_path, dim, blocks_ahead)
            if raw_data is None:
                return SalienceResult(error="No XSEC-RANK training data in datalog")
            X, hk2idx = raw_data["hist"]
            n_miners = len(hk2idx)
            n_rows = X.shape[0]
            _p(f"Loaded {n_rows} timesteps, {n_miners} miners (xsec)", 50)

            _p("Injecting agent embeddings into xsec matrix...", 55)
            modified = _inject_agent_into_xsec(
                raw_data, agent_embeddings, agent_minute_ms,
                holdout_start_sidx, dim,
            )

            _p("Running XSEC-RANK salience computation...", 60)
            salience = _run_salience(loss_func, ticker, modified, progress_cb)

        elif loss_func == "funding_xsec":
            raw_data = _load_funding_xsec_training(db_path, dim, blocks_ahead)
            if raw_data is None:
                return SalienceResult(error="No FUNDING-XSEC training data in datalog "
                                     "(this is a newer challenge — data may not be "
                                     "available yet)")
            X, hk2idx = raw_data["hist"]
            n_miners = len(hk2idx)
            n_rows = X.shape[0]
            _p(f"Loaded {n_rows} timesteps, {n_miners} miners (funding-xsec)", 50)

            _p("Injecting agent embeddings into funding-xsec matrix...", 55)
            modified = _inject_agent_into_xsec(
                raw_data, agent_embeddings, agent_minute_ms,
                holdout_start_sidx, dim,
            )

            _p("Running FUNDING-XSEC salience computation...", 60)
            salience = _run_salience(loss_func, ticker, modified, progress_cb)

        else:
            return SalienceResult(error=f"Unsupported loss_func: {loss_func}")

        # 6. Extract results
        agent_sal = salience.get(AGENT_HOTKEY, 0.0)
        sorted_miners = sorted(salience.items(), key=lambda kv: -kv[1])
        rank = 1
        for hk, v in sorted_miners:
            if hk == AGENT_HOTKEY:
                break
            rank += 1
        total = len(sorted_miners)
        percentile = ((total - rank) / max(total - 1, 1)) * 100 if total > 1 else 0

        estimated_global = (agent_sal * weight / TOTAL_CHALLENGE_WEIGHT) if agent_sal > 0 else 0

        top_10 = [{"hotkey": hk[:12] + "..." if len(hk) > 16 else hk,
                    "salience": round(v, 6),
                    "is_agent": hk == AGENT_HOTKEY}
                   for hk, v in sorted_miners[:10]]

        elapsed = _time.monotonic() - t0
        _p(f"Done! Agent salience: {agent_sal:.6f} (rank {rank}/{total})", 100)

        caveat = ""
        if loss_func in ("xsec_rank", "funding_xsec"):
            caveat = ("Salience estimates for cross-sectional challenges (XSEC-RANK, "
                       "FUNDING-XSEC) may be less accurate due to limited datalog "
                       "history — these are newer challenges with less training data "
                       "available for calibration.")

        return SalienceResult(
            agent_salience=agent_sal,
            agent_rank=rank,
            total_miners=total,
            percentile=round(percentile, 1),
            challenge_name=challenge_name,
            challenge_weight=weight,
            estimated_global_share=round(estimated_global, 6),
            top_miners=top_10,
            all_salience={k: round(v, 6) for k, v in sorted_miners},
            elapsed_s=round(elapsed, 1),
            caveat=caveat,
        )

    except Exception as exc:
        logger.exception("Salience estimation failed")
        return SalienceResult(
            error=str(exc)[:1000],
            elapsed_s=round(_time.monotonic() - t0, 1),
        )
