"""Prefetch and cache market data (Binance OHLCV + CoinGlass derivatives).

Data is saved as parquet/npz and shared across all agents. CoinGlass data
is causally aligned to prevent lookahead bias.
"""

import json
import logging
import threading
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import os as _os
DATA_DIR = Path(_os.environ.get("MANTIS_DATA_DIR", str(Path(__file__).parent / ".data")))
MANIFEST_NAME = "manifest.json"

_prefetch_lock = threading.Lock()
_prefetch_status = {
    "running": False,
    "progress": "",
    "step": 0,
    "total_steps": 0,
    "error": None,
    "finished_at": None,
}


def get_prefetch_status():
    return dict(_prefetch_status)


def _all_assets():
    from mantis_model_iteration_tool.evaluator import CHALLENGES
    assets = set()
    for cfg in CHALLENGES.values():
        assets.update(cfg.assets)
    return sorted(assets)


def cache_dir_for(days_back):
    return DATA_DIR / f"{days_back}d"


def _fetch_one_ohlcv(asset, interval, days_back, cache_dir):
    """Fetch OHLCV for a single asset. Returns (asset, DataFrame) or None."""
    from mantis_model_iteration_tool.data import fetch_klines, TICKER_TO_SYMBOL
    import os
    symbol = TICKER_TO_SYMBOL.get(asset)
    if not symbol:
        return None
    cache_path = os.path.join(cache_dir, "{}_{}_{}d.csv".format(asset, interval, days_back))
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        logger.info("Cached %s: %d rows", asset, len(df))
        return (asset, df)
    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days_back * 86_400_000
    logger.info("Fetching %s (%s) from Binance...", asset, symbol)
    df = fetch_klines(symbol, interval, start_ms, now_ms)
    if df.empty:
        logger.warning("No data for %s", asset)
        return None
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cache_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, cache_path)
    logger.info("Fetched %s: %d rows", asset, len(df))
    return (asset, df)


def _fetch_one_cg(asset, minute_ts, api_key, days_back):
    """Fetch CoinGlass for a single asset. Returns (asset, feats) or None."""
    from mantis_model_iteration_tool.coinglass import fetch_coinglass_features
    logger.info("Fetching CoinGlass for %s...", asset)
    feats = fetch_coinglass_features(asset, minute_ts, api_key, days_back=days_back)
    if feats:
        return (asset, feats)
    return None


def _write_manifest(manifest_path, days_back, data_all, coinglass_data):
    """Write (or overwrite) the manifest.json for a cache directory."""
    manifest = {
        "days_back": days_back,
        "assets": list(data_all.keys()),
        "has_coinglass": bool(coinglass_data),
        "coinglass_assets": list(coinglass_data.keys()),
        "fetched_at": _time.time(),
        "n_rows": {a: len(df) for a, df in data_all.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def prefetch(days_back=60, coinglass_api_key=None, force=False):
    """Fetch all data and save to disk. Parallelized for speed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    assets = _all_assets()
    out = cache_dir_for(days_back)
    manifest_path = out / MANIFEST_NAME

    if not force and manifest_path.exists():
        raw_m = manifest_path.read_text().strip()
        try:
            manifest = json.loads(raw_m) if raw_m else {}
        except (json.JSONDecodeError, ValueError):
            manifest = {}
        age_hours = (_time.time() - manifest.get("fetched_at", 0)) / 3600
        if age_hours < 1 and set(manifest.get("assets", [])) >= set(assets):
            logger.info("Cache fresh (%.1fh old), skipping prefetch", age_hours)
            return str(out)

    out.mkdir(parents=True, exist_ok=True)
    ohlcv_dir = out / "ohlcv"
    ohlcv_dir.mkdir(exist_ok=True)

    n_assets = len(assets)
    has_cg = bool(coinglass_api_key)
    total = n_assets + (n_assets if has_cg else 0)
    _prefetch_status.update(
        running=True, progress="Fetching OHLCV data (parallel)...",
        step=0, total_steps=total, error=None, finished_at=None)

    logger.info("Prefetching %d assets, %d days (cg=%s) in parallel...",
                n_assets, days_back, has_cg)

    ohlcv_cache = os.path.join(str(DATA_DIR), ".cache")
    data_all = {}
    done_count = [0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_fetch_one_ohlcv, a, "1m", days_back, ohlcv_cache): a
            for a in assets
        }
        for fut in as_completed(futures):
            asset_name = futures[fut]
            result = None
            try:
                result = fut.result()
            except Exception as exc:
                logger.error("OHLCV fetch failed for %s: %s", asset_name, exc)
            if result:
                asset, df = result
                data_all[asset] = df
                try:
                    path = ohlcv_dir / "{}.parquet".format(asset)
                    df.to_parquet(path, index=False)
                except Exception as exc:
                    logger.error("Parquet write failed for %s: %s", asset, exc)
            done_count[0] += 1
            _prefetch_status.update(
                step=done_count[0],
                progress="OHLCV: {}/{} assets".format(done_count[0], n_assets))

    expected_rows = days_back * 1440
    total_rows = sum(len(df) for df in data_all.values())
    total_expected = expected_rows * n_assets
    overall_pct = round(total_rows / total_expected * 100, 1) if total_expected else 0
    logger.info("OHLCV completeness: %d/%d assets, %.1f%% rows (%d/%d)",
                len(data_all), n_assets, overall_pct, total_rows, total_expected)
    if overall_pct < 90:
        logger.warning("Data completeness below 90%% threshold (%.1f%%)", overall_pct)

    # Write intermediate manifest so cache is recognised even if CoinGlass
    # phase fails or the process is interrupted.
    _write_manifest(manifest_path, days_back, data_all, {})

    coinglass_data = {}
    cg_dir = out / "coinglass"
    cg_dir.mkdir(exist_ok=True)

    pre_uploaded = {}
    for npz_path in cg_dir.glob("*.npz"):
        asset_name = npz_path.stem
        if asset_name in data_all:
            try:
                with np.load(npz_path) as npz:
                    pre_uploaded[asset_name] = {k: npz[k] for k in npz.files}
                logger.info("Pre-loaded CoinGlass %s from upload (%d fields)",
                            asset_name, len(pre_uploaded[asset_name]))
            except Exception:
                logger.warning("Corrupt CG npz %s, will re-fetch", asset_name)

    if pre_uploaded:
        coinglass_data.update(pre_uploaded)
        logger.info("Using %d pre-uploaded CoinGlass assets, skipping API fetch",
                     len(pre_uploaded))

    need_cg = set(data_all.keys()) - set(pre_uploaded.keys())
    if has_cg and need_cg and data_all:
        _prefetch_status["progress"] = "Fetching CoinGlass (parallel)..."
        min_len = min(len(df) for df in data_all.values())
        if min_len > 0:
            ref_key = "BTC" if "BTC" in data_all else next(iter(data_all))
            ref_df = data_all[ref_key].iloc[:min_len]
            if "timestamp" in ref_df.columns:
                _epoch = pd.Timestamp("1970-01-01", tz="UTC")
                _ts = pd.to_datetime(ref_df["timestamp"], utc=True)
                minute_ts = ((_ts - _epoch).dt.total_seconds() * 1000).astype(np.int64).values
            else:
                now_ms = int(_time.time() * 1000)
                start_ms = now_ms - days_back * 86_400_000
                minute_ts = (np.arange(min_len, dtype=np.int64)
                             * 60_000 + start_ms)

            cg_done = [0]

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(_fetch_one_cg, a, minute_ts,
                                coinglass_api_key, days_back): a
                    for a in need_cg
                }
                for fut in as_completed(futures):
                    cg_asset = futures[fut]
                    result = None
                    try:
                        result = fut.result()
                    except Exception as exc:
                        logger.error("CoinGlass fetch failed for %s: %s", cg_asset, exc)
                    if result:
                        asset, feats = result
                        coinglass_data[asset] = feats
                        np.savez_compressed(
                            cg_dir / "{}.npz".format(asset),
                            **{k: v for k, v in feats.items()},
                        )
                        logger.info("Saved CoinGlass %s: %d fields",
                                    asset, len(feats))
                    cg_done[0] += 1
                    _prefetch_status.update(
                        step=n_assets + cg_done[0],
                        progress="CoinGlass: {}/{} assets".format(
                            cg_done[0], len(need_cg)))

    _write_manifest(manifest_path, days_back, data_all, coinglass_data)
    _prefetch_status.update(
        running=False, step=total, progress="Done",
        finished_at=_time.time())
    logger.info("Prefetch complete: %d assets, cg=%d in %s",
                len(data_all), len(coinglass_data), out)
    return str(out)


def prefetch_background(days_back=60, coinglass_api_key=None, force=False):
    """Run prefetch in a background thread. Returns immediately."""
    if not _prefetch_lock.acquire(blocking=False):
        return False
    if _prefetch_status["running"]:
        _prefetch_lock.release()
        return False
    _prefetch_status.update(
        running=True, progress="Starting...",
        step=0, total_steps=0, error=None, finished_at=None)
    _prefetch_lock.release()

    def _run():
        with _prefetch_lock:
            import traceback
            try:
                prefetch(days_back, coinglass_api_key, force)
            except Exception as exc:
                logger.error("Prefetch thread crashed: %s\n%s",
                             exc, traceback.format_exc())
                _prefetch_status.update(error=str(exc))
            finally:
                _prefetch_status.update(
                    running=False, finished_at=_time.time())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True


def load_cached(days_back, assets=None):
    """Load prefetched data from disk.

    Returns (data_dict, coinglass_dict) where data_dict maps asset->DataFrame
    and coinglass_dict maps asset->dict[field->np.ndarray].

    Tries parquet files first, then falls back to the raw CSV download
    cache so that a CSV-only cache is still usable.
    """
    out = cache_dir_for(days_back)
    manifest_path = out / MANIFEST_NAME
    csv_cache_dir = DATA_DIR / ".cache"

    if assets is None:
        if manifest_path.exists():
            raw_m = manifest_path.read_text().strip()
            try:
                manifest = json.loads(raw_m) if raw_m else {}
            except (json.JSONDecodeError, ValueError):
                manifest = {}
            assets = manifest.get("assets", [])
        else:
            assets = _all_assets()

    ohlcv_dir = out / "ohlcv"
    data = {}
    missing = []
    for asset in assets:
        path = ohlcv_dir / f"{asset}.parquet"
        if path.exists():
            data[asset] = pd.read_parquet(path)
        else:
            csv_path = csv_cache_dir / f"{asset}_1m_{days_back}d.csv"
            if csv_path.exists():
                data[asset] = pd.read_csv(csv_path, parse_dates=["timestamp"])
            else:
                missing.append(asset)
    if missing:
        logger.warning("load_cached: %d/%d assets missing OHLCV: %s",
                        len(missing), len(assets), missing)

    coinglass = {}
    cg_dir = out / "coinglass"
    if cg_dir.exists():
        for asset in assets:
            path = cg_dir / f"{asset}.npz"
            if path.exists():
                with np.load(path) as npz:
                    coinglass[asset] = {k: npz[k] for k in npz.files}

    if not data:
        return None, None
    return data, coinglass


def audit_cache(days_back):
    """Return detailed completeness info for a cache.

    Checks parquet files first, then falls back to the raw CSV download
    cache.  Returns a dict with keys: exists, source, assets, expected_assets,
    per_asset (asset->{rows, expected, pct}), overall_pct, complete (>=90%).
    """
    expected_rows = days_back * 1440
    expected_assets = _all_assets()
    out = cache_dir_for(days_back)
    parquet_dir = out / "ohlcv"
    csv_cache_dir = DATA_DIR / ".cache"

    per_asset = {}
    source = None

    # 1) Try parquet files
    if parquet_dir.is_dir():
        for asset in expected_assets:
            p = parquet_dir / f"{asset}.parquet"
            if p.exists():
                try:
                    rows = len(pd.read_parquet(p, columns=["timestamp"]))
                except Exception:
                    rows = 0
                per_asset[asset] = {
                    "rows": rows,
                    "expected": expected_rows,
                    "pct": round(rows / expected_rows * 100, 1) if expected_rows else 0,
                }
        if per_asset:
            source = "parquet"

    # 2) Fall back to CSV download cache
    if not per_asset and csv_cache_dir.is_dir():
        for asset in expected_assets:
            csv_path = csv_cache_dir / f"{asset}_1m_{days_back}d.csv"
            if csv_path.exists():
                try:
                    rows = sum(1 for _ in open(csv_path)) - 1  # minus header
                except Exception:
                    rows = 0
                per_asset[asset] = {
                    "rows": rows,
                    "expected": expected_rows,
                    "pct": round(rows / expected_rows * 100, 1) if expected_rows else 0,
                }
        if per_asset:
            source = "csv"

    # Check for CoinGlass .npz files on disk
    cg_dir = out / "coinglass"
    cg_assets = []
    if cg_dir.is_dir():
        cg_assets = [p.stem for p in cg_dir.glob("*.npz")]

    if not per_asset:
        return {
            "exists": False,
            "source": None,
            "assets": 0,
            "expected_assets": len(expected_assets),
            "per_asset": {},
            "row_pct": 0,
            "complete": False,
            "has_coinglass": bool(cg_assets),
            "coinglass_assets": len(cg_assets),
        }

    total_rows = sum(v["rows"] for v in per_asset.values())
    total_expected_found = expected_rows * len(per_asset)
    row_pct = round(total_rows / total_expected_found * 100, 1) if total_expected_found else 0

    return {
        "exists": True,
        "source": source,
        "assets": len(per_asset),
        "expected_assets": len(expected_assets),
        "per_asset": per_asset,
        "row_pct": row_pct,
        "complete": row_pct >= 90,
        "has_coinglass": bool(cg_assets),
        "coinglass_assets": len(cg_assets),
    }


def is_cached(days_back, max_age_hours=12):
    """Check if a valid cache exists for the given days_back.

    Checks manifest age first; falls back to auditing actual files so
    that a CSV-only cache (no manifest) is still recognised.  Any cache
    with data is considered valid (completeness is a display concern).
    """
    manifest_path = cache_dir_for(days_back) / MANIFEST_NAME
    if manifest_path.exists():
        raw_m = manifest_path.read_text().strip()
        try:
            manifest = json.loads(raw_m) if raw_m else {}
        except (json.JSONDecodeError, ValueError):
            manifest = {}
        age_hours = (_time.time() - manifest.get("fetched_at", 0)) / 3600
        if age_hours < max_age_hours:
            return True

    audit = audit_cache(days_back)
    return audit.get("exists", False)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prefetch market data")
    parser.add_argument("--days-back", type=int, default=60)
    parser.add_argument("--coinglass-key", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prefetch(args.days_back, args.coinglass_key, args.force)
