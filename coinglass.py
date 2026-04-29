"""CoinGlass API fetcher with causal multi-resolution alignment.

At minute t, only the last fully closed candle of each resolution is
visible. A candle at time T with period P completes at T + P; enforced
via np.searchsorted.
"""

import hashlib
import json
import logging
import os
import time as _time

import numpy as np
import requests

logger = logging.getLogger(__name__)

COINGLASS_BASE = "https://open-api-v4.coinglass.com"

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def align_to_minutes(candle_times_ms, candle_values, minute_times_ms, period_ms):
    """Align candle data to 1m grid. NaN where no completed candle exists yet."""
    candle_times_ms = np.asarray(candle_times_ms, dtype=np.int64)
    candle_values = np.asarray(candle_values, dtype=np.float64)
    minute_times_ms = np.asarray(minute_times_ms, dtype=np.int64)

    order = np.argsort(candle_times_ms)
    sorted_times = candle_times_ms[order]
    sorted_vals = candle_values[order]

    deadlines = minute_times_ms - period_ms
    idx = np.searchsorted(sorted_times, deadlines, side="right") - 1

    result = np.full(len(minute_times_ms), np.nan)
    valid = idx >= 0
    result[valid] = sorted_vals[idx[valid]]
    return result


def _cache_path(cache_dir, endpoint, params, interval):
    key = hashlib.sha256(
        f"{endpoint}|{json.dumps(params, sort_keys=True)}|{interval}".encode()
    ).hexdigest()[:24]
    return os.path.join(cache_dir, f"cg_{key}.json")


def _fetch_coinglass(endpoint, params, api_key, cache_dir=None):
    """Generic CoinGlass API fetcher with caching and rate-limit handling."""
    if cache_dir:
        cp = _cache_path(cache_dir, endpoint, params, params.get("interval", ""))
        if os.path.exists(cp):
            try:
                with open(cp) as f:
                    data = json.load(f)
                logger.info("CoinGlass cache hit: %s", endpoint)
                return data
            except (json.JSONDecodeError, ValueError, OSError):
                logger.warning("Corrupted CG cache at %s, refetching", cp)

    url = f"{COINGLASS_BASE}{endpoint}"
    headers = {"CG-API-KEY": api_key, "accept": "application/json"}

    for attempt in range(3):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            logger.warning("CoinGlass rate limited, waiting %ds", wait)
            _time.sleep(wait)
            continue
        if resp.status_code != 200:
            logger.warning("CoinGlass %s HTTP %d: %s", endpoint, resp.status_code, resp.text[:200])
            return []
        try:
            body = resp.json()
        except ValueError:
            logger.warning("CoinGlass %s: non-JSON response", endpoint)
            return []
        if str(body.get("code", "")) != "0":
            logger.warning("CoinGlass %s error: %s", endpoint, body.get("msg", ""))
            return []
        data = body.get("data", [])
        if cache_dir and data:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cp + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, cp)
        return data

    return []


def _dedup_by_time(data):
    """Remove duplicate rows by timestamp, preserving order."""
    seen = set()
    out = []
    for row in data:
        t = row.get("time", row.get("t", 0))
        if t not in seen:
            seen.add(t)
            out.append(row)
    return out


def _paginated_fetch(endpoint, params, api_key, cache_dir, cache_label,
                     interval, end_ms):
    """Generic paginated CoinGlass fetch with deduplication and caching."""
    if cache_dir:
        cp = _cache_path(cache_dir, cache_label,
                         {"symbol": params.get("symbol", ""),
                          "interval": interval}, interval)
        if os.path.exists(cp):
            try:
                with open(cp) as f:
                    data = json.load(f)
                if data:
                    logger.info("CoinGlass full-cache hit: %s %s", cache_label,
                                params.get("symbol", ""))
                    return data
            except (json.JSONDecodeError, ValueError, OSError):
                pass

    all_data = []
    while True:
        data = _fetch_coinglass(endpoint, params, api_key, cache_dir=None)
        if not data:
            break
        all_data.extend(data)
        last_time = data[-1].get("time", data[-1].get("t", 0))
        if end_ms and last_time >= end_ms:
            break
        if len(data) < 1000:
            break
        params["start_time"] = last_time + INTERVAL_MS.get(interval, 60000)
        _time.sleep(0.3)

    all_data = _dedup_by_time(all_data)

    if cache_dir and all_data:
        os.makedirs(cache_dir, exist_ok=True)
        cp = _cache_path(cache_dir, cache_label,
                         {"symbol": params.get("symbol", ""),
                          "interval": interval}, interval)
        tmp = cp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_data, f)
        os.replace(tmp, cp)

    return all_data


def fetch_oi(symbol, interval, api_key, start_ms=None, end_ms=None, cache_dir=None):
    """Fetch aggregated open interest OHLC history."""
    params = {"symbol": symbol, "interval": interval, "limit": 1000}
    if start_ms:
        params["start_time"] = int(start_ms)
    if end_ms:
        params["end_time"] = int(end_ms)
    return _paginated_fetch(
        "/api/futures/open-interest/aggregated-history",
        params, api_key, cache_dir, "oi-agg", interval, end_ms)


def fetch_funding(symbol, interval, api_key, exchange="Binance",
                  start_ms=None, end_ms=None, cache_dir=None):
    """Fetch funding rate OHLC history for a specific exchange."""
    params = {
        "exchange": exchange,
        "symbol": f"{symbol}USDT",
        "interval": interval,
        "limit": 1000,
    }
    if start_ms:
        params["start_time"] = int(start_ms)
    if end_ms:
        params["end_time"] = int(end_ms)
    return _paginated_fetch(
        "/api/futures/funding-rate/history",
        params, api_key, cache_dir, "funding", interval, end_ms)


def fetch_liquidations(symbol, interval, api_key,
                       start_ms=None, end_ms=None, cache_dir=None):
    """Fetch aggregated liquidation history (long + short USD)."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "exchange_list": "Binance",
        "limit": 1000,
    }
    if start_ms:
        params["start_time"] = int(start_ms)
    if end_ms:
        params["end_time"] = int(end_ms)
    return _paginated_fetch(
        "/api/futures/liquidation/aggregated-history",
        params, api_key, cache_dir, "liq-agg", interval, end_ms)


def fetch_ls_ratio(symbol, interval, api_key,
                   start_ms=None, end_ms=None, cache_dir=None):
    """Fetch global long/short account ratio history."""
    sym = symbol if "USDT" in symbol else symbol + "USDT"
    params = {
        "exchange": "Binance",
        "symbol": sym,
        "interval": interval,
        "limit": 1000,
    }
    if start_ms:
        params["start_time"] = int(start_ms)
    if end_ms:
        params["end_time"] = int(end_ms)
    return _paginated_fetch(
        "/api/futures/global-long-short-account-ratio/history",
        params, api_key, cache_dir, "ls-ratio", interval, end_ms)


def _extract_ohlc_close(data):
    """Extract (times_ms, close_values) from OHLC data list."""
    times = []
    vals = []
    for row in data:
        t = row.get("time", row.get("t", 0))
        c = row.get("close", row.get("c"))
        if t and c is not None and c != 0:
            times.append(int(t))
            vals.append(float(c))
    return np.array(times, dtype=np.int64), np.array(vals, dtype=np.float64)


def _extract_liq_fields(data):
    """Extract (times_ms, long_usd, short_usd) from liquidation data."""
    times = []
    longs = []
    shorts = []
    for row in data:
        t = row.get("time", row.get("t", 0))
        if not t:
            continue
        times.append(int(t))
        longs.append(float(row.get("aggregated_long_liquidation_usd",
                     row.get("longLiquidationUsd", row.get("buyVolUsd", 0))) or 0))
        shorts.append(float(row.get("aggregated_short_liquidation_usd",
                      row.get("shortLiquidationUsd", row.get("sellVolUsd", 0))) or 0))
    return (
        np.array(times, dtype=np.int64),
        np.array(longs, dtype=np.float64),
        np.array(shorts, dtype=np.float64),
    )


def _extract_ratio(data):
    """Extract (times_ms, ratio_values) from LS ratio data."""
    times = []
    vals = []
    for row in data:
        t = row.get("time", row.get("t", 0))
        if not t:
            continue
        times.append(int(t))
        ratio = row.get("global_account_long_short_ratio",
                      row.get("longRate", row.get("longAccount",
                      row.get("ratio", None))))
        if ratio is not None:
            vals.append(float(ratio))
        else:
            vals.append(np.nan)
    return np.array(times, dtype=np.int64), np.array(vals, dtype=np.float64)


def fetch_coinglass_features(asset, minute_timestamps_ms, api_key,
                             days_back=30, cache_dir=None):
    """Fetch all CoinGlass features for an asset, aligned to 1m grid."""
    if cache_dir is None:
        data_root = os.environ.get("MANTIS_DATA_DIR", os.path.join(os.path.dirname(__file__), ".cache"))
        cache_dir = os.path.join(data_root, "coinglass")

    minute_timestamps_ms = np.asarray(minute_timestamps_ms, dtype=np.int64)
    if len(minute_timestamps_ms) == 0:
        return {}
    start_ms = int(minute_timestamps_ms[0]) - 2 * 86_400_000
    end_ms = int(minute_timestamps_ms[-1])
    logger.info("CoinGlass fetch %s: start=%d end=%d len=%d",
                asset, start_ms, end_ms, len(minute_timestamps_ms))

    features = {}

    for interval in ("1h", "1d"):
        period_ms = INTERVAL_MS[interval]
        data = fetch_oi(asset, interval, api_key, start_ms, end_ms, cache_dir)
        if data:
            times, vals = _extract_ohlc_close(data)
            features[f"oi_{interval}"] = align_to_minutes(
                times, vals, minute_timestamps_ms, period_ms,
            )
            logger.info("Aligned oi_%s for %s: %d candles", interval, asset, len(data))

    funding_data = fetch_funding(asset, "1h", api_key, "Binance", start_ms, end_ms, cache_dir)
    if funding_data:
        times, vals = _extract_ohlc_close(funding_data)
        features["funding_1h"] = align_to_minutes(
            times, vals, minute_timestamps_ms, INTERVAL_MS["1h"],
        )
        logger.info("Aligned funding_1h for %s: %d candles", asset, len(funding_data))

    liq_data = fetch_liquidations(asset, "1h", api_key, start_ms, end_ms, cache_dir)
    if liq_data:
        times, long_vals, short_vals = _extract_liq_fields(liq_data)
        features["liq_long_1h"] = align_to_minutes(
            times, long_vals, minute_timestamps_ms, INTERVAL_MS["1h"],
        )
        features["liq_short_1h"] = align_to_minutes(
            times, short_vals, minute_timestamps_ms, INTERVAL_MS["1h"],
        )
        logger.info("Aligned liq_1h for %s: %d candles", asset, len(liq_data))

    ls_data = fetch_ls_ratio(asset, "1h", api_key, start_ms, end_ms, cache_dir)
    if ls_data:
        times, vals = _extract_ratio(ls_data)
        features["ls_ratio_1h"] = align_to_minutes(
            times, vals, minute_timestamps_ms, INTERVAL_MS["1h"],
        )
        logger.info("Aligned ls_ratio_1h for %s: %d candles", asset, len(ls_data))

    return features
