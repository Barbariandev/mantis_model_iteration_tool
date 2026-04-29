import os
import time as _time
import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api.binance.us",
]
_binance_api_cache: str | None = None

TICKER_TO_SYMBOL = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "SOL": "SOLUSDT", "TRX": "TRXUSDT", "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT", "BCH": "BCHUSDT", "XMR": "XMRUSDT",
    "LINK": "LINKUSDT", "XLM": "XLMUSDT", "ZEC": "ZECUSDT",
    "SUI": "SUIUSDT", "LTC": "LTCUSDT", "AVAX": "AVAXUSDT",
    "HBAR": "HBARUSDT", "SHIB": "SHIBUSDT", "TON": "TONUSDT",
    "DOT": "DOTUSDT", "UNI": "UNIUSDT", "NEAR": "NEARUSDT",
    "ICP": "ICPUSDT", "ETC": "ETCUSDT", "ONDO": "ONDOUSDT",
    "AAVE": "AAVEUSDT", "PEPE": "PEPEUSDT", "TAO": "TAOUSDT",
    "CRO": "CROUSDT", "MNT": "MNTUSDT",
}

BREAKOUT_ASSETS = [
    "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
    "LINK", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
    "TON", "CRO", "DOT", "UNI", "MNT", "TAO", "AAVE", "PEPE",
    "NEAR", "ICP", "ETC", "ONDO",
]

FUNDING_ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "SUI",
    "NEAR", "AAVE", "UNI", "LTC", "HBAR", "PEPE", "TRX", "SHIB", "TAO", "ONDO",
]


def _resolve_binance_api() -> str:
    """Pick a working Binance API endpoint, caching the result."""
    global _binance_api_cache
    if _binance_api_cache:
        return _binance_api_cache
    override = os.environ.get("BINANCE_API_URL")
    if override:
        _binance_api_cache = override.rstrip("/")
        return _binance_api_cache
    for ep in BINANCE_ENDPOINTS:
        try:
            r = requests.get(f"{ep}/api/v3/ping", timeout=5)
            if r.status_code == 200:
                logger.info("Using Binance endpoint: %s", ep)
                _binance_api_cache = ep
                return ep
            logger.info("Binance %s returned HTTP %d, trying next", ep, r.status_code)
        except requests.RequestException:
            logger.info("Binance %s unreachable, trying next", ep)
    _binance_api_cache = BINANCE_ENDPOINTS[0]
    return _binance_api_cache


def fetch_klines(symbol, interval="1m", start_ms=None, end_ms=None):
    api_base = _resolve_binance_api()
    url = f"{api_base}/api/v3/klines"
    rows = []
    retries_429 = 0
    max_retries_429 = 5
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            retries_429 += 1
            if retries_429 > max_retries_429:
                logger.error("Rate limited %d times for %s, giving up", retries_429, symbol)
                break
            logger.warning("Rate limited (%d/%d), sleeping 60s", retries_429, max_retries_429)
            _time.sleep(60)
            continue
        retries_429 = 0
        if resp.status_code != 200:
            logger.warning("Binance API error for %s: HTTP %d", symbol, resp.status_code)
            break
        try:
            data = resp.json()
        except ValueError:
            logger.warning("Invalid JSON from Binance for %s", symbol)
            break
        if not isinstance(data, list):
            logger.warning("Binance returned non-list for %s: %s",
                           symbol, str(data)[:200])
            break
        if not data:
            break

        rows.extend(data)
        start_ms = data[-1][6] + 1
        if end_ms and start_ms > end_ms:
            break
        if len(data) < 1000:
            break
        _time.sleep(0.2 if "binance.us" in api_base else 0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_vol", "taker_buy_quote_vol", "ignore",
    ])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_assets(assets=None, interval="1m", days_back=90, cache_dir=None,
                 coinglass_api_key=None):
    """Fetch Binance OHLCV data, optionally with CoinGlass features.

    Always returns (data, coinglass_data) -- a 2-tuple.
    coinglass_data is an empty dict when no CoinGlass key is provided.
    """
    if assets is None:
        assets = BREAKOUT_ASSETS
    if cache_dir is None:
        data_root = os.environ.get("MANTIS_DATA_DIR", os.path.dirname(__file__))
        cache_dir = os.path.join(data_root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days_back * 86_400_000

    result = {}
    for asset in assets:
        symbol = TICKER_TO_SYMBOL.get(asset)
        if not symbol:
            logger.warning("No Binance symbol for %s, skipping", asset)
            continue

        cache_path = os.path.join(cache_dir, f"{asset}_{interval}_{days_back}d.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            expected_minutes = days_back * 1440
            if len(df) < expected_minutes * 0.5:
                logger.warning("Cache for %s looks truncated (%d rows, expected ~%d), refetching",
                               asset, len(df), expected_minutes)
                os.remove(cache_path)
            else:
                logger.info("Cached %s: %d rows", asset, len(df))
                result[asset] = df
                continue
        logger.info("Fetching %s (%s) from Binance...", asset, symbol)
        df = fetch_klines(symbol, interval, start_ms, now_ms)
        if df.empty:
            logger.warning("No data for %s", asset)
            continue
        tmp_path = cache_path + ".tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, cache_path)
        logger.info("Fetched %s: %d rows", asset, len(df))
        result[asset] = df
        _time.sleep(1)

    if not coinglass_api_key:
        return result, {}

    from mantis_model_iteration_tool.coinglass import fetch_coinglass_features

    min_len = min(len(df) for df in result.values()) if result else 0
    if min_len == 0:
        return result, {}

    ref_asset = next(iter(result))
    ref_df = result[ref_asset].iloc[:min_len]
    if "timestamp" in ref_df.columns:
        _epoch = pd.Timestamp("1970-01-01", tz="UTC")
        _ts = pd.to_datetime(ref_df["timestamp"], utc=True)
        minute_ts = ((_ts - _epoch).dt.total_seconds() * 1000).astype(np.int64).values
    else:
        minute_ts = np.arange(min_len, dtype=np.int64) * 60_000 + start_ms

    cg_cache = os.path.join(cache_dir, "coinglass")
    coinglass_data = {}
    for asset in result:
        logger.info("Fetching CoinGlass features for %s...", asset)
        feats = fetch_coinglass_features(
            asset, minute_ts, coinglass_api_key,
            days_back=days_back, cache_dir=cg_cache,
        )
        if feats:
            coinglass_data[asset] = feats
            logger.info("CoinGlass %s: %d features", asset, len(feats))

    return result, coinglass_data


class CausalView:
    """Time-bounded view into multi-asset data up to timestep t.
    Name-mangled slots, blocked setattr, and copy-on-read prevent data leakage.
    """

    __slots__ = (
        "_CausalView__closes",
        "_CausalView__ohlcv",
        "_CausalView__t",
        "_CausalView__assets",
        "_CausalView__coinglass",
    )

    def __init__(self, closes, ohlcv, t, assets, coinglass=None):
        object.__setattr__(self, '_CausalView__closes', closes)
        object.__setattr__(self, '_CausalView__ohlcv', ohlcv)
        object.__setattr__(self, '_CausalView__t', int(t))
        object.__setattr__(self, '_CausalView__assets', list(assets))
        object.__setattr__(self, '_CausalView__coinglass', coinglass or {})

    def __setattr__(self, name, value):
        raise AttributeError("CausalView is immutable")

    def __delattr__(self, name):
        raise AttributeError("CausalView is immutable")

    @property
    def t(self):
        return self.__t

    @property
    def assets(self):
        return list(self.__assets)

    def prices(self, asset):
        """Close prices from index 0 to t inclusive. Returns an owned copy."""
        return self.__closes[asset][:self.__t + 1].copy()

    def ohlcv(self, asset):
        """OHLCV as (t+1, 5) array: [open, high, low, close, volume]. Returns an owned copy."""
        return self.__ohlcv[asset][:self.__t + 1].copy()

    def prices_matrix(self):
        """Close prices for all assets as (t+1, num_assets) array."""
        cols = [self.__closes[a][:self.__t + 1].copy() for a in self.__assets]
        return np.column_stack(cols)

    def cg(self, asset, field):
        """CoinGlass feature aligned to 1m, sliced to [0..t]. Causally safe."""
        asset_cg = self.__coinglass.get(asset, {})
        arr = asset_cg.get(field)
        if arr is None:
            raise KeyError(f"No CoinGlass field '{field}' for asset '{asset}'. "
                           f"Available: {list(asset_cg.keys())}")
        return arr[:self.__t + 1].copy()

    def cg_fields(self, asset):
        """List available CoinGlass field names for an asset."""
        return list(self.__coinglass.get(asset, {}).keys())

    def has_cg(self):
        """Whether any CoinGlass data is loaded."""
        return bool(self.__coinglass)


class DataProvider:
    """Aligned multi-asset OHLCV data with optional CoinGlass features."""

    def __init__(self, data, coinglass=None):
        self._assets = sorted(data.keys()) if data else []
        if not self._assets:
            self._length = 0
            self._closes = {}
            self._ohlcv = {}
            self._coinglass = {}
            return

        min_len = min(len(data[a]) for a in self._assets)
        self._length = min_len
        self._closes = {}
        self._ohlcv = {}
        for a in self._assets:
            df = data[a].iloc[:min_len]
            self._closes[a] = df["close"].values.astype(np.float64)
            self._ohlcv[a] = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)

        self._coinglass = {}
        if coinglass:
            for asset, fields in coinglass.items():
                self._coinglass[asset] = {
                    k: v[:min_len] for k, v in fields.items()
                }

    @property
    def assets(self):
        return self._assets

    @property
    def length(self):
        return self._length

    def view(self, t):
        if not (0 <= t < self._length):
            raise IndexError(f"t={t} out of range [0, {self._length})")
        return CausalView(self._closes, self._ohlcv, t, self._assets, self._coinglass)

    def prices(self, asset):
        """Full price series (for label generation, not for featurizers)."""
        return self._closes[asset].copy()

    def prices_matrix(self):
        """Full price matrix (T, num_assets) for label generation."""
        if not self._assets:
            raise ValueError("DataProvider has no assets loaded")
        return np.column_stack([self._closes[a] for a in self._assets])
