"""Example: basic Ridge model for ETH-1H-BINARY.

Featurizer computes simple technical indicators (MAs, RSI, Bollinger, returns).
Predictor fits a Ridge classifier on a rolling window internally and outputs
probabilities. Entirely self-contained -- run this file directly.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground import Featurizer, Predictor, evaluate, DataProvider

LOOKBACK = 200
TRAIN_WINDOW = 2000
HORIZON = 60


def _sma(prices, window):
    if len(prices) < window:
        return np.nan
    return prices[-window:].mean()


def _ema(prices, span):
    if len(prices) < span:
        return np.nan
    alpha = 2.0 / (span + 1)
    weights = (1 - alpha) ** np.arange(span)[::-1]
    weights /= weights.sum()
    return np.dot(prices[-span:], weights)


def _rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    up = deltas.copy()
    down = deltas.copy()
    up[up < 0] = 0.0
    down[down > 0] = 0.0
    avg_up = up.mean()
    avg_down = -down.mean()
    if avg_down == 0:
        return 100.0
    rs = avg_up / avg_down
    return 100.0 - 100.0 / (1.0 + rs)


def _bollinger_pct(prices, window=20):
    if len(prices) < window:
        return 0.5
    segment = prices[-window:]
    mu = segment.mean()
    std = segment.std()
    if std < 1e-12:
        return 0.5
    return (prices[-1] - mu) / (2.0 * std) + 0.5


class TechFeaturizer(Featurizer):
    warmup = LOOKBACK
    compute_interval = 1

    def compute(self, view):
        p = view.prices("ETH")
        cur = p[-1]

        log_r = np.diff(np.log(p[-LOOKBACK:]))

        sma_10 = _sma(p, 10)
        sma_50 = _sma(p, 50)
        sma_200 = _sma(p, 200)
        ema_12 = _ema(p, 12)
        ema_26 = _ema(p, 26)

        features = np.array([
            log_r[-1],
            log_r[-5:].mean(),
            log_r[-20:].mean(),
            log_r[-60:].mean(),
            log_r.std(),
            log_r[-20:].std(),
            (cur - sma_10) / (sma_10 + 1e-12),
            (cur - sma_50) / (sma_50 + 1e-12),
            (sma_10 - sma_50) / (sma_50 + 1e-12),
            (sma_50 - sma_200) / (sma_200 + 1e-12),
            (ema_12 - ema_26) / (ema_26 + 1e-12),
            _rsi(p, 14) / 100.0,
            _rsi(p, 28) / 100.0,
            _bollinger_pct(p, 20),
            _bollinger_pct(p, 50),
            np.log(p[-1] / p[-HORIZON]) if len(p) > HORIZON else 0.0,
        ], dtype=np.float64)

        return {"tech": features}


class RidgePredictor(Predictor):
    def __init__(self):
        self._X_buf = []
        self._y_buf = []
        self._model = None
        self._last_train_size = 0

    def predict(self, features):
        x = features["tech"]
        self._X_buf.append(x.copy())

        if len(self._X_buf) > HORIZON:
            label = 1.0 if self._X_buf[-1][-1] > 0 else 0.0
            idx = len(self._X_buf) - 1 - HORIZON
            self._y_buf.append((idx, label))

        if (len(self._y_buf) >= 500
                and len(self._y_buf) - self._last_train_size >= 200):
            X_train = []
            y_train = []
            start = max(0, len(self._y_buf) - TRAIN_WINDOW)
            for idx, label in self._y_buf[start:]:
                X_train.append(self._X_buf[idx])
                y_train.append(label)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            mu = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std < 1e-12] = 1.0
            X_norm = (X_train - mu) / std

            self._model = RidgeClassifier(alpha=1.0)
            self._model.fit(X_norm, y_train)
            self._mu = mu
            self._std = std
            self._last_train_size = len(self._y_buf)

        if self._model is not None:
            x_norm = (x - self._mu) / self._std
            score = self._model.decision_function(x_norm.reshape(1, -1))[0]
            p_up = 1.0 / (1.0 + np.exp(-score))
        else:
            p_up = 0.5

        return np.array([p_up, 1.0 - p_up])


if __name__ == "__main__":
    from playground import fetch_assets

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Fetching ETH 1m candles from Binance ({days} days)...")

    data, _ = fetch_assets(assets=["ETH"], interval="1m", days_back=days)
    provider = DataProvider(data)
    print(f"Loaded {provider.length} candles\n")

    result = evaluate(
        "ETH-1H-BINARY",
        TechFeaturizer(),
        RidgePredictor(),
        provider=provider,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"Challenge: {result['challenge']}")
    print(f"Timesteps: {result['n_timesteps']}")
    print(f"Evaluated: {result['n_evaluated']}")
    print(f"Walk-forward windows: {len(result['windows'])}")
    print(f"Mean AUC:  {result['mean_auc']:.4f}")
    print("\nPer-window:")
    for w in result["windows"]:
        print(f"  [{w['start']:>6d} - {w['end']:>6d}]  "
              f"auc={w['auc']:.4f}")
