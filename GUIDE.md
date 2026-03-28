# MANTIS Agentic Mining

Build and evaluate featurizers and predictors against MANTIS challenges using real Binance data, with walk-forward evaluation and strict data leakage prevention.

## Quick start

```python
import numpy as np
from playground import Featurizer, Predictor, evaluate

class MyFeaturizer(Featurizer):
    warmup = 200
    compute_interval = 1

    def compute(self, view):
        prices = view.prices("ETH")
        log_r = np.diff(np.log(prices[-100:]))
        return {
            "momentum": np.array([log_r.mean()]),
            "volatility": np.array([log_r.std()]),
        }

class MyPredictor(Predictor):
    def predict(self, features):
        m = features["momentum"][0]
        p_up = 0.5 + 0.5 * np.tanh(m * 500)
        return np.array([p_up, 1.0 - p_up])

result = evaluate("ETH-1H-BINARY", MyFeaturizer(), MyPredictor(), days_back=90)
print(result)
```

## Challenges

| Name | Type | Asset(s) | Embedding dim | Horizon (minutes) | What you're predicting |
|------|------|----------|---------------|-------------------|----------------------|
| `ETH-1H-BINARY` | binary | ETH | 2 | 60 | Will price go up in 1h? |
| `ETH-HITFIRST-100M` | hitfirst | ETH | 3 | 100 | Will price hit +1σ or -1σ first? |
| `ETH-LBFGS` | lbfgs | ETH | 17 | 60 | Vol-normalized return bucket (5 classes) |
| `BTC-LBFGS-6H` | lbfgs | BTC | 17 | 360 | Same as above, BTC, 6h horizon |
| `MULTI-BREAKOUT` | breakout | 29 assets | 58 | event-driven | Range breakout continuation vs reversal |
| `XSEC-RANK` | xsec_rank | 29 assets | 29 | 240 | Will asset outperform cross-sectional median? |

## What you implement

Two classes: a `Featurizer` and a `Predictor`.

### Featurizer

Extracts features from historical data. Gets a `CausalView` — a read-only, time-bounded window into the data that only shows prices up to the current timestep `t`.

```python
class Featurizer(ABC):
    warmup: int = 0           # min timesteps before first call
    compute_interval: int = 1 # recompute every N steps, cache between

    def setup(self, assets, total_timesteps):
        """Called once before evaluation. Use for pre-allocation."""
        pass

    def compute(self, view: CausalView) -> dict[str, np.ndarray]:
        """Return named feature arrays of any shape."""
        ...
```

### Predictor

Maps features to challenge-specific embeddings.

```python
class Predictor(ABC):
    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Return flat embedding vector of the correct dimension."""
        ...
```

## CausalView API

The `view` object passed to `compute()`. This is your only data access.

| Method | Returns | Description |
|--------|---------|-------------|
| `view.t` | `int` | Current timestep index |
| `view.assets` | `list[str]` | All loaded asset tickers |
| `view.prices(asset)` | `np.ndarray` shape `(t+1,)` | Close prices from index 0 to t |
| `view.ohlcv(asset)` | `np.ndarray` shape `(t+1, 5)` | Columns: open, high, low, close, volume |
| `view.prices_matrix()` | `np.ndarray` shape `(t+1, N)` | Close prices for all N assets |

All returned arrays are owned copies. You cannot leak future data through the public API.

### CoinGlass Derivatives Data (optional)

When CoinGlass data is available, these additional methods are accessible:

| Method | Returns | Description |
|--------|---------|-------------|
| `view.has_cg()` | `bool` | Whether CoinGlass data is loaded |
| `view.cg(asset, field)` | `np.ndarray` shape `(t+1,)` | CoinGlass feature values up to t |
| `view.cg_fields(asset)` | `list[str]` | Available CoinGlass field names for asset |

Common fields: `oi_1h`, `oi_1d` (open interest), `funding_1h` (funding rate), `liq_long_1h` / `liq_short_1h` (liquidations), `ls_ratio_1h` (long/short ratio).

**Rules:**
- Only use the methods listed above.
- Do not access underscore attributes on the view.

## Embedding formats

### Binary (dim=2)
```python
[p_up, p_down]    # probabilities, should sum to ~1
```

### Hitfirst (dim=3)
```python
[p_up_first, p_down_first, p_neither]   # probabilities, should sum to ~1
```
- up_first: price hits +1σ barrier before -1σ
- down_first: price hits -1σ barrier before +1σ
- neither: price stays within barriers

### LBFGS (dim=17)
```python
[bucket_0, bucket_1, bucket_2, bucket_3, bucket_4,   # 5 bucket probs
 q_path_0, ..., q_path_11]                           # 12 q-path probs
```
Buckets are vol-normalized return bins at z-score thresholds [-2, -1, 1, 2]:
- 0: z <= -2 (large drop)
- 1: -2 < z < -1
- 2: -1 <= z <= 1 (neutral)
- 3: 1 < z < 2
- 4: z >= 2 (large rise)

### Multi-Breakout (dim=2*N_assets)
```python
[asset_0_cont_prob, asset_0_rev_prob,    # per-asset: continuation vs reversal
 asset_1_cont_prob, asset_1_rev_prob,
 ...]
```

### XSEC-Rank (dim=N_assets)
```python
[asset_0_score, asset_1_score, ...]   # higher = more likely to outperform median
```

## Using pre-fetched data

Fetch once, evaluate multiple strategies:

```python
from playground import fetch_assets, DataProvider, evaluate

data, _ = fetch_assets(assets=["ETH"], interval="1m", days_back=90)
provider = DataProvider(data)

result = evaluate("ETH-1H-BINARY", my_featurizer, my_predictor, provider=provider)
```

For multi-asset challenges, load all assets:

```python
from playground import BREAKOUT_ASSETS

data, _ = fetch_assets(assets=BREAKOUT_ASSETS, interval="1m", days_back=90)
provider = DataProvider(data)

result = evaluate("MULTI-BREAKOUT", my_featurizer, my_predictor, provider=provider)
```

Data is cached to `playground/.data/` on first fetch (configurable via `MANTIS_DATA_DIR`).

## What the evaluation returns

```python
{
    "challenge": "ETH-1H-BINARY",
    "type": "binary",
    "n_timesteps": 129600,
    "n_evaluated": 129500,
    "horizon": 60,
    "windows": [
        {"start": 560, "end": 4560, "auc": 0.53},
        ...
    ],
    "mean_auc": 0.53,
}
```

- `mean_auc`: walk-forward AUC of your raw embedding scores against labels (no meta-model)

For hitfirst, you get `up_auc` and `dn_auc` (separate up-first and down-first scoring via walk-forward LogReg, matching the original MANTIS evaluation).

## Heavy featurizers

If your feature computation is expensive, set `compute_interval`:

```python
class HeavyFeaturizer(Featurizer):
    warmup = 500
    compute_interval = 60   # recompute every 60 steps, cache between

    def compute(self, view):
        # expensive cross-asset correlation matrix
        mat = view.prices_matrix()
        corr = np.corrcoef(np.diff(np.log(mat[-500:]), axis=0).T)
        return {"corr": corr}  # shape (N, N) -- any shape works
```

## Complex feature shapes

Features can be any shape. The predictor is responsible for mapping them to the right embedding dimension.

```python
class MultiScaleFeaturizer(Featurizer):
    warmup = 1000
    compute_interval = 1

    def compute(self, view):
        p = view.prices("ETH")
        returns = np.diff(np.log(p))
        windows = [10, 50, 200, 500]
        # 2D feature: (4 windows, 3 stats)
        stats = np.array([
            [returns[-w:].mean(), returns[-w:].std(), np.median(returns[-w:])]
            for w in windows if len(returns) >= w
        ])
        # 1D: cross-asset momentum
        mat = view.prices_matrix()
        xasset = np.diff(np.log(mat[-50:]), axis=0).mean(axis=0) if mat.shape[0] > 50 else np.zeros(mat.shape[1])
        return {"multi_scale": stats, "cross_asset": xasset}
```

## File layout

```
playground/
    __init__.py         # public API exports
    data.py             # Binance fetcher, CausalView, DataProvider
    featurizer.py       # Featurizer and Predictor base classes
    evaluator.py        # label generation, walk-forward eval, all 6 challenges
    coinglass.py        # CoinGlass derivatives data (OI, funding, liquidations)
    data_cache.py       # data prefetch and caching
    example_binary.py   # example strategy implementation
    test_playground.py  # 22 tests (causality, labels, leakage, evaluation)
```
