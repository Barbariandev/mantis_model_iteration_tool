# MANTIS Agentic Mining -- API Reference

## Featurizer Interface

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mantis_model_iteration_tool.featurizer import Featurizer, Predictor
from mantis_model_iteration_tool.data import CausalView

class TechFeaturizer(Featurizer):
    warmup: int = 0           # minimum timesteps before first valid output
    compute_interval: int = 1 # compute every N timesteps

    def compute(self, view: CausalView) -> dict[str, np.ndarray]:
        # Return dict of named feature arrays (any shape)
        ...

class TechPredictor(Predictor):
    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        # Return embedding array of shape (embedding_dim,)
        ...
```

## CausalView API

- `view.t` -> int: current timestep index
- `view.assets` -> list[str]: loaded tickers
- `view.prices(asset)` -> np.ndarray shape (t+1,): close prices 0..t
- `view.ohlcv(asset)` -> np.ndarray shape (t+1, 5): open, high, low, close, volume columns
- `view.prices_matrix()` -> np.ndarray shape (t+1, N): all assets' close prices

### CoinGlass Derivatives Data (if loaded)

- `view.has_cg()` -> bool: whether CoinGlass data is available
- `view.cg_fields(asset)` -> list[str]: available field names for an asset
- `view.cg(asset, field)` -> np.ndarray shape (t+1,): causally aligned feature

Available fields (when CoinGlass API key is configured):
- `oi_1h`, `oi_1d` -- aggregated open interest close (hourly/daily)
- `funding_1h` -- funding rate close (hourly)
- `liq_long_1h` -- long liquidation USD (hourly)
- `liq_short_1h` -- short liquidation USD (hourly)
- `ls_ratio_1h` -- global long/short account ratio (hourly)

Each value is from the **last fully completed** candle. NaN where no completed candle exists.

All returned arrays are **copies**. No future data is accessible.

## Strategy File Requirements

Your strategy file MUST start with:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
Then import from the local `mantis_model_iteration_tool/` package:
```python
import numpy as np
from mantis_model_iteration_tool.featurizer import Featurizer, Predictor
from mantis_model_iteration_tool.data import CausalView
```

You must define:
- `TechFeaturizer(Featurizer)` with a `compute()` method
- `TechPredictor(Predictor)` with a `predict()` method

## Available Libraries

numpy, scipy, sklearn, pandas, and standard library modules.

## Dev / Eval Split

Data is split into a **dev** period and an **eval holdout** (last 20 days).
- You explore features and compute correlations using DEV data only.
- Walk-forward evaluation runs on ALL data; the test windows fall in the holdout period.
  The walk-forward metrics (the primary metric, etc.) ARE your real out-of-sample scores.
  These are the numbers you should trust, report, and use to judge what works vs what doesn't.
- `dev_emb_dim_N_corr` values are raw dev-period hints, NOT performance metrics. Ignore them
  when deciding if a strategy is good -- only the walk-forward holdout metrics matter.
- You must NEVER manually inspect, analyze, or compute statistics on the eval period.
- If you write custom analysis scripts, truncate data to exclude the last 28800 minutes.

## Custom Data (Advanced)

You can fetch data from **any external API** and make it available in the evaluator.
Place arrays in `custom_data/` in your workspace:

**Option A -- per-asset directory:**
```
custom_data/BTC/my_sentiment.npy    # np.save(path, arr)
custom_data/ETH/my_indicator.npy
```

**Option B -- per-asset npz:**
```
custom_data/BTC.npz    # np.savez(path, my_field1=arr1, my_field2=arr2)
```

Each array must be 1D float64, length matching the OHLCV data (1-minute resolution).
The eval script automatically merges these into the CoinGlass layer. Access via:
```python
view.cg(asset, "my_sentiment")  # your custom field, causally sliced
```

**CRITICAL: YOU must ensure causal alignment.** Each value at minute `t` must only use
data available at time `t`. Use last-completed-candle logic (e.g., for hourly data, the
value at minute 61 is from the candle that closed at minute 60, NOT the candle closing
at minute 120). Misaligned data = lookahead bias = fake performance.

The dashboard flags iterations using custom data with a prominent warning banner.
Only use this when the pre-cached data lacks fields you need.

## File Structure

Your workspace contains:
- `mantis_model_iteration_tool/` -- the framework source code (read-only reference)
- `API_REFERENCE.md` -- this file
- `GOAL.md` -- your research goal
- `STATE.md` -- your persistent memory (read + update every iteration)
- `HISTORY.md` -- previous iteration results
- `INSTRUCTIONS.md` -- what to do this iteration
- `INBOX.md` -- messages from the human operator (if any)
- `_eval_N.py` -- evaluation scripts (do NOT modify)
- `iteration_N.py` -- your strategy files (you write these)
- `_result_N.json` -- evaluation results (eval writes these)
- `iteration_N_analysis.json` -- your analysis (you write these)
- `custom_data/` -- optional: your custom data arrays (you create this)
