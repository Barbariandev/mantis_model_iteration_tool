"""Minimal ETH direction strategy for the MANTIS evaluation framework.

Run from an editable install with:

    python -m mantis_model_iteration_tool.example_binary
"""

import numpy as np

from mantis_model_iteration_tool import Featurizer, Predictor, evaluate


class TechFeaturizer(Featurizer):
    """Compute a small momentum/volatility feature set from causal price history."""

    warmup = 120
    compute_interval = 5

    def compute(self, view):
        prices = view.prices("ETH")
        if len(prices) < 60:
            return {
                "momentum": np.array([0.0]),
                "volatility": np.array([0.0]),
            }

        returns = np.diff(np.log(prices[-60:]))
        short_returns = returns[-15:]
        return {
            "momentum": np.array([float(short_returns.mean())]),
            "volatility": np.array([float(returns.std())]),
        }


class TechPredictor(Predictor):
    """Convert features into an up/down probability vector."""

    def predict(self, features):
        momentum = float(features["momentum"][0])
        volatility = max(float(features["volatility"][0]), 1e-8)
        score = np.clip(momentum / volatility, -4.0, 4.0)
        p_up = float(1.0 / (1.0 + np.exp(-score)))
        return np.array([p_up, 1.0 - p_up])


def main():
    result = evaluate(
        "ETH-1H-BINARY",
        TechFeaturizer(),
        TechPredictor(),
        days_back=60,
    )
    print(result)


if __name__ == "__main__":
    main()
