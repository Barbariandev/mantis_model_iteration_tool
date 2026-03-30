from abc import ABC, abstractmethod

import numpy as np

from model_iteration_tool.data import CausalView


class Featurizer(ABC):
    """Extract features from causal data views. Features can be any shape."""

    compute_interval: int = 1
    warmup: int = 0

    def setup(self, assets, total_timesteps):
        """Called once before evaluation starts. Override for initialization."""
        pass

    @abstractmethod
    def compute(self, view: CausalView) -> dict[str, np.ndarray]:
        """Return dict of named feature arrays. See CausalView for available data."""
        ...


class Predictor(ABC):
    """Converts features from a Featurizer into challenge-specific embeddings."""

    @abstractmethod
    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Given features, produce the embedding vector for the challenge."""
        ...
