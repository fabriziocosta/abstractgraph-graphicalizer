"""Smoke test for abstractgraph-graphicalizer."""

from __future__ import annotations

import numpy as np

from abstractgraph_graphicalizer.attention import AbstractGraphPreprocessor


def main() -> None:
    X = [np.random.randn(8, 4), np.random.randn(6, 4)]
    y = [0, 1]
    preprocessor = AbstractGraphPreprocessor(d_model=8, n_heads=2, num_layers=1, n_epochs=1)
    preprocessor.fit(X, y)
    graphs = preprocessor.transform(X)
    print("attention_graphs", len(graphs), len(graphs[0]))


if __name__ == "__main__":
    main()
