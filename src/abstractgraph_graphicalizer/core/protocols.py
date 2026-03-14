"""Shared graphicalizer protocol helpers."""

from __future__ import annotations


class GraphicalizerMixin:
    """Minimal transformer-like interface for graphicalizers."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
