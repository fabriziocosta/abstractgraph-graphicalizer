"""Sequence-to-graph graphicalizers adapted from CoCoGraPE."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx

from abstractgraph_graphicalizer.core import GraphicalizerMixin


def sequence_to_graph(
    sequence: Sequence[Any],
    *,
    edge_label: str = "-",
    start_label: str | None = None,
    end_label: str | None = None,
) -> nx.Graph:
    """Convert a token sequence into a path graph with labeled nodes."""
    graph = nx.Graph()
    node_labels: list[Any] = []
    if start_label is not None:
        node_labels.append(start_label)
    node_labels.extend(sequence)
    if end_label is not None:
        node_labels.append(end_label)

    for node_idx, label in enumerate(node_labels):
        graph.add_node(node_idx, label=label)
        if node_idx > 0:
            graph.add_edge(node_idx - 1, node_idx, label=edge_label)

    graph.graph["source"] = "sequence"
    return graph


def string_to_graph(
    text: str,
    *,
    separator: str = "",
    edge_label: str = "-",
    start_label: str | None = None,
    end_label: str | None = None,
) -> nx.Graph:
    """Convert a string into a token path graph."""
    if separator == "":
        tokens = list(text)
    else:
        tokens = text.split(separator)
    graph = sequence_to_graph(
        tokens,
        edge_label=edge_label,
        start_label=start_label,
        end_label=end_label,
    )
    graph.graph["source"] = "string"
    graph.graph["input"] = text
    return graph


class SequenceGraphicalizer(GraphicalizerMixin):
    """Convert iterables of token sequences into labeled path graphs."""

    def __init__(
        self,
        *,
        edge_label: str = "-",
        start_label: str | None = None,
        end_label: str | None = None,
    ) -> None:
        self.edge_label = edge_label
        self.start_label = start_label
        self.end_label = end_label

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [
            sequence_to_graph(
                sequence,
                edge_label=self.edge_label,
                start_label=self.start_label,
                end_label=self.end_label,
            )
            for sequence in X
        ]


class StringGraphicalizer(GraphicalizerMixin):
    """Convert iterables of strings into labeled token path graphs."""

    def __init__(
        self,
        *,
        separator: str = "",
        edge_label: str = "-",
        start_label: str | None = None,
        end_label: str | None = None,
    ) -> None:
        self.separator = separator
        self.edge_label = edge_label
        self.start_label = start_label
        self.end_label = end_label

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [
            string_to_graph(
                text,
                separator=self.separator,
                edge_label=self.edge_label,
                start_label=self.start_label,
                end_label=self.end_label,
            )
            for text in X
        ]
