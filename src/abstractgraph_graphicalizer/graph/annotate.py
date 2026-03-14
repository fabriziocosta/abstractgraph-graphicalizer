"""Graph annotation graphicalizers adapted from CoCoGraPE."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD

from abstractgraph_graphicalizer.core import GraphicalizerMixin


def _resize_right(data: np.ndarray, desired_dim: int) -> np.ndarray:
    if desired_dim < data.shape[1]:
        return data[:, :desired_dim]
    padding_dim = desired_dim - data.shape[1]
    return np.pad(data, pad_width=((0, 0), (0, padding_dim)))


def normalized_laplacian_svd(graph: nx.Graph, n_components: int) -> np.ndarray:
    """Return a fixed-width SVD embedding of the normalized Laplacian."""
    if len(graph) == 0:
        return np.zeros((0, n_components))
    if len(graph) == 1:
        return np.zeros((1, n_components))

    laplacian = nx.normalized_laplacian_matrix(graph)
    effective_n_components = min(n_components, len(graph) - 1)
    data = TruncatedSVD(n_components=effective_n_components).fit_transform(laplacian)
    return _resize_right(data, desired_dim=n_components)


def annotate_normalized_laplacian_svd(
    graph: nx.Graph,
    *,
    n_components: int,
    attribute_key: str = "vec",
) -> nx.Graph:
    """Annotate nodes with normalized-Laplacian SVD features."""
    out_graph = graph.copy()
    data = normalized_laplacian_svd(graph, n_components=n_components)
    for row_idx, node_idx in enumerate(graph.nodes()):
        existing = out_graph.nodes[node_idx].get(attribute_key)
        if existing is None:
            vec = data[row_idx]
        else:
            vec = np.hstack([np.asarray(existing).reshape(-1), data[row_idx]])
        out_graph.nodes[node_idx][attribute_key] = vec
    out_graph.graph["source"] = "normalized_laplacian_svd"
    return out_graph


class NormalizedLaplacianSVDGraphGraphicalizer(GraphicalizerMixin):
    """Annotate graphs with node embeddings from the normalized Laplacian."""

    def __init__(self, *, n_components: int = 10, attribute_key: str = "vec") -> None:
        self.n_components = n_components
        self.attribute_key = attribute_key

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [
            annotate_normalized_laplacian_svd(
                graph,
                n_components=self.n_components,
                attribute_key=self.attribute_key,
            )
            for graph in X
        ]


class NodeEmbedderGraphGraphicalizer(GraphicalizerMixin):
    """Attach node-level predictions or embeddings from an external transformer."""

    def __init__(self, node_transformer, *, attribute_key: str = "pred") -> None:
        self.node_transformer = node_transformer
        self.attribute_key = attribute_key

    def fit(self, X, y=None):
        self.node_transformer.fit(X, y)
        return self

    def transform_single(self, graph: nx.Graph, node_embeddings) -> nx.Graph:
        out_graph = graph.copy()
        for node_idx, node_embedding in zip(out_graph.nodes(), node_embeddings):
            out_graph.nodes[node_idx][self.attribute_key] = node_embedding
        out_graph.graph["source"] = "node_embedder"
        return out_graph

    def transform(self, X, y=None) -> list[nx.Graph]:
        node_embeddings_list = self.node_transformer.transform(X)
        return [
            self.transform_single(graph, node_embeddings)
            for graph, node_embeddings in zip(X, node_embeddings_list)
        ]


def product_graph(graph: nx.Graph, factor_graph: nx.Graph) -> nx.Graph:
    """Return a relabeled Cartesian product graph."""
    product = nx.cartesian_product(graph, factor_graph)
    product = nx.convert_node_labels_to_integers(product)
    relabeled = product.copy()
    for node_idx in relabeled.nodes():
        label_src, label_dst = relabeled.nodes[node_idx]["label"]
        label_src = str(label_src)
        label_dst = str(label_dst)
        if label_src > label_dst:
            label_src, label_dst = label_dst, label_src
        relabeled.nodes[node_idx]["label"] = f"{label_src}:{label_dst}"
    relabeled.graph["source"] = "product_graph"
    return relabeled


class ProductGraphGraphicalizer(GraphicalizerMixin):
    """Take Cartesian products with one or more fixed factor graphs."""

    def __init__(self, factor_graphs: Iterable[nx.Graph]) -> None:
        factor_graph = nx.Graph()
        for graph in factor_graphs:
            factor_graph = nx.disjoint_union(factor_graph, graph)
        self.factor_graph = factor_graph

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [product_graph(graph, self.factor_graph) for graph in X]
