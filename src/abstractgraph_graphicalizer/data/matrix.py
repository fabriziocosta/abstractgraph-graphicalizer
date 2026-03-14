"""Data-matrix graphicalizers adapted from CoCoGraPE."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy as sp

from abstractgraph_graphicalizer.core import GraphicalizerMixin


def norm_importance_func(data_mtx: np.ndarray) -> np.ndarray:
    return np.linalg.norm(data_mtx, axis=0)


def data_matrix_to_feature_graph(
    data_mtx,
    k: int,
    *,
    importance_func=norm_importance_func,
    labeled: bool = True,
    use_rank_correlation: bool = True,
) -> nx.Graph:
    """Create a feature-dependency graph from a 2D data matrix."""
    data = np.asarray(data_mtx)
    if data.ndim != 2:
        raise ValueError("data_mtx must be a 2D array")
    if data.shape[1] == 0:
        raise ValueError("data_mtx must contain at least one feature column")

    if use_rank_correlation:
        correlations = sp.stats.spearmanr(data, axis=0).correlation
    else:
        correlations = np.corrcoef(data.T)
    correlations = np.asarray(correlations)
    if correlations.ndim == 0:
        correlations = np.ones((data.shape[1], data.shape[1]))
    elif correlations.ndim == 1:
        correlations = np.corrcoef(data.T)
    correlations = np.absolute(correlations)
    correlations = np.nan_to_num(correlations)

    neighbours = np.argsort(-correlations, axis=1)
    importance = importance_func(data)
    n_nodes = len(importance)
    parents_list: list[list[int]] = []
    for feature_idx, neighbour_idxs in enumerate(neighbours):
        feature_parents: list[int] = []
        for neighbour_idx in neighbour_idxs:
            if len(feature_parents) >= k:
                break
            if importance[neighbour_idx] > importance[feature_idx]:
                feature_parents.append(int(neighbour_idx))
        parents_list.append(feature_parents)

    graph = nx.Graph()
    if labeled:
        graph.add_nodes_from(
            (idx, {"label": idx, "importance": float(importance[idx])}) for idx in range(n_nodes)
        )
    else:
        graph.add_nodes_from((idx, {"label": "-"}) for idx in range(n_nodes))
    for node_idx, parents in enumerate(parents_list):
        for parent_idx in parents:
            graph.add_edge(node_idx, parent_idx, label="-", weight=float(correlations[node_idx, parent_idx]))
    graph.graph["source"] = "data_matrix"
    return graph


class DataMatrixGraphicalizer(GraphicalizerMixin):
    """Convert batches of data matrices into feature graphs."""

    def __init__(
        self,
        *,
        importance_func=norm_importance_func,
        n_edges: int = 1,
        labeled: bool = True,
        use_rank_correlation: bool = True,
    ) -> None:
        self.importance_func = importance_func
        self.n_edges = n_edges
        self.labeled = labeled
        self.use_rank_correlation = use_rank_correlation

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [
            data_matrix_to_feature_graph(
                data_mtx,
                self.n_edges,
                importance_func=self.importance_func,
                labeled=self.labeled,
                use_rank_correlation=self.use_rank_correlation,
            )
            for data_mtx in X
        ]


def data_to_graph(
    orig_data_mtx,
    *,
    targets=None,
    max_n_edges: int = 5,
    min_corrcoef: float = 0.9,
    max_corrcoef: float = 0.99,
    min_corrcoef_to_target: float = 0.5,
    eps: float = 1e-6,
) -> nx.Graph:
    """Infer a template graph from a data matrix via feature correlations."""
    data_mtx = np.asarray(orig_data_mtx) + np.random.rand(*np.asarray(orig_data_mtx).shape) * eps
    if targets is not None:
        data_mtx = np.hstack([data_mtx, np.asarray(targets).reshape(-1, 1)])
    corr = np.absolute(np.corrcoef(data_mtx.T))
    if targets is not None:
        node_to_target_corrcoeffs = corr[-1, :-2]
        corr = corr[:-2, :-2]
    min_th = np.quantile(corr, min_corrcoef)
    max_th = np.quantile(corr, max_corrcoef)
    corr[corr < min_th] = 0
    corr[corr > max_th] = 0
    idxs_mtx = np.argsort(corr, axis=1)[:, : corr.shape[1] - max_n_edges]
    for row_idx, idxs in enumerate(idxs_mtx):
        for col_idx in idxs:
            corr[row_idx, col_idx] = 0
    corr = (corr + corr.T) / 2
    corr = corr.astype(bool).astype(int)
    corr = corr - np.diag(np.diag(corr))
    graph = nx.from_numpy_array(corr)
    nx.set_node_attributes(graph, {i: str(i) for i in range(len(corr))}, "label")
    nx.set_edge_attributes(graph, "-", "label")
    if targets is not None:
        node_idxs = np.where(node_to_target_corrcoeffs >= min_corrcoef_to_target)[0]
        graph = graph.subgraph(node_idxs).copy()
    graph.graph["source"] = "feature_correlation_template"
    return graph


class FeatureCorrelationGraphicalizer(GraphicalizerMixin):
    """Fit a correlation template graph, then instantiate sample-specific graphs."""

    def __init__(
        self,
        *,
        max_n_edges: int = 5,
        min_corrcoef: float = 0.9,
        max_corrcoef: float = 0.99,
        min_corrcoef_to_target: float = 0.5,
        eps: float = 1e-1,
        attribute_key: str = "vec",
    ) -> None:
        self.max_n_edges = max_n_edges
        self.min_corrcoef = min_corrcoef
        self.max_corrcoef = max_corrcoef
        self.min_corrcoef_to_target = min_corrcoef_to_target
        self.eps = eps
        self.attribute_key = attribute_key
        self.graph_template_: nx.Graph | None = None

    def fit(self, X, y=None):
        self.graph_template_ = data_to_graph(
            X,
            targets=y,
            max_n_edges=self.max_n_edges,
            min_corrcoef=self.min_corrcoef,
            max_corrcoef=self.max_corrcoef,
            min_corrcoef_to_target=self.min_corrcoef_to_target,
            eps=self.eps,
        )
        return self

    def transform(self, X, y=None) -> list[nx.Graph]:
        if self.graph_template_ is None:
            raise ValueError("FeatureCorrelationGraphicalizer must be fitted before transform")
        graphs: list[nx.Graph] = []
        for row in np.asarray(X):
            graph = nx.Graph(self.graph_template_)
            for idx, val in enumerate(row):
                if idx in graph.nodes():
                    if np.absolute(val) <= self.eps:
                        graph.remove_node(idx)
                    else:
                        graph.nodes[idx][self.attribute_key] = np.array([val])
            graph.graph["source"] = "feature_correlation"
            graphs.append(graph)
        return graphs
