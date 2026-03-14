"""Vector-to-graph graphicalizers adapted from CoCoGraPE."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph

from abstractgraph_graphicalizer.core import GraphicalizerMixin


def _pairwise_distance_matrix(instance: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if len(instance) == 0:
        raise ValueError("instance must contain at least one row")
    distances = squareform(pdist(instance, metric=metric))
    np.fill_diagonal(distances, math.inf)
    return distances


def mutual_nearest_neighbour_graph(
    instance,
    *,
    n_neighbors: int = 5,
    n_dense_links: int = 1,
    edge_label: str = "-",
    metric: str = "euclidean",
    attribute_key: str = "vec",
) -> nx.Graph:
    """Build a mutual-nearest-neighbour graph from a 2D feature matrix."""
    instance_array = np.asarray(instance)
    if instance_array.ndim != 2:
        raise ValueError("instance must be a 2D array of shape (n_nodes, n_features)")

    pdists = _pairwise_distance_matrix(instance_array, metric=metric)
    density = np.sum(1 / pdists, axis=1).flatten()
    nearest = np.argsort(pdists, axis=1)
    k_nearest = nearest[:, :n_neighbors]

    mask = np.zeros(pdists.shape, dtype=bool)
    for mask_row, neighbours_row in zip(mask, k_nearest):
        mask_row[neighbours_row] = True
    mask &= mask.T

    graph_mask = csr_matrix(mask)
    for idx in range(len(instance_array)):
        dense_links_counter = 0
        for nb_idx in nearest[idx]:
            if density[nb_idx] > density[idx]:
                graph_mask[idx, nb_idx] = True
                graph_mask[nb_idx, idx] = True
                dense_links_counter += 1
            if dense_links_counter >= n_dense_links:
                break

    weighted_distances = pdists.copy()
    weighted_distances[~graph_mask.toarray()] = 0
    graph = nx.from_scipy_sparse_array(csr_matrix(weighted_distances), edge_attribute="distance")
    for node_idx in graph.nodes():
        graph.nodes[node_idx]["label"] = node_idx
        graph.nodes[node_idx][attribute_key] = instance_array[node_idx]
    nx.set_edge_attributes(graph, values=edge_label, name="label")
    graph.graph["source"] = "mutual_nearest_neighbour"
    return graph


class MutualNearestNeighbourGraphicalizer(GraphicalizerMixin):
    """Convert batches of instance matrices into mutual-nearest-neighbour graphs."""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        n_dense_links: int = 1,
        edge_label: str = "-",
        metric: str = "euclidean",
        attribute_key: str = "vec",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_dense_links = n_dense_links
        self.edge_label = edge_label
        self.metric = metric
        self.attribute_key = attribute_key

    def transform_single(self, instance) -> nx.Graph:
        return mutual_nearest_neighbour_graph(
            instance,
            n_neighbors=self.n_neighbors,
            n_dense_links=self.n_dense_links,
            edge_label=self.edge_label,
            metric=self.metric,
            attribute_key=self.attribute_key,
        )

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [self.transform_single(instance) for instance in X]


class NearestNeighborVectorGraphicalizer(GraphicalizerMixin):
    """Build local kNN graphs from a 2D dataset, one graph per instance neighborhood."""

    def __init__(
        self,
        *,
        instance_n_neighbors: int,
        connectivity_n_neighbors: int,
        discretization_factor: float,
        attribute_key: str = "vec",
        edge_label: str = "-",
    ) -> None:
        self.instance_n_neighbors = instance_n_neighbors
        self.connectivity_n_neighbors = connectivity_n_neighbors
        self.discretization_factor = discretization_factor
        self.attribute_key = attribute_key
        self.edge_label = edge_label

    def transform(self, X, y=None) -> list[nx.Graph]:
        data = np.asarray(X)
        if data.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        neighbourhood_graph = kneighbors_graph(
            data,
            self.instance_n_neighbors,
            mode="connectivity",
            include_self=True,
        )
        neighbourhoods = [data[row.nonzero()[1]] for row in neighbourhood_graph]

        graphs: list[nx.Graph] = []
        for neighbourhood in neighbourhoods:
            graph = nx.from_scipy_sparse_array(
                kneighbors_graph(
                    neighbourhood,
                    self.connectivity_n_neighbors,
                    mode="connectivity",
                    include_self=True,
                )
            )
            for node_idx in graph.nodes():
                graph.nodes[node_idx]["label"] = int(node_idx * self.discretization_factor)
                graph.nodes[node_idx][self.attribute_key] = neighbourhood[node_idx]
            nx.set_edge_attributes(graph, values=self.edge_label, name="label")
            graph.graph["source"] = "nearest_neighbour_vector"
            graphs.append(graph)
        return graphs
