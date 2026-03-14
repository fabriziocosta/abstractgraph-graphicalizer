from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from abstractgraph_graphicalizer.graph import (
    NodeEmbedderGraphGraphicalizer,
    NormalizedLaplacianSVDGraphGraphicalizer,
    ProductGraphGraphicalizer,
    annotate_normalized_laplacian_svd,
    normalized_laplacian_svd,
    product_graph,
)


class _DummyNodeTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[np.array([float(node_idx)]) for node_idx in graph.nodes()] for graph in X]


class GraphAnnotationGraphicalizerTest(unittest.TestCase):
    def test_normalized_laplacian_svd_has_fixed_width(self) -> None:
        graph = nx.path_graph(4)
        embeddings = normalized_laplacian_svd(graph, 3)
        self.assertEqual(embeddings.shape, (4, 3))

    def test_annotate_normalized_laplacian_svd_appends_existing_features(self) -> None:
        graph = nx.path_graph(3)
        for node in graph.nodes():
            graph.nodes[node]["vec"] = np.array([1.0])
        annotated = annotate_normalized_laplacian_svd(graph, n_components=2)
        self.assertEqual(annotated.nodes[0]["vec"].shape[0], 3)
        self.assertEqual(annotated.graph["source"], "normalized_laplacian_svd")

    def test_normalized_laplacian_graphicalizer(self) -> None:
        graphs = NormalizedLaplacianSVDGraphGraphicalizer(n_components=2).transform([nx.path_graph(3)])
        self.assertEqual(len(graphs), 1)
        self.assertEqual(graphs[0].nodes[0]["vec"].shape[0], 2)

    def test_node_embedder_graphicalizer(self) -> None:
        graph = nx.path_graph(3)
        embedded = NodeEmbedderGraphGraphicalizer(_DummyNodeTransformer()).fit_transform([graph])[0]
        self.assertEqual(embedded.nodes[2]["pred"][0], 2.0)
        self.assertEqual(embedded.graph["source"], "node_embedder")

    def test_product_graph(self) -> None:
        graph_a = nx.path_graph(2)
        graph_b = nx.path_graph(2)
        nx.set_node_attributes(graph_a, {0: "A", 1: "B"}, "label")
        nx.set_node_attributes(graph_b, {0: "X", 1: "Y"}, "label")
        out = product_graph(graph_a, graph_b)
        self.assertGreater(out.number_of_nodes(), 0)
        self.assertIn(":", out.nodes[0]["label"])

    def test_product_graphicalizer(self) -> None:
        factor = nx.path_graph(2)
        nx.set_node_attributes(factor, {0: "F0", 1: "F1"}, "label")
        graph = nx.path_graph(2)
        nx.set_node_attributes(graph, {0: "G0", 1: "G1"}, "label")
        graphs = ProductGraphGraphicalizer([factor]).transform([graph])
        self.assertEqual(len(graphs), 1)
        self.assertEqual(graphs[0].graph["source"], "product_graph")


if __name__ == "__main__":
    unittest.main()
