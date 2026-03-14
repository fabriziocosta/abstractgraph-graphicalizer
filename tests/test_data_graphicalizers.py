from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from abstractgraph_graphicalizer.data import (
    DataMatrixGraphicalizer,
    FeatureCorrelationGraphicalizer,
    data_matrix_to_feature_graph,
    data_to_graph,
)


class DataGraphicalizerTest(unittest.TestCase):
    def test_data_matrix_to_feature_graph(self) -> None:
        data = np.array([[1.0, 2.0, 3.0], [1.2, 2.1, 2.9], [0.9, 2.2, 3.1]])
        graph = data_matrix_to_feature_graph(data, 1)
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.graph["source"], "data_matrix")
        self.assertIn("importance", graph.nodes[0])

    def test_data_matrix_graphicalizer_batch(self) -> None:
        X = [
            np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 2.2]]),
            np.array([[0.0, 1.0], [0.2, 1.2], [0.1, 0.9]]),
        ]
        graphs = DataMatrixGraphicalizer(n_edges=1).transform(X)
        self.assertEqual(len(graphs), 2)

    def test_data_to_graph(self) -> None:
        X = np.array(
            [
                [1.0, 2.0, 0.1],
                [1.1, 2.1, 0.0],
                [0.9, 1.9, 0.2],
                [1.2, 2.2, 0.1],
            ]
        )
        graph = data_to_graph(X, max_n_edges=2)
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.graph["source"], "feature_correlation_template")

    def test_feature_correlation_graphicalizer(self) -> None:
        X = np.array(
            [
                [1.0, 2.0, 0.1],
                [1.1, 2.1, 0.0],
                [0.9, 1.9, 0.2],
                [1.2, 2.2, 0.1],
            ]
        )
        graphicalizer = FeatureCorrelationGraphicalizer(max_n_edges=2, eps=0.05).fit(X)
        graphs = graphicalizer.transform(X)
        self.assertEqual(len(graphs), len(X))
        self.assertEqual(graphs[0].graph["source"], "feature_correlation")
        self.assertTrue(any("vec" in data for _, data in graphs[0].nodes(data=True)))


if __name__ == "__main__":
    unittest.main()
