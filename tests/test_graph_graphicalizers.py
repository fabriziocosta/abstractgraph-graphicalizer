from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from abstractgraph_graphicalizer.graph import (
    MutualNearestNeighbourGraphicalizer,
    NearestNeighborVectorGraphicalizer,
    SequenceGraphicalizer,
    StringGraphicalizer,
    mutual_nearest_neighbour_graph,
    sequence_to_graph,
    string_to_graph,
)


class GraphGraphicalizerTest(unittest.TestCase):
    def test_sequence_to_graph_adds_boundary_tokens(self) -> None:
        graph = sequence_to_graph(["A", "B"], start_label="<s>", end_label="</s>")
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.nodes[0]["label"], "<s>")
        self.assertEqual(graph.nodes[3]["label"], "</s>")
        self.assertEqual(graph.edges[0, 1]["label"], "-")
        self.assertEqual(graph.graph["source"], "sequence")

    def test_string_graphicalizer_splits_tokens(self) -> None:
        graphicalizer = StringGraphicalizer(separator="-", start_label="^", end_label="$")
        graph = graphicalizer.transform(["AA-BB"])[0]
        self.assertEqual([graph.nodes[i]["label"] for i in graph.nodes()], ["^", "AA", "BB", "$"])
        self.assertEqual(graph.graph["source"], "string")

    def test_sequence_graphicalizer_builds_graph_batch(self) -> None:
        graphs = SequenceGraphicalizer().transform([["x", "y"], ["z"]])
        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].number_of_nodes(), 2)
        self.assertEqual(graphs[1].nodes[0]["label"], "z")

    def test_mutual_nearest_neighbour_graph_has_distances(self) -> None:
        instance = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]])
        graph = mutual_nearest_neighbour_graph(instance, n_neighbors=2)
        self.assertIsInstance(graph, nx.Graph)
        self.assertGreaterEqual(graph.number_of_edges(), 1)
        self.assertIn("distance", next(iter(graph.edges(data=True)))[2])
        self.assertIn("vec", graph.nodes[0])

    def test_mutual_nearest_neighbour_graphicalizer_batch(self) -> None:
        X = [
            np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [0.0, 1.1], [1.0, 0.0]]),
        ]
        graphs = MutualNearestNeighbourGraphicalizer(n_neighbors=2).transform(X)
        self.assertEqual(len(graphs), 2)
        self.assertTrue(all(graph.graph["source"] == "mutual_nearest_neighbour" for graph in graphs))

    def test_nearest_neighbor_vector_graphicalizer(self) -> None:
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [1.0, 1.0],
            ]
        )
        graphs = NearestNeighborVectorGraphicalizer(
            instance_n_neighbors=3,
            connectivity_n_neighbors=2,
            discretization_factor=0.5,
        ).transform(X)
        self.assertEqual(len(graphs), len(X))
        self.assertEqual(graphs[0].graph["source"], "nearest_neighbour_vector")
        self.assertIn("vec", graphs[0].nodes[0])


if __name__ == "__main__":
    unittest.main()
