from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import torch

from abstractgraph_graphicalizer import BottleneckGraphicalizer as TopLevelBottleneckGraphicalizer
from abstractgraph_graphicalizer.bottleneck import (
    BottleneckGraphicalizer,
    BottleneckOutput,
    GraphInterpretationBottleneck,
    bottleneck_output_to_networkx,
)


class BottleneckGraphicalizerTest(unittest.TestCase):
    def test_imports_and_top_level_exports(self) -> None:
        self.assertIs(TopLevelBottleneckGraphicalizer, BottleneckGraphicalizer)
        self.assertTrue(GraphInterpretationBottleneck)

    def test_forward_shapes_and_losses(self) -> None:
        torch.manual_seed(0)
        model = GraphInterpretationBottleneck(
            d_model=8,
            num_prototypes=6,
            top_k_edges=2,
            gnn_layers=1,
            hidden_dim=8,
        )
        output = model(torch.randn(5, 8))
        self.assertEqual(output.node_embeddings.shape, (6, 8))
        self.assertEqual(output.adjacency.shape, (6, 6))
        self.assertEqual(output.assignments.shape, (5, 6))
        self.assertEqual(output.token_to_nodes.shape, (5,))
        for key in {
            "loss",
            "loss_node",
            "loss_token",
            "loss_sparse",
            "loss_binary",
            "loss_entropy",
            "loss_transition",
            "loss_balance",
        }:
            self.assertIn(key, output.losses)
            self.assertEqual(output.losses[key].dim(), 0)
            self.assertTrue(torch.isfinite(output.losses[key]))

    def test_top_k_edge_sparsity(self) -> None:
        torch.manual_seed(1)
        model = GraphInterpretationBottleneck(
            d_model=4,
            num_prototypes=5,
            top_k_edges=2,
            edge_threshold=0.0,
            gnn_layers=1,
            hidden_dim=4,
        )
        output = model(torch.randn(7, 4))
        outgoing = (output.adjacency.detach() > 0).sum(dim=1)
        self.assertTrue(torch.all(outgoing <= 2))

    def test_networkx_conversion_active_only_and_graph_kind(self) -> None:
        output = BottleneckOutput(
            node_embeddings=torch.randn(3, 4),
            adjacency=torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            edge_probabilities=torch.tensor(
                [
                    [0.0, 0.9, 0.0],
                    [0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0],
                ]
            ),
            assignments=torch.tensor(
                [
                    [0.7, 0.3, 0.0],
                    [0.6, 0.4, 0.0],
                ]
            ),
            node_types=torch.arange(3),
            token_to_nodes=torch.tensor([0, 0]),
            active_node_mask=torch.tensor([True, True, False]),
            losses={},
            tokens=["a", "b"],
            input_id="demo",
            metadata={"kind": "test"},
        )
        directed = bottleneck_output_to_networkx(output, output_graph="directed")
        self.assertIsInstance(directed, nx.DiGraph)
        self.assertEqual(directed.number_of_nodes(), 2)
        self.assertEqual(directed.number_of_edges(), 1)
        self.assertEqual(directed.graph["source"], "graph_interpretation_bottleneck")
        self.assertEqual(directed.graph["graph_kind"], "predicted_bottleneck_edges")
        self.assertIn("losses", directed.graph)
        self.assertEqual(directed.graph["tokens"], ["a", "b"])
        self.assertIn("prototype_id", directed.nodes[0])
        self.assertIn("probability", directed.edges[0, 1])

        undirected = bottleneck_output_to_networkx(output, output_graph="undirected")
        self.assertIsInstance(undirected, nx.Graph)
        self.assertNotIsInstance(undirected, nx.DiGraph)

    def test_fit_transform_on_tiny_numeric_inputs(self) -> None:
        rng = np.random.default_rng(0)
        X = [rng.normal(size=(5, 3)), rng.normal(size=(4, 3))]
        graphicalizer = BottleneckGraphicalizer(
            d_model=6,
            num_prototypes=4,
            n_epochs=1,
            gnn_layers=1,
            hidden_dim=6,
            top_k_edges=2,
            device="cpu",
        )
        graphs = graphicalizer.fit_transform(X)
        self.assertEqual(len(graphs), 2)
        self.assertTrue(all(isinstance(graph, nx.DiGraph) for graph in graphs))
        self.assertTrue(all(graph.graph["source"] == "graph_interpretation_bottleneck" for graph in graphs))
        self.assertEqual(len(graphicalizer.training_history_), 1)
        self.assertIn("loss_transition", graphicalizer.training_history_[0])


if __name__ == "__main__":
    unittest.main()
