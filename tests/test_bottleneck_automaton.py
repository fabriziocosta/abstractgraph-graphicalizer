from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import torch

from abstractgraph_graphicalizer.bottleneck import (
    BottleneckGraphicalizer,
    aggregate_bottleneck_graphs,
    collapse_graph_to_states,
    edge_recovery_diagnostics,
    evaluate_automaton_recovery,
    extract_sequence_embeddings,
    generate_automaton_sequences,
    sample_hidden_automaton,
    state_assignment_diagnostics,
    train_tiny_sequence_transformer,
    transition_graph_from_assignments,
)


class BottleneckAutomatonTest(unittest.TestCase):
    def test_hidden_automaton_smoke_path(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)
        automaton = sample_hidden_automaton(n_states=3, vocab_size=4, random_state=0)
        data = generate_automaton_sequences(automaton, n_sequences=4, length=8, random_state=1)
        self.assertEqual(len(data["sequences"]), 4)
        self.assertEqual(len(data["hidden_states"]), 4)

        encoder = train_tiny_sequence_transformer(
            data["sequences"],
            vocab_size=automaton.vocab_size,
            d_model=8,
            n_heads=2,
            num_layers=1,
            n_epochs=1,
            device="cpu",
        )
        embeddings = extract_sequence_embeddings(encoder, data["sequences"], device="cpu")
        graphicalizer = BottleneckGraphicalizer(
            d_model=8,
            num_prototypes=5,
            use_encoder=False,
            n_epochs=1,
            gnn_layers=1,
            hidden_dim=8,
            top_k_edges=2,
            device="cpu",
        )
        graphs = graphicalizer.fit_transform(embeddings)
        assignments = [graph.graph["token_to_nodes"] for graph in graphs]
        metrics = evaluate_automaton_recovery(
            assignments,
            data["hidden_states"],
            learned_graph=graphs[0],
            automaton=automaton,
        )
        for key in {
            "clustering_accuracy",
            "purity",
            "ari",
            "nmi",
            "n_learned_nodes",
            "n_true_states",
            "mean_prototype_state_entropy",
            "max_prototypes_per_state",
            "edge_precision",
            "edge_recall",
            "edge_auroc",
            "edge_symmetric_difference",
        }:
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

    def test_assignment_and_edge_diagnostics_for_perfect_recovery(self) -> None:
        hidden = [np.array([0, 1, 2, 0, 1, 2])]
        learned = [np.array([10, 11, 12, 10, 11, 12])]
        diagnostics = state_assignment_diagnostics(learned, hidden)
        self.assertEqual(diagnostics["metrics"]["clustering_accuracy"], 1.0)
        self.assertEqual(diagnostics["metrics"]["ari"], 1.0)
        self.assertEqual(diagnostics["metrics"]["nmi"], 1.0)
        self.assertEqual(diagnostics["prototypes_per_state"], {0: 1, 1: 1, 2: 1})

        transition_graph = transition_graph_from_assignments(learned, tokens=[np.array([0, 1, 2, 0, 1, 2])])
        self.assertIsInstance(transition_graph, nx.DiGraph)
        self.assertTrue(transition_graph.has_edge(10, 11))
        self.assertEqual(transition_graph.edges[10, 11]["edge_type"], "assignment_transition")
        self.assertEqual(transition_graph.edges[10, 11]["top_symbol"], 1)

        collapsed = collapse_graph_to_states(
            transition_graph,
            diagnostics["majority_state"],
            n_states=3,
        )
        true_graph = nx.DiGraph()
        true_graph.add_nodes_from([0, 1, 2])
        true_graph.add_edge(0, 1, probability=1.0)
        true_graph.add_edge(1, 2, probability=1.0)
        true_graph.add_edge(2, 0, probability=1.0)
        edge_metrics = edge_recovery_diagnostics(true_graph, collapsed, n_states=3)
        self.assertEqual(edge_metrics["edge_precision"], 1.0)
        self.assertEqual(edge_metrics["edge_recall"], 1.0)
        self.assertEqual(edge_metrics["edge_auroc"], 1.0)

    def test_aggregate_bottleneck_graphs_preserves_predicted_graph_kind(self) -> None:
        graph = nx.DiGraph()
        graph.add_node(0, prototype_id=10, assignment_mass=2.0)
        graph.add_node(1, prototype_id=11, assignment_mass=1.0)
        graph.add_edge(0, 1, probability=0.75, weight=1.0)
        aggregate = aggregate_bottleneck_graphs([graph], top_k_per_source=1)
        self.assertEqual(aggregate.graph["graph_kind"], "aggregated_predicted_bottleneck_edges")
        self.assertTrue(aggregate.has_edge(10, 11))
        self.assertEqual(aggregate.edges[10, 11]["edge_type"], "predicted_bottleneck_edge")


if __name__ == "__main__":
    unittest.main()
