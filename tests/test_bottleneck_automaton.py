from __future__ import annotations

import unittest

import numpy as np
import torch

from abstractgraph_graphicalizer.bottleneck import (
    BottleneckGraphicalizer,
    evaluate_automaton_recovery,
    extract_sequence_embeddings,
    generate_automaton_sequences,
    sample_hidden_automaton,
    train_tiny_sequence_transformer,
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
            "n_learned_nodes",
            "n_true_states",
            "edge_precision",
            "edge_recall",
        }:
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))


if __name__ == "__main__":
    unittest.main()
