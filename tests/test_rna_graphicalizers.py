from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from abstractgraph_graphicalizer.rna import (
    RNAFoldGraphicalizer,
    RNASequenceGraphicalizer,
    SequenceReverseComplementGraphicalizer,
    make_reverse_complement_graph,
    read_fasta,
    rnafold_to_graphs,
    seq_struct_to_graph,
    seq_to_graph,
    sequence_dotbracket_to_graph,
)


class RNAGraphicalizerTest(unittest.TestCase):
    def test_sequence_dotbracket_to_graph(self) -> None:
        graph = sequence_dotbracket_to_graph("AUGC", "(..)")
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.edges[0, 1]["label"], "bk")
        self.assertEqual(graph.edges[0, 3]["label"], "bp")

    def test_seq_struct_to_graph_labels_secondary_structure(self) -> None:
        graph = seq_struct_to_graph("rna1", "AUGC", "(..)", mode="non_stem")
        self.assertEqual(graph.graph["source"], "rna_structure")
        self.assertEqual(graph.nodes[0]["label"], "S")
        self.assertEqual(graph.nodes[1]["secondary_structure"], "unpaired")

    def test_seq_to_graph(self) -> None:
        graph = seq_to_graph("rna2", "AUGC")
        self.assertEqual(graph.graph["source"], "rna_sequence")
        self.assertEqual(graph.number_of_edges(), 3)

    def test_read_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seqs.fasta"
            path.write_text(">a\nAUGC\n>b\nTTAA\n")
            records = list(read_fasta(path))
        self.assertEqual(records[0], ("a", "AUGC"))
        self.assertEqual(records[1], ("b", "UUAA"))

    def test_rnafold_to_graphs_falls_back_to_sequence(self) -> None:
        with patch(
            "abstractgraph_graphicalizer.rna.graphs.rnafold_wrapper",
            side_effect=FileNotFoundError("RNAfold not found"),
        ):
            graphs = rnafold_to_graphs([("a", "AUGC")], on_error="sequence")
        self.assertEqual(graphs[0].graph["source"], "rna_sequence")

    def test_rnafold_to_graphs_uses_wrapper(self) -> None:
        with patch("abstractgraph_graphicalizer.rna.graphs.rnafold_wrapper", return_value=("AUGC", "(..)")):
            graph = rnafold_to_graphs([("a", "AUGC")], on_error="raise")[0]
        self.assertEqual(graph.graph["source"], "rnafold")
        self.assertEqual(graph.edges[0, 3]["label"], "bp")

    def test_reverse_complement_graph(self) -> None:
        graph = make_reverse_complement_graph("AUGCAU", min_k=2, max_k=2)
        self.assertIsInstance(graph, nx.Graph)
        self.assertTrue(any(data["label"] == "rc" for _, _, data in graph.edges(data=True)))

    def test_graphicalizer_classes(self) -> None:
        seq_graph = RNASequenceGraphicalizer().transform([("a", "AUGC")])[0]
        fold_graph = RNAFoldGraphicalizer(on_error="sequence").transform([("a", "AUGC")])[0]
        rc_graph = SequenceReverseComplementGraphicalizer(min_k=2, max_k=2).transform(["AUGCAU"])[0]
        self.assertEqual(seq_graph.graph["source"], "rna_sequence")
        self.assertIn(fold_graph.graph["source"], {"rna_sequence", "rnafold"})
        self.assertEqual(rc_graph.graph["source"], "reverse_complement")


if __name__ == "__main__":
    unittest.main()
