from __future__ import annotations

import unittest

import networkx as nx

from abstractgraph_graphicalizer.chem import MoleculeParseError, graph_to_rdmol, smiles_to_graph
from abstractgraph_graphicalizer.chem.molecules import Chem


@unittest.skipIf(Chem is None, "RDKit not installed")
class ChemistryTest(unittest.TestCase):
    def test_smiles_to_graph_has_labels(self) -> None:
        graph = smiles_to_graph("CCO")
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.nodes[0]["label"], "C")
        self.assertIn("label", next(iter(graph.edges(data=True)))[2])

    def test_invalid_smiles_raises(self) -> None:
        with self.assertRaises(MoleculeParseError):
            smiles_to_graph("not-a-smiles")

    def test_round_trip_supported_graph(self) -> None:
        graph = smiles_to_graph("C=C")
        mol = graph_to_rdmol(graph)
        self.assertEqual(mol.GetNumAtoms(), 2)


if __name__ == "__main__":
    unittest.main()
