from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from abstractgraph_graphicalizer.chem import (
    CHEM_EDGE_SCHEMA,
    CHEM_NODE_SCHEMA,
    MoleculeParseError,
    PubChemLoader,
    bundled_pubchem_root,
    default_pubchem_root,
    draw_graph,
    draw_molecule,
    graph_to_rdmol,
    local_pubchem_root,
    pubchem_search_roots,
    sdf_to_graphs,
    smi_to_graphs,
    smiles_list_to_graphs,
    smiles_to_graph,
)
from abstractgraph_graphicalizer.chem.molecules import Chem


@unittest.skipIf(Chem is None, "RDKit not installed")
class ChemistryTest(unittest.TestCase):
    def test_smiles_to_graph_has_labels(self) -> None:
        graph = smiles_to_graph("CCO")
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.nodes[0]["label"], "C")
        self.assertIn("label", next(iter(graph.edges(data=True)))[2])
        self.assertEqual(graph.graph["source"], "smiles")
        self.assertIn("label", CHEM_NODE_SCHEMA)
        self.assertIn("label", CHEM_EDGE_SCHEMA)

    def test_invalid_smiles_raises(self) -> None:
        with self.assertRaises(MoleculeParseError):
            smiles_to_graph("not-a-smiles")

    def test_round_trip_supported_graph(self) -> None:
        graph = smiles_to_graph("C=C")
        mol = graph_to_rdmol(graph)
        self.assertEqual(mol.GetNumAtoms(), 2)
        self.assertEqual(str(mol.GetBondWithIdx(0).GetBondType()), "DOUBLE")

    def test_smiles_list_skip_invalid_records(self) -> None:
        graphs = smiles_list_to_graphs(["CCO", "not-a-smiles", "C#N"], on_error="skip")
        self.assertEqual(len(graphs), 2)

    def test_invalid_on_error_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            smiles_list_to_graphs(["CCO"], on_error="ignore")

    def test_smi_reader_uses_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "molecules.smi"
            path.write_text("CCO ethanol\nnot-a-smiles broken\nC#N cyanide\n")
            graphs = list(smi_to_graphs(path, on_error="skip"))
            self.assertEqual(len(graphs), 2)
            self.assertEqual(graphs[0].graph["source"], "smi")

    def test_sdf_reader_and_draw_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "molecules.sdf"
            writer = Chem.SDWriter(str(path))
            writer.write(Chem.MolFromSmiles("CCO"))
            writer.write(Chem.MolFromSmiles("C=C"))
            writer.close()

            graphs = list(sdf_to_graphs(path))
            self.assertEqual(len(graphs), 2)
            self.assertEqual(graphs[1].graph["source"], "sdf")

            image = draw_molecule(graphs[0])
            ax = draw_graph(graphs[1])
            self.assertTrue(hasattr(image, "size"))
            self.assertTrue(hasattr(ax, "set_axis_off"))

    def test_pubchem_loader_reads_active_and_inactive_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_path = root / "AID624249_active.sdf"
            inactive_path = root / "AID624249_inactive.sdf"

            active_writer = Chem.SDWriter(str(active_path))
            active_writer.write(Chem.MolFromSmiles("CCO"))
            active_writer.write(Chem.MolFromSmiles("CCN"))
            active_writer.close()

            inactive_writer = Chem.SDWriter(str(inactive_path))
            inactive_writer.write(Chem.MolFromSmiles("C=C"))
            inactive_writer.close()

            loader = PubChemLoader(root)
            paths = loader.resolve_paths("AID624249")
            self.assertEqual(paths.assay_id, "624249")

            active_graphs, inactive_graphs = loader.load_split(624249)
            self.assertEqual(len(active_graphs), 2)
            self.assertEqual(len(inactive_graphs), 1)
            self.assertEqual(active_graphs[0].graph["pubchem_activity"], "active")
            self.assertEqual(inactive_graphs[0].graph["pubchem_activity"], "inactive")
            self.assertEqual(active_graphs[0].graph["target"], 1)
            self.assertEqual(inactive_graphs[0].graph["target"], 0)

            graphs, targets = loader.load(624249)
            self.assertEqual(len(graphs), 3)
            self.assertEqual(targets, [0, 1, 1])

    def test_pubchem_loader_root_resolution_helpers(self) -> None:
        roots = pubchem_search_roots()
        self.assertIn(local_pubchem_root().resolve(), roots)
        self.assertIn(bundled_pubchem_root().resolve(), roots)
        self.assertEqual(default_pubchem_root(), local_pubchem_root().resolve())

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"ABSTRACTGRAPH_PUBCHEM_ROOT": tmpdir}, clear=False):
                self.assertEqual(default_pubchem_root(), Path(tmpdir).resolve())


if __name__ == "__main__":
    unittest.main()
