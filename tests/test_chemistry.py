from __future__ import annotations

import tempfile
import unittest
import pickle
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from abstractgraph_graphicalizer.chem import (
    CHEM_EDGE_SCHEMA,
    CHEM_NODE_SCHEMA,
    DEFAULT_ZINC_TARGET_COLUMNS,
    MoleculeParseError,
    PubChemLoader,
    SupervisedDataSetLoader,
    ZINCLoader,
    build_zinc_graph_corpus,
    bundled_pubchem_root,
    bundled_zinc_root,
    default_pubchem_root,
    default_zinc_root,
    draw_graph,
    draw_molecule,
    extract_zinc_targets,
    graph_to_rdmol,
    load_pubchem_graph_dataset,
    load_zinc_graph_dataset,
    normalize_graph_schema,
    local_pubchem_root,
    local_zinc_root,
    pubchem_search_roots,
    sdf_to_graphs,
    smi_to_graphs,
    smiles_list_to_graphs,
    smiles_to_graph,
    zinc_search_roots,
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

    def test_normalize_graph_schema_upgrades_legacy_labels(self) -> None:
        graph = nx.Graph()
        graph.add_node(0, label="C")
        graph.add_node(1, label="O")
        graph.add_edge(0, 1, label="1")

        normalized = normalize_graph_schema(graph)

        self.assertEqual(normalized.edges[(0, 1)]["label"], "single")
        self.assertEqual(normalized.edges[(0, 1)]["bond_order"], 1.0)
        self.assertEqual(normalized.edges[(0, 1)]["bond_type"], "SINGLE")

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

            active_graphs, inactive_graphs = loader.load_split(624249, limit=1)
            self.assertEqual(len(active_graphs), 1)
            self.assertEqual(len(inactive_graphs), 1)

            graphs, targets = loader.load(624249, limit=1)
            self.assertEqual(len(graphs), 2)
            self.assertEqual(targets, [0, 1])

            active_graphs, inactive_graphs = loader.load_split(
                624249,
                limit=1,
                limit_active=2,
            )
            self.assertEqual(len(active_graphs), 2)
            self.assertEqual(len(inactive_graphs), 1)

    def test_pubchem_loader_root_resolution_helpers(self) -> None:
        roots = pubchem_search_roots()
        self.assertIn(local_pubchem_root().resolve(), roots)
        self.assertIn(bundled_pubchem_root().resolve(), roots)
        expected_default = (
            local_pubchem_root().resolve()
            if local_pubchem_root().resolve().exists()
            else bundled_pubchem_root().resolve()
        )
        self.assertEqual(default_pubchem_root(), expected_default)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"ABSTRACTGRAPH_PUBCHEM_ROOT": tmpdir}, clear=False):
                self.assertEqual(default_pubchem_root(), Path(tmpdir).resolve())

    def test_pubchem_loader_lists_assay_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_path = root / "AID1706_active.sdf"
            inactive_path = root / "AID1706_inactive.sdf"

            active_writer = Chem.SDWriter(str(active_path))
            active_writer.write(Chem.MolFromSmiles("CCO"))
            active_writer.close()

            inactive_writer = Chem.SDWriter(str(inactive_path))
            inactive_writer.write(Chem.MolFromSmiles("C=C"))
            inactive_writer.close()

            loader = PubChemLoader(root)
            summaries = loader.list_assays()
            self.assertEqual(len(summaries), 1)
            summary = summaries[0]
            self.assertEqual(summary.assay_id, "1706")
            self.assertEqual(summary.active_path, active_path)
            self.assertEqual(summary.inactive_path, inactive_path)
            self.assertGreater(summary.active_size_bytes, 0)
            self.assertGreater(summary.inactive_size_bytes, 0)
            self.assertEqual(summary.active_molecule_count, 1)
            self.assertEqual(summary.inactive_molecule_count, 1)
            self.assertEqual(summary.total_molecule_count, 2)
            self.assertEqual(
                summary.total_size_bytes,
                summary.active_size_bytes + summary.inactive_size_bytes,
            )

    def test_pubchem_loader_formats_assay_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for assay_id, smiles in {
                "1706": ("CCO", "C=C"),
                "2631": ("CCN", "C#N"),
            }.items():
                active_path = root / f"AID{assay_id}_active.sdf"
                inactive_path = root / f"AID{assay_id}_inactive.sdf"
                active_writer = Chem.SDWriter(str(active_path))
                active_writer.write(Chem.MolFromSmiles(smiles[0]))
                active_writer.close()
                inactive_writer = Chem.SDWriter(str(inactive_path))
                inactive_writer.write(Chem.MolFromSmiles(smiles[1]))
                inactive_writer.close()

            loader = PubChemLoader(root)
            table = loader.format_assay_table()
            self.assertIn("assay_id", table)
            self.assertIn("1706", table)
            self.assertIn("2631", table)
            self.assertIn("total_mols", table)

    def test_zinc_loader_reads_csv_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "zinc_small.csv"
            csv_path.write_text(
                "zinc_id,smiles,logP,qed,SAS\n"
                "z1,CCO,1.0,0.5,2.0\n"
                "z2,C=C,1.2,0.4,2.1\n"
            )

            loader = ZINCLoader(root)
            paths = loader.resolve_paths("zinc_small")
            self.assertEqual(paths.dataset_name, "zinc_small")
            self.assertEqual(paths.csv_path, csv_path)

            frame = loader.load_frame("zinc_small", limit=1)
            self.assertEqual(frame["zinc_id"].tolist(), ["z1"])

            graphs, metadata = loader.load("zinc_small")
            self.assertEqual(len(graphs), 2)
            self.assertEqual(metadata["zinc_id"].tolist(), ["z1", "z2"])
            self.assertEqual(graphs[0].graph["zinc_dataset"], "zinc_small")
            self.assertEqual(graphs[0].graph["zinc_id"], "z1")
            self.assertEqual(graphs[0].graph["source"], "zinc")
            self.assertTrue(
                (root / "graph_corpus_cache" / "zinc_small" / "graph_corpus" / "manifest.pkl").exists()
            )

    def test_zinc_loader_skip_invalid_smiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "zinc_small.csv"
            csv_path.write_text(
                "zinc_id,smiles,logP\n"
                "z1,CCO,1.0\n"
                "z2,not-a-smiles,1.1\n"
                "z3,C#N,1.2\n"
            )

            loader = ZINCLoader(root, on_error="skip")
            graphs, metadata = loader.load("zinc_small")

            self.assertEqual(len(graphs), 2)
            self.assertEqual(metadata["zinc_id"].tolist(), ["z1", "z3"])

    def test_zinc_loader_reuses_graph_corpus_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "zinc_small.csv").write_text(
                "zinc_id,smiles,logP\n"
                "z1,CCO,1.0\n"
                "z2,C=C,1.1\n"
            )

            loader = ZINCLoader(root)
            graphs, metadata = loader.load("zinc_small", min_node_count=2, max_node_count=2)

            self.assertEqual(len(graphs), 1)
            self.assertEqual(metadata["zinc_id"].tolist(), ["z2"])

            with patch.object(loader, "_graph_from_row", side_effect=AssertionError("cache not reused")):
                cached_graphs, cached_metadata = loader.load("zinc_small", min_node_count=3, max_node_count=3)

            self.assertEqual(len(cached_graphs), 1)
            self.assertEqual(cached_metadata["zinc_id"].tolist(), ["z1"])

    def test_zinc_loader_uses_separate_caches_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "zinc_a.csv").write_text("zinc_id,smiles\nza,CCO\n")
            (root / "zinc_b.csv").write_text("zinc_id,smiles\nzb,C=C\n")

            loader = ZINCLoader(root)
            graphs_a, metadata_a = loader.load("zinc_a")
            graphs_b, metadata_b = loader.load("zinc_b")

            self.assertEqual(metadata_a["zinc_id"].tolist(), ["za"])
            self.assertEqual(metadata_b["zinc_id"].tolist(), ["zb"])
            self.assertEqual(len(graphs_a), 1)
            self.assertEqual(len(graphs_b), 1)
            self.assertTrue(
                (root / "graph_corpus_cache" / "zinc_a" / "graph_corpus" / "manifest.pkl").exists()
            )
            self.assertTrue(
                (root / "graph_corpus_cache" / "zinc_b" / "graph_corpus" / "manifest.pkl").exists()
            )

    def test_zinc_loader_root_resolution_helpers(self) -> None:
        roots = zinc_search_roots()
        self.assertIn(local_zinc_root().resolve(), roots)
        self.assertIn(bundled_zinc_root().resolve(), roots)
        expected_default = (
            local_zinc_root().resolve()
            if local_zinc_root().resolve().exists()
            else bundled_zinc_root().resolve()
        )
        self.assertEqual(default_zinc_root(), expected_default)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"ABSTRACTGRAPH_ZINC_ROOT": tmpdir}, clear=False):
                self.assertEqual(default_zinc_root(), Path(tmpdir).resolve())

    def test_zinc_loader_formats_dataset_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "zinc_a.csv").write_text("zinc_id,smiles\nz1,CCO\n")
            (root / "zinc_b.csv").write_text("zinc_id,smiles\nz2,C=C\nz3,C#N\n")

            loader = ZINCLoader(root)
            table = loader.format_dataset_table()

            self.assertIn("dataset", table)
            self.assertIn("zinc_a", table)
            self.assertIn("zinc_b", table)
            self.assertIn("molecules", table)

    def test_supervised_dataset_loader_equalizes_and_resizes(self) -> None:
        def load_func():
            return ["a", "b", "c", "d"], np.asarray([0, 0, 1, 1])

        data, targets = SupervisedDataSetLoader(
            load_func=load_func,
            size=2,
            use_equalized=True,
            random_state=0,
        ).load()

        self.assertEqual(len(data), 2)
        self.assertEqual(sorted(np.unique(targets).tolist()), [0, 1])

    def test_load_pubchem_graph_dataset_filters_node_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_path = root / "AID624249_active.sdf"
            inactive_path = root / "AID624249_inactive.sdf"

            active_writer = Chem.SDWriter(str(active_path))
            active_writer.write(Chem.MolFromSmiles("CCO"))
            active_writer.close()

            inactive_writer = Chem.SDWriter(str(inactive_path))
            inactive_writer.write(Chem.MolFromSmiles("C1=CC=CC=C1"))
            inactive_writer.close()

            graphs, targets, metadata = load_pubchem_graph_dataset(
                root,
                assay_id="624249",
                max_node_count=3,
            )

            self.assertEqual(len(graphs), 1)
            self.assertEqual(targets.tolist(), [1])
            self.assertEqual(metadata["node_count"].tolist(), [3])

    def test_build_and_load_zinc_graph_corpus_normalizes_legacy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "zinc_small.csv"
            pd.DataFrame(
                [
                    {"zinc_id": "z1", "smiles": "CCO", "logP": 1.0, "qed": 0.5, "SAS": 2.0},
                    {"zinc_id": "z2", "smiles": "C=C", "logP": 1.2, "qed": 0.4, "SAS": 2.1},
                ]
            ).to_csv(csv_path, index=False)

            manifest = build_zinc_graph_corpus(root, csv_path)
            bucket_path = Path(manifest["bucket_files"][2])
            with bucket_path.open("rb") as handle:
                items = pickle.load(handle)
            legacy_graph = items[0][0]
            for _, _, data in legacy_graph.edges(data=True):
                data["label"] = "1"
                data.pop("bond_order", None)
                data.pop("bond_type", None)
            with bucket_path.open("wb") as handle:
                pickle.dump(items, handle)

            graphs, metadata = load_zinc_graph_dataset(root, max_molecules=10)

            self.assertEqual(len(graphs), 2)
            self.assertEqual(sorted(metadata["zinc_id"].tolist()), ["z1", "z2"])
            self.assertEqual(next(iter(graphs[0].edges(data=True)))[2]["label"], "single")

    def test_extract_zinc_targets_uses_default_columns(self) -> None:
        metadata = pd.DataFrame([{"logP": 1.0, "qed": 0.5, "SAS": 2.0, "extra": 7}])
        targets = extract_zinc_targets(metadata)
        self.assertEqual(targets.columns.tolist(), list(DEFAULT_ZINC_TARGET_COLUMNS))


if __name__ == "__main__":
    unittest.main()
