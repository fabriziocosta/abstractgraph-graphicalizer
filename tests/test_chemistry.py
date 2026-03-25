from __future__ import annotations

import tempfile
import unittest
import pickle
import gzip
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from abstractgraph_graphicalizer.chem import (
    CHEM_EDGE_SCHEMA,
    CHEM_NODE_SCHEMA,
    DEFAULT_GDB_MODE,
    DEFAULT_QM9_TARGET_COLUMNS,
    DEFAULT_ZINC_TARGET_COLUMNS,
    GDBLoader,
    MoleculeParseError,
    PubChemAssayLoader,
    QM9Loader,
    SupervisedDataSetLoader,
    ZINCLoader,
    build_gdb_graph_corpus,
    build_qm9_graph_corpus,
    build_zinc_graph_corpus,
    bundled_gdb_root,
    bundled_pubchem_root,
    bundled_qm9_root,
    bundled_zinc_root,
    default_gdb_root,
    default_pubchem_root,
    default_qm9_root,
    default_zinc_root,
    download_gdb_archive,
    download_qm9_dataset,
    draw_graph,
    draw_molecule,
    extract_qm9_targets,
    extract_zinc_targets,
    graph_to_rdmol,
    gdb_cache_root,
    iter_gdb_records,
    load_gdb_graph_dataset,
    load_pubchem_graph_dataset,
    load_qm9_graph_dataset,
    load_zinc_graph_dataset,
    local_gdb_root,
    normalize_graph_schema,
    read_gdb_metadata,
    resolve_gdb_mode,
    gdb_search_roots,
    local_pubchem_root,
    local_qm9_root,
    local_zinc_root,
    pubchem_search_roots,
    qm9_search_roots,
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

            loader = PubChemAssayLoader(root)
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
            self.assertTrue(
                (root / "graph_corpus_cache" / "AID624249" / "graph_corpus" / "manifest.pkl").exists()
            )

    def test_pubchem_loader_reuses_graph_corpus_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active_path = root / "AID624249_active.sdf"
            inactive_path = root / "AID624249_inactive.sdf"

            active_writer = Chem.SDWriter(str(active_path))
            active_writer.write(Chem.MolFromSmiles("CCO"))
            active_writer.close()

            inactive_writer = Chem.SDWriter(str(inactive_path))
            inactive_writer.write(Chem.MolFromSmiles("C=C"))
            inactive_writer.close()

            loader = PubChemAssayLoader(root)
            graphs, targets = loader.load("624249")
            self.assertEqual(len(graphs), 2)
            self.assertEqual(targets, [0, 1])

            with patch(
                "abstractgraph_graphicalizer.chem.pubchem.sdf_to_graphs",
                side_effect=AssertionError("cache not reused"),
            ):
                cached_graphs, cached_targets = loader.load("624249")

            self.assertEqual(len(cached_graphs), 2)
            self.assertEqual(cached_targets, [0, 1])

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

            loader = PubChemAssayLoader(root)
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

            loader = PubChemAssayLoader(root)
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

    def test_qm9_loader_reads_csv_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "qm9.csv"
            csv_path.write_text(
                "mol_id,smiles,mu,alpha,gap\n"
                "gdb_1,C,0.0,13.21,0.5048\n"
                "gdb_2,CCO,1.2,25.1,0.3000\n"
            )

            loader = QM9Loader(root, auto_download=False)
            paths = loader.resolve_paths("qm9")
            self.assertEqual(paths.dataset_name, "qm9")
            self.assertEqual(paths.csv_path, csv_path)

            frame = loader.load_frame("qm9", limit=1)
            self.assertEqual(frame["mol_id"].tolist(), ["gdb_1"])

            graphs, metadata = loader.load("qm9")
            self.assertEqual(len(graphs), 2)
            self.assertEqual(metadata["mol_id"].tolist(), ["gdb_1", "gdb_2"])
            self.assertEqual(graphs[0].graph["qm9_dataset"], "qm9")
            self.assertEqual(graphs[0].graph["mol_id"], "gdb_1")
            self.assertEqual(graphs[0].graph["source"], "qm9")
            self.assertTrue(
                (root / "graph_corpus_cache" / "qm9" / "graph_corpus" / "manifest.pkl").exists()
            )

    def test_qm9_loader_auto_downloads_default_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = (
                "mol_id,smiles,mu,alpha\n"
                "gdb_1,C,0.0,13.21\n"
            ).encode()

            class Response:
                def __init__(self, content: bytes) -> None:
                    self.content = content

                def raise_for_status(self) -> None:
                    return None

            with patch("abstractgraph_graphicalizer.chem.mol_loader.requests.get", return_value=Response(payload)):
                loader = QM9Loader(root)
                summaries = loader.list_datasets()

            self.assertEqual([summary.dataset_name for summary in summaries], ["qm9"])
            self.assertTrue((root / "qm9.csv").exists())

    def test_qm9_loader_skip_invalid_smiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "qm9.csv").write_text(
                "mol_id,smiles,mu\n"
                "gdb_1,C,0.0\n"
                "gdb_2,not-a-smiles,0.2\n"
                "gdb_3,C#N,0.3\n"
            )

            loader = QM9Loader(root, on_error="skip", auto_download=False)
            graphs, metadata = loader.load("qm9")

            self.assertEqual(len(graphs), 2)
            self.assertEqual(metadata["mol_id"].tolist(), ["gdb_1", "gdb_3"])

    def test_qm9_loader_reuses_graph_corpus_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "qm9.csv").write_text(
                "mol_id,smiles,mu\n"
                "gdb_1,C,0.0\n"
                "gdb_2,CCO,1.0\n"
            )

            loader = QM9Loader(root, auto_download=False)
            graphs, metadata = loader.load("qm9", min_node_count=1, max_node_count=1)

            self.assertEqual(len(graphs), 1)
            self.assertEqual(metadata["mol_id"].tolist(), ["gdb_1"])

            with patch.object(loader, "_graph_from_row", side_effect=AssertionError("cache not reused")):
                cached_graphs, cached_metadata = loader.load("qm9", min_node_count=3, max_node_count=3)

            self.assertEqual(len(cached_graphs), 1)
            self.assertEqual(cached_metadata["mol_id"].tolist(), ["gdb_2"])

    def test_qm9_loader_root_resolution_helpers(self) -> None:
        roots = qm9_search_roots()
        self.assertIn(local_qm9_root().resolve(), roots)
        self.assertIn(bundled_qm9_root().resolve(), roots)
        expected_default = (
            local_qm9_root().resolve()
            if local_qm9_root().resolve().exists()
            else bundled_qm9_root().resolve()
        )
        self.assertEqual(default_qm9_root(), expected_default)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"ABSTRACTGRAPH_QM9_ROOT": tmpdir}, clear=False):
                self.assertEqual(default_qm9_root(), Path(tmpdir).resolve())

    def test_qm9_loader_formats_dataset_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "qm9.csv").write_text("mol_id,smiles,mu\ngdb_1,C,0.0\ngdb_2,CCO,1.2\n")

            loader = QM9Loader(root, auto_download=False)
            table = loader.format_dataset_table()

            self.assertIn("dataset", table)
            self.assertIn("qm9", table)
            self.assertIn("molecules", table)

    def test_gdb_mode_resolution_defaults_and_validates(self) -> None:
        self.assertEqual(resolve_gdb_mode().mode, DEFAULT_GDB_MODE)
        self.assertEqual(resolve_gdb_mode("auto").mode, DEFAULT_GDB_MODE)
        self.assertEqual(resolve_gdb_mode("50M").mode, "50M")
        with self.assertRaises(ValueError):
            resolve_gdb_mode("not-a-mode")

    def test_iter_gdb_records_reads_gzip_and_tar_gz_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gz_path = root / "subset.smi.gz"
            gz_path.write_bytes(gzip.compress(b"CCO ethanol\nC=C ethene\n"))

            gz_records = list(iter_gdb_records(gz_path))
            self.assertEqual([record.smiles for record in gz_records], ["CCO", "C=C"])
            self.assertEqual(gz_records[0].annotation, "ethanol")

            tgz_path = root / "subset.tgz"
            with tarfile.open(tgz_path, "w:gz") as archive:
                payloads = {
                    "part_a.smi": b"C methane\n",
                    "part_b.smi": b"C#N cyanide\n",
                }
                for member_name, payload in payloads.items():
                    info = tarfile.TarInfo(member_name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

            tgz_records = list(iter_gdb_records(tgz_path))
            self.assertEqual([record.smiles for record in tgz_records], ["C", "C#N"])
            self.assertEqual([record.row_index for record in tgz_records], [0, 1])

    def test_gdb_loader_downloads_default_mode_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = gzip.compress(b"CCO mol1\n")

            class Response:
                def raise_for_status(self) -> None:
                    return None

                def iter_content(self, chunk_size: int = 0):
                    del chunk_size
                    yield payload

            with patch(
                "abstractgraph_graphicalizer.chem.gdb.requests.get",
                return_value=Response(),
            ):
                result = download_gdb_archive(root, None, verbose=False)

            self.assertTrue(result.archive_path.exists())
            metadata = read_gdb_metadata(result.metadata_path)
            self.assertEqual(metadata["mode"], DEFAULT_GDB_MODE)
            self.assertEqual(Path(metadata["archive_path"]), result.archive_path)

    def test_gdb_loader_builds_chunked_graph_corpus_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = resolve_gdb_mode("lead_like")
            (root / spec.archive_filename).write_bytes(
                gzip.compress(b"CCO first\nC=C second\nnot-a-smiles broken\nC#N third\n")
            )

            loader = GDBLoader(root, on_error="skip", auto_download=False, verbose=False)
            graphs, metadata = loader.load("lead_like", chunk_size=1, max_node_count=3)

            self.assertEqual(len(graphs), 3)
            self.assertEqual(metadata["gdb_mode"].tolist(), ["lead_like", "lead_like", "lead_like"])
            self.assertEqual(graphs[0].graph["source"], "gdb")
            manifest = build_gdb_graph_corpus(root, mode="lead_like", chunk_size=1, on_error="skip", verbose=False)
            self.assertIsInstance(manifest["bucket_files"][2], list)
            self.assertTrue(
                (root / "graph_corpus_cache" / "gdb_lead_like" / "graph_corpus" / "manifest.pkl").exists()
            )

            with patch.object(loader, "_graph_from_record", side_effect=AssertionError("cache not reused")):
                cached_graphs, cached_metadata = loader.load(
                    "lead_like",
                    min_node_count=2,
                    max_node_count=2,
                    chunk_size=1,
                )

            self.assertEqual(len(cached_graphs), 2)
            self.assertEqual(cached_metadata["smiles"].tolist(), ["C=C", "C#N"])

            loaded_graphs, loaded_metadata = load_gdb_graph_dataset(root, mode="lead_like", max_molecules=10)
            self.assertEqual(len(loaded_graphs), 3)
            self.assertEqual(loaded_metadata["gdb_family"].iloc[0], spec.family)

    def test_gdb_loader_root_resolution_helpers(self) -> None:
        roots = gdb_search_roots()
        self.assertIn(local_gdb_root().resolve(), roots)
        self.assertIn(bundled_gdb_root().resolve(), roots)
        expected_default = (
            local_gdb_root().resolve()
            if local_gdb_root().resolve().exists()
            else bundled_gdb_root().resolve()
        )
        self.assertEqual(default_gdb_root(), expected_default)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"ABSTRACTGRAPH_GDB_ROOT": tmpdir}, clear=False):
                self.assertEqual(default_gdb_root(), Path(tmpdir).resolve())

    def test_gdb_loader_formats_mode_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GDBLoader(tmpdir, auto_download=False, verbose=False)
            table = loader.format_mode_table()

            self.assertIn("mode", table)
            self.assertIn("lead_like", table)
            self.assertIn("50M", table)
            self.assertIn("downloaded", table)

    def test_bundled_loader_roots_share_top_level_cache_parent(self) -> None:
        shared_cache_parent = bundled_zinc_root().parent / "graph_corpus_cache"

        self.assertEqual(ZINCLoader(bundled_zinc_root())._cache_root("zinc_250k"), shared_cache_parent / "zinc_250k")
        self.assertEqual(QM9Loader(bundled_qm9_root(), auto_download=False)._cache_root("qm9"), shared_cache_parent / "qm9")
        self.assertEqual(PubChemAssayLoader(bundled_pubchem_root())._cache_root("463230"), shared_cache_parent / "AID463230")
        self.assertEqual(gdb_cache_root(bundled_gdb_root(), "lead_like"), shared_cache_parent / "gdb_lead_like")

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
            bucket_path = root / "graph_corpus_cache" / "zinc_small" / manifest["bucket_files"][2]
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
            self.assertTrue(
                (root / "graph_corpus_cache" / "zinc_small" / "graph_corpus" / "manifest.pkl").exists()
            )

    def test_extract_zinc_targets_uses_default_columns(self) -> None:
        metadata = pd.DataFrame([{"logP": 1.0, "qed": 0.5, "SAS": 2.0, "extra": 7}])
        targets = extract_zinc_targets(metadata)
        self.assertEqual(targets.columns.tolist(), list(DEFAULT_ZINC_TARGET_COLUMNS))

    def test_build_and_load_qm9_graph_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "qm9.csv"
            pd.DataFrame(
                [
                    {"mol_id": "gdb_1", "smiles": "C=C", "mu": 0.0, "alpha": 13.21},
                    {"mol_id": "gdb_2", "smiles": "CCO", "mu": 1.2, "alpha": 25.1},
                ]
            ).to_csv(csv_path, index=False)

            manifest = build_qm9_graph_corpus(root, csv_path)
            bucket_path = root / "graph_corpus_cache" / "qm9" / manifest["bucket_files"][2]
            with bucket_path.open("rb") as handle:
                items = pickle.load(handle)
            legacy_graph = items[0][0]
            for _, _, data in legacy_graph.edges(data=True):
                data["label"] = "1"
                data.pop("bond_order", None)
                data.pop("bond_type", None)
            with bucket_path.open("wb") as handle:
                pickle.dump(items, handle)

            graphs, metadata = load_qm9_graph_dataset(root, max_molecules=10, max_node_count=3)

            self.assertEqual(len(graphs), 2)
            self.assertEqual(sorted(metadata["mol_id"].tolist()), ["gdb_1", "gdb_2"])
            self.assertEqual(next(iter(graphs[0].edges(data=True)))[2]["label"], "single")
            self.assertTrue(
                (root / "graph_corpus_cache" / "qm9" / "graph_corpus" / "manifest.pkl").exists()
            )

    def test_extract_qm9_targets_uses_default_columns(self) -> None:
        metadata = pd.DataFrame([{column: float(index) for index, column in enumerate(DEFAULT_QM9_TARGET_COLUMNS, start=1)}])
        metadata["extra"] = 7
        targets = extract_qm9_targets(metadata)
        self.assertEqual(targets.columns.tolist(), list(DEFAULT_QM9_TARGET_COLUMNS))


if __name__ == "__main__":
    unittest.main()
