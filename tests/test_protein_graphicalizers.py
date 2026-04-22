from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from abstractgraph_graphicalizer.protein import (
    AMINO_ACID_ALPHABETS,
    ProteinChainRecord,
    ProteinContactNetworkGraphicalizer,
    ProteinContactNetworkLoader,
    ProteinLabelGraphicalizer,
    ResidueCA,
    download_mmcif,
    extract_chain_record,
    label_protein_contact_graph,
    protein_chain_record_to_pcn,
)


def _record(coords: tuple[tuple[float, float, float], ...]) -> ProteinChainRecord:
    names = ("ALA", "GLY", "HIS", "SER", "LYS")
    residues = tuple(
        ResidueCA(names[idx], "A", "X", str(idx + 1), str(101 + idx), "", coord)
        for idx, coord in enumerate(coords)
    )
    return ProteinChainRecord(
        sample_id="demo.A",
        pdb_id="demo",
        label_asym_id="A",
        auth_asym_id="X",
        residues=residues,
    )


def _mmcif_text() -> str:
    return """data_demo
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA ALA A X 1 101 ? 0.0 0.0 0.0 1
ATOM 2 C CA ALA A X 1 101 ? 2.0 0.0 0.0 1
ATOM 3 C CA GLY A X 2 102 ? 5.0 0.0 0.0 1
ATOM 4 C CA HIS A X 3 103 ? 9.0 0.0 0.0 1
ATOM 5 C CA SER B Y 1 201 ? 0.0 0.0 0.0 1
ATOM 6 C CA LYS B Y 2 202 ? 6.0 0.0 0.0 1
ATOM 7 C CA VAL B Y 3 203 ? 12.0 0.0 0.0 1
ATOM 8 C CA GLY C Z 1 301 ? 0.0 0.0 0.0 2
#
"""


def _protein_path(labels: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["min_distance"] = 4.0
    graph.graph["max_distance"] = 8.0
    for idx, label in enumerate(labels):
        graph.add_node(idx, label=label)
    for idx in range(len(labels) - 1):
        graph.add_edge(idx, idx + 1, distance=4.5 + idx)
    return graph


class ProteinGraphicalizerTest(unittest.TestCase):
    def test_protein_chain_record_to_pcn_uses_contact_distance_window(self) -> None:
        graph = protein_chain_record_to_pcn(
            _record(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (5.0, 0.0, 0.0), (11.0, 0.0, 0.0))),
            min_distance=4.0,
            max_distance=8.0,
        )

        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(set(graph.edges()), {(0, 2), (1, 3), (2, 3)})
        self.assertEqual(graph.nodes[0]["label"], "ALA")
        self.assertEqual(graph.nodes[0]["auth_chain_id"], "X")
        self.assertEqual(graph.edges[0, 2]["distance"], 5.0)
        self.assertNotIn("label", graph.edges[0, 2])
        self.assertEqual(graph.graph["source"], "protein_contact_network")
        self.assertEqual(graph.graph["edge_distance_key"], "distance")

    def test_protein_chain_record_to_pcn_can_scale_edge_widths_by_distance(self) -> None:
        graph = protein_chain_record_to_pcn(
            _record(((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (8.0, 0.0, 0.0))),
            min_distance=4.0,
            max_distance=8.0,
            scale_edge_width_by_distance=True,
            min_edge_width=0.5,
            max_edge_width=2.5,
        )

        self.assertEqual(graph.edges[0, 1]["edge_width"], 2.5)
        self.assertEqual(graph.edges[0, 2]["edge_width"], 0.5)

    def test_protein_chain_record_to_pcn_validates_parameters_and_empty_contacts(self) -> None:
        with self.assertRaises(ValueError):
            protein_chain_record_to_pcn(_record(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))), min_distance=8.0, max_distance=4.0)
        with self.assertRaises(ValueError):
            protein_chain_record_to_pcn(
                _record(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
                min_distance=4.0,
                max_distance=8.0,
            )

    def test_extract_chain_record_handles_plain_and_requested_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.cif"
            path.write_text(_mmcif_text())

            record = extract_chain_record(path, pdb_id="DEMO", sample_id="sample.A", label_asym_id="A")

        self.assertEqual(record.sample_id, "sample.A")
        self.assertEqual(record.pdb_id, "demo")
        self.assertEqual(record.label_asym_id, "A")
        self.assertEqual(record.auth_asym_id, "X")
        self.assertEqual(len(record.residues), 3)
        self.assertEqual(record.residues[0].coord, (1.0, 0.0, 0.0))
        self.assertEqual(record.residues[0].auth_seq_id, "101")

    def test_extract_chain_record_handles_gzip_longest_chain_and_auth_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.cif.gz"
            path.write_bytes(gzip.compress(_mmcif_text().encode()))

            longest = extract_chain_record(path, pdb_id="demo")
            auth = extract_chain_record(path, pdb_id="demo", auth_asym_id="Y")

        self.assertEqual(longest.label_asym_id, "A")
        self.assertEqual(len(longest.residues), 3)
        self.assertEqual(auth.label_asym_id, "B")
        self.assertEqual(auth.auth_asym_id, "Y")
        self.assertEqual(len(auth.residues), 3)

    def test_extract_chain_record_rejects_too_short_chains(self) -> None:
        text = _mmcif_text().replace(
            "ATOM 6 C CA LYS B Y 2 202 ? 6.0 0.0 0.0 1\nATOM 7 C CA VAL B Y 3 203 ? 12.0 0.0 0.0 1\n",
            "",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.cif"
            path.write_text(text)
            with self.assertRaises(ValueError):
                extract_chain_record(path, pdb_id="demo", auth_asym_id="Y")

    def test_download_mmcif_writes_response_and_reuses_cache(self) -> None:
        class Response:
            content = b"payload"

            def raise_for_status(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("abstractgraph_graphicalizer.protein.pcn.requests.get", return_value=Response()) as get:
                first = download_mmcif("1UBQ", tmpdir)
                second = download_mmcif("1ubq", tmpdir)
            payload = first.read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(payload, b"payload")

    def test_graphicalizer_and_loader_accept_records_tokens_and_reuse_graph_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mmcif_dir = root / "mmcif"
            graph_dir = root / "graphs"
            mmcif_dir.mkdir()
            (mmcif_dir / "demo.cif.gz").write_bytes(gzip.compress(_mmcif_text().encode()))

            graphicalizer = ProteinContactNetworkGraphicalizer(graph_dir=graph_dir)
            record_graph = graphicalizer.transform([_record(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)))])[0]
            self.assertEqual(record_graph.graph["source"], "protein_contact_network")

            loader = ProteinContactNetworkLoader(mmcif_dir=mmcif_dir, graph_dir=graph_dir, return_graphs=True)
            graph = loader.load(["demo.A"])[0]
            with patch(
                "abstractgraph_graphicalizer.protein.pcn.download_mmcif",
                side_effect=AssertionError("graph cache was not reused"),
            ):
                cached_graph = loader.load(["demo.A"])[0]

        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(cached_graph.graph["sample_id"], "demo.A")

    def test_all_expected_amino_acid_alphabets_are_available(self) -> None:
        self.assertEqual(
            set(AMINO_ACID_ALPHABETS),
            {"aa20", "hp2", "charge4", "physchem5", "chem7", "dayhoff6"},
        )

    def test_protein_label_graphicalizer_relabels_without_mutating_input_graph(self) -> None:
        graph = _protein_path(["ALA", "ASP", "LYS", "GLY", "PHE"])
        labeler = ProteinLabelGraphicalizer(alphabet="physchem5")

        relabeled = labeler.label_graph(graph)

        self.assertEqual(
            [relabeled.nodes[idx]["label"] for idx in range(5)],
            ["hydrophobic", "negative", "positive", "special", "hydrophobic"],
        )
        self.assertEqual([graph.nodes[idx]["label"] for idx in range(5)], ["ALA", "ASP", "LYS", "GLY", "PHE"])
        self.assertNotIn("label", graph.edges[0, 1])
        self.assertEqual(relabeled.edges[0, 1]["label"], "contact")

    def test_protein_label_graphicalizer_supports_named_alphabets(self) -> None:
        expected_by_alphabet = {
            "aa20": ["ALA", "ASP", "LYS", "GLY", "PHE"],
            "hp2": ["hydrophobic", "polar", "polar", "polar", "hydrophobic"],
            "charge4": ["nonpolar", "negative", "positive", "nonpolar", "nonpolar"],
            "physchem5": ["hydrophobic", "negative", "positive", "special", "hydrophobic"],
            "chem7": ["aliphatic_special", "acidic", "basic", "aliphatic_special", "aromatic"],
            "dayhoff6": ["small_polar", "acid_amide", "basic", "small_polar", "aromatic"],
        }
        graph = _protein_path(["ALA", "ASP", "LYS", "GLY", "PHE"])

        for alphabet, expected in expected_by_alphabet.items():
            with self.subTest(alphabet=alphabet):
                relabeled = ProteinLabelGraphicalizer(alphabet=alphabet).label_graph(graph)
                self.assertEqual([relabeled.nodes[idx]["label"] for idx in range(5)], expected)

    def test_protein_label_graphicalizer_accepts_one_letter_residue_codes(self) -> None:
        graph = _protein_path(["A", "D", "K", "G", "F"])

        relabeled = ProteinLabelGraphicalizer(alphabet="chem7").label_graph(graph)

        self.assertEqual(
            [relabeled.nodes[idx]["label"] for idx in range(5)],
            ["aliphatic_special", "acidic", "basic", "aliphatic_special", "aromatic"],
        )

    def test_protein_label_graphicalizer_discretizes_edge_labels_from_distances(self) -> None:
        graph = _protein_path(["ALA", "ASP", "LYS"])
        labeler = ProteinLabelGraphicalizer(alphabet="physchem5", edge_distance_thresholds=[5.0])

        relabeled = labeler.transform([graph])[0]

        self.assertEqual(relabeled.edges[0, 1]["label"], "contact_close")
        self.assertEqual(relabeled.edges[1, 2]["label"], "contact_far")
        self.assertNotIn("label", graph.edges[0, 1])

    def test_label_protein_contact_graph_convenience_function(self) -> None:
        graph = _protein_path(["ALA", "ASP"])

        relabeled = label_protein_contact_graph(graph, alphabet="hp2")

        self.assertEqual(relabeled.nodes[0]["label"], "hydrophobic")
        self.assertEqual(relabeled.nodes[1]["label"], "polar")

    def test_protein_label_graphicalizer_rejects_thresholds_outside_contact_window(self) -> None:
        graph = _protein_path(["ALA", "ASP"])
        labeler = ProteinLabelGraphicalizer(alphabet="physchem5", edge_distance_thresholds=[8.0])

        with self.assertRaisesRegex(ValueError, "inside the contact window"):
            labeler.label_graph(graph)

    def test_protein_label_graphicalizer_requires_distance_for_edge_discretization(self) -> None:
        graph = nx.Graph()
        graph.graph["min_distance"] = 4.0
        graph.graph["max_distance"] = 8.0
        graph.add_node(0, label="ALA")
        graph.add_node(1, label="ASP")
        graph.add_edge(0, 1)
        labeler = ProteinLabelGraphicalizer(alphabet="physchem5", edge_distance_thresholds=[5.0])

        with self.assertRaisesRegex(ValueError, "missing distance"):
            labeler.label_graph(graph)

    def test_protein_label_graphicalizer_unknown_policy_error_rejects_unknown_labels(self) -> None:
        graph = _protein_path(["ALA", "UNK"])
        labeler = ProteinLabelGraphicalizer(alphabet="hp2", unknown_policy="error")

        with self.assertRaisesRegex(ValueError, "Unknown amino-acid label"):
            labeler.label_graph(graph)


if __name__ == "__main__":
    unittest.main()
