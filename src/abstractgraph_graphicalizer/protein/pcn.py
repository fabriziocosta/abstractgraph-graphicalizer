"""Protein contact network graphicalizers and loaders."""

from __future__ import annotations

from collections.abc import Iterable as CollectionsIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import gzip
import pickle
import shlex
import time

import networkx as nx
import numpy as np
import pandas as pd
import requests
from requests import RequestException
from sklearn.base import BaseEstimator, TransformerMixin

from abstractgraph_graphicalizer.core import GraphicalizerMixin


RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

AA1_TO_AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
AA3_TO_AA1 = {value: key for key, value in AA1_TO_AA3.items()}


def _map_groups(groups: dict[str, CollectionsIterable[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group_name, residues in groups.items():
        for residue in residues:
            code = residue.upper()
            if len(code) == 1:
                mapping[code] = group_name
                mapping[AA1_TO_AA3[code]] = group_name
            elif len(code) == 3:
                mapping[code] = group_name
                if code in AA3_TO_AA1:
                    mapping[AA3_TO_AA1[code]] = group_name
            else:
                raise ValueError(f"Unexpected residue code {residue!r}")
    return mapping


AMINO_ACID_ALPHABETS: dict[str, dict[str, str]] = {
    "aa20": {},
    "hp2": _map_groups(
        {
            "hydrophobic": "AVLIMFWYC",
            "polar": "STNQDEKRHGP",
        }
    ),
    "charge4": _map_groups(
        {
            "positive": "KRH",
            "negative": "DE",
            "polar_neutral": "STNQCY",
            "nonpolar": "AVLIMFWGP",
        }
    ),
    "physchem5": _map_groups(
        {
            "hydrophobic": "AVLIMFWY",
            "polar_uncharged": "STNQC",
            "positive": "KRH",
            "negative": "DE",
            "special": "GP",
        }
    ),
    "chem7": _map_groups(
        {
            "acidic": "DE",
            "basic": "KRH",
            "hydroxyl": "STY",
            "amide": "NQ",
            "sulfur": "CM",
            "aromatic": "FW",
            "aliphatic_special": "AVLIGP",
        }
    ),
    "dayhoff6": _map_groups(
        {
            "cysteine": "C",
            "small_polar": "STPAG",
            "acid_amide": "NDEQ",
            "basic": "HRK",
            "aliphatic": "MILV",
            "aromatic": "FYW",
        }
    ),
}


@dataclass(frozen=True)
class ResidueCA:
    """A residue represented by averaged C-alpha coordinates."""

    residue_name: str
    label_asym_id: str
    auth_asym_id: str
    label_seq_id: str
    auth_seq_id: str
    insertion_code: str
    coord: tuple[float, float, float]


@dataclass(frozen=True)
class ProteinChainRecord:
    """Parsed C-alpha record for one protein chain/sample."""

    sample_id: str
    pdb_id: str
    label_asym_id: str
    auth_asym_id: str
    residues: tuple[ResidueCA, ...]


@dataclass(frozen=True)
class GraphBuildResult:
    """Result for one requested protein sample."""

    sample_id: str
    pdb_id: str
    graph_path: str
    status: str
    message: str = ""


def read_text_maybe_gzip(path: str | Path) -> str:
    """Read plain or gzip-compressed text."""
    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return input_path.read_text(encoding="utf-8")


def _tokenize_cif_line(line: str) -> list[str]:
    return shlex.split(line, posix=False)


def _iter_atom_site_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "loop_":
            idx += 1
            continue

        idx += 1
        tags: list[str] = []
        while idx < len(lines) and lines[idx].strip().startswith("_"):
            tag = lines[idx].strip().split()[0]
            tags.append(tag)
            idx += 1

        if not tags or not all(tag.startswith("_atom_site.") for tag in tags):
            while idx < len(lines) and lines[idx].strip() not in {"#", "loop_"}:
                idx += 1
            continue

        values: list[str] = []
        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue
            if stripped == "#":
                idx += 1
                break
            if stripped == "loop_" or stripped.startswith("_"):
                break
            values.extend(_tokenize_cif_line(stripped))
            idx += 1

        width = len(tags)
        for start in range(0, len(values) - width + 1, width):
            row_values = values[start : start + width]
            rows.append({tag.removeprefix("_atom_site."): value for tag, value in zip(tags, row_values)})
    return rows


def _clean(value: str | None) -> str:
    if value is None or value in {".", "?"}:
        return ""
    return str(value).strip("'\"")


def _model_number(row: dict[str, str]) -> str:
    return _clean(row.get("pdbx_PDB_model_num")) or "1"


def extract_chain_record(
    mmcif_path: str | Path,
    *,
    pdb_id: str,
    sample_id: str = "",
    label_asym_id: str = "",
    auth_asym_id: str = "",
) -> ProteinChainRecord:
    """Extract a chain's C-alpha residues from an mmCIF file.

    If no chain is supplied, the chain with the most C-alpha residues is selected.
    Alternate C-alpha locations are averaged per residue.
    """
    rows = _iter_atom_site_rows(read_text_maybe_gzip(mmcif_path))
    ca_rows = [
        row
        for row in rows
        if _clean(row.get("group_PDB")) == "ATOM"
        and _clean(row.get("label_atom_id")).upper() == "CA"
        and _model_number(row) == "1"
    ]
    if not ca_rows:
        raise ValueError(f"No C-alpha ATOM rows found in {mmcif_path}")

    requested_label = _clean(label_asym_id)
    requested_auth = _clean(auth_asym_id)
    if requested_label or requested_auth:
        ca_rows = [
            row
            for row in ca_rows
            if (requested_label and _clean(row.get("label_asym_id")) == requested_label)
            or (requested_auth and _clean(row.get("auth_asym_id")) == requested_auth)
        ]
        if not ca_rows:
            requested = requested_label or requested_auth
            raise ValueError(f"No C-alpha rows found for requested chain {requested!r}")
    else:
        counts: dict[str, int] = {}
        for row in ca_rows:
            chain = _clean(row.get("label_asym_id")) or _clean(row.get("auth_asym_id"))
            counts[chain] = counts.get(chain, 0) + 1
        selected_chain = max(counts, key=counts.get)
        ca_rows = [
            row
            for row in ca_rows
            if (_clean(row.get("label_asym_id")) or _clean(row.get("auth_asym_id"))) == selected_chain
        ]

    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[float, float, float]]] = {}
    for row in ca_rows:
        key = (
            _clean(row.get("label_asym_id")),
            _clean(row.get("auth_asym_id")),
            _clean(row.get("label_seq_id")),
            _clean(row.get("auth_seq_id")),
            _clean(row.get("pdbx_PDB_ins_code")),
            _clean(row.get("label_comp_id")),
        )
        coord = (
            float(_clean(row.get("Cartn_x"))),
            float(_clean(row.get("Cartn_y"))),
            float(_clean(row.get("Cartn_z"))),
        )
        grouped.setdefault(key, []).append(coord)

    residues: list[ResidueCA] = []
    for key, coords in grouped.items():
        label_chain, auth_chain, label_seq, auth_seq, insertion_code, residue_name = key
        avg = np.asarray(coords, dtype=float).mean(axis=0)
        residues.append(
            ResidueCA(
                residue_name=residue_name,
                label_asym_id=label_chain,
                auth_asym_id=auth_chain,
                label_seq_id=label_seq,
                auth_seq_id=auth_seq,
                insertion_code=insertion_code,
                coord=(float(avg[0]), float(avg[1]), float(avg[2])),
            )
        )

    residues.sort(key=lambda item: (_sequence_sort_key(item.label_seq_id), item.auth_seq_id, item.insertion_code))
    if len(residues) < 2:
        raise ValueError("Protein chain has fewer than two C-alpha residues")

    selected_label = residues[0].label_asym_id
    selected_auth = residues[0].auth_asym_id
    resolved_sample_id = sample_id or f"{pdb_id.lower()}.{selected_label or selected_auth}"
    return ProteinChainRecord(
        sample_id=resolved_sample_id,
        pdb_id=pdb_id.lower(),
        label_asym_id=selected_label,
        auth_asym_id=selected_auth,
        residues=tuple(residues),
    )


def protein_chain_record_to_pcn(
    record: ProteinChainRecord,
    *,
    min_distance: float = 4.0,
    max_distance: float = 8.0,
    scale_edge_width_by_distance: bool = False,
    min_edge_width: float = 0.3,
    max_edge_width: float = 2.2,
) -> nx.Graph:
    """Convert a parsed protein chain record into a protein contact network."""
    if min_distance < 0 or max_distance <= min_distance:
        raise ValueError("Expected 0 <= min_distance < max_distance")
    if min_edge_width <= 0 or max_edge_width < min_edge_width:
        raise ValueError("Expected 0 < min_edge_width <= max_edge_width")

    graph = nx.Graph()
    graph.graph["source"] = "protein_contact_network"
    graph.graph["sample_id"] = record.sample_id
    graph.graph["pdb_id"] = record.pdb_id
    graph.graph["label_asym_id"] = record.label_asym_id
    graph.graph["auth_asym_id"] = record.auth_asym_id
    graph.graph["min_distance"] = float(min_distance)
    graph.graph["max_distance"] = float(max_distance)
    graph.graph["edge_distance_key"] = "distance"
    graph.graph["scale_edge_width_by_distance"] = bool(scale_edge_width_by_distance)

    coords = np.asarray([residue.coord for residue in record.residues], dtype=float)
    for node_idx, residue in enumerate(record.residues):
        graph.add_node(
            node_idx,
            label=residue.residue_name,
            residue_name=residue.residue_name,
            residue_index=node_idx,
            label_seq_id=residue.label_seq_id,
            auth_seq_id=residue.auth_seq_id,
            insertion_code=residue.insertion_code,
            chain_id=residue.label_asym_id,
            auth_chain_id=residue.auth_asym_id,
        )

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    row_idx, col_idx = np.where((distances >= min_distance) & (distances <= max_distance))
    for source, target in zip(row_idx.tolist(), col_idx.tolist()):
        if source >= target:
            continue
        distance = float(distances[source, target])
        edge_attrs = {"distance": distance}
        if scale_edge_width_by_distance:
            edge_attrs["edge_width"] = _edge_width_from_distance(
                distance,
                min_distance=min_distance,
                max_distance=max_distance,
                min_edge_width=min_edge_width,
                max_edge_width=max_edge_width,
            )
        graph.add_edge(source, target, **edge_attrs)

    if graph.number_of_edges() == 0:
        raise ValueError("Protein contact network has no edges")
    return graph


def label_protein_contact_graph(
    graph: nx.Graph,
    *,
    alphabet: str = "aa20",
    node_label_key: str = "label",
    edge_label_key: str = "label",
    edge_distance_key: str = "distance",
    edge_distance_thresholds: tuple[float, ...] | list[float] | None = None,
    unknown_policy: str = "keep",
    unknown_label: str = "unknown",
) -> nx.Graph:
    """Return a copy of a PCN graph with representation-specific labels."""
    return ProteinLabelGraphicalizer(
        alphabet=alphabet,
        node_label_key=node_label_key,
        edge_label_key=edge_label_key,
        edge_distance_key=edge_distance_key,
        edge_distance_thresholds=edge_distance_thresholds,
        unknown_policy=unknown_policy,
        unknown_label=unknown_label,
    ).label_graph(graph)


class ProteinLabelGraphicalizer(GraphicalizerMixin, BaseEstimator, TransformerMixin):
    """Relabel protein contact network nodes and edges on the fly.

    Cached PCN graphs should keep raw residue node labels and edge distances.
    This graphicalizer copies those graphs and derives reduced amino-acid node
    labels plus contact edge labels for downstream graph vectorizers.
    """

    def __init__(
        self,
        *,
        alphabet: str = "aa20",
        node_label_key: str = "label",
        edge_label_key: str = "label",
        edge_distance_key: str = "distance",
        edge_distance_thresholds: tuple[float, ...] | list[float] | None = None,
        unknown_policy: str = "keep",
        unknown_label: str = "unknown",
    ) -> None:
        self.alphabet = alphabet
        self.node_label_key = node_label_key
        self.edge_label_key = edge_label_key
        self.edge_distance_key = edge_distance_key
        self.edge_distance_thresholds = edge_distance_thresholds
        self.unknown_policy = unknown_policy
        self.unknown_label = unknown_label

    def transform(self, X, y=None) -> list[nx.Graph]:
        graphs = [X] if isinstance(X, nx.Graph) else list(X)
        return [self.label_graph(graph) for graph in graphs]

    def label_graph(self, graph: nx.Graph) -> nx.Graph:
        """Return a copy of `graph` with representation-specific labels."""
        mapping = self._alphabet_mapping()
        edge_thresholds = self._edge_distance_thresholds(graph)

        relabeled = graph.copy()
        for node, attrs in relabeled.nodes(data=True):
            old_label = attrs.get(self.node_label_key, "")
            attrs[self.node_label_key] = self._map_node_label(old_label, mapping, node=node)
        for source, target, attrs in relabeled.edges(data=True):
            attrs[self.edge_label_key] = self._map_edge_label(
                attrs,
                thresholds=edge_thresholds,
                edge=(source, target),
            )
        relabeled.graph["node_label_alphabet"] = self.alphabet
        relabeled.graph["node_label_source_key"] = self.node_label_key
        relabeled.graph["edge_label_key"] = self.edge_label_key
        relabeled.graph["edge_distance_key"] = self.edge_distance_key
        relabeled.graph["edge_distance_thresholds"] = list(edge_thresholds)
        return relabeled

    def _alphabet_mapping(self) -> dict[str, str]:
        alphabet_key = str(self.alphabet).strip().lower()
        if alphabet_key not in AMINO_ACID_ALPHABETS:
            valid = ", ".join(sorted(AMINO_ACID_ALPHABETS))
            raise ValueError(f"Unknown amino-acid alphabet {self.alphabet!r}; expected one of: {valid}")
        return AMINO_ACID_ALPHABETS[alphabet_key]

    def _map_node_label(self, label: Any, mapping: dict[str, str], *, node: Any) -> str:
        normalized = str(label).strip().upper()
        if normalized in mapping:
            return mapping[normalized]
        if not mapping and str(self.alphabet).strip().lower() == "aa20":
            return str(label)
        if self.unknown_policy == "keep":
            return str(label)
        if self.unknown_policy == "unknown":
            return str(self.unknown_label)
        if self.unknown_policy == "error":
            raise ValueError(f"Unknown amino-acid label {label!r} on node {node!r}")
        raise ValueError("unknown_policy must be 'keep', 'unknown', or 'error'")

    def _edge_distance_thresholds(self, graph: nx.Graph) -> tuple[float, ...]:
        thresholds = self.edge_distance_thresholds
        if thresholds is None:
            thresholds = ()
        normalized = tuple(sorted(float(threshold) for threshold in thresholds))
        min_distance = graph.graph.get("min_distance")
        max_distance = graph.graph.get("max_distance")
        if min_distance is None or max_distance is None:
            return normalized
        min_distance = float(min_distance)
        max_distance = float(max_distance)
        for threshold in normalized:
            if threshold <= min_distance or threshold >= max_distance:
                raise ValueError("Edge distance thresholds must be inside the contact window")
        return normalized

    def _map_edge_label(
        self,
        attrs: dict[str, Any],
        *,
        thresholds: tuple[float, ...],
        edge: tuple[Any, Any],
    ) -> str:
        if not thresholds:
            return "contact"
        if self.edge_distance_key not in attrs:
            raise ValueError(
                f"Cannot discretize edge {edge!r}: missing distance attribute {self.edge_distance_key!r}"
            )
        distance = float(attrs[self.edge_distance_key])
        if len(thresholds) == 1:
            return "contact_close" if distance < thresholds[0] else "contact_far"

        bin_names = ("very_close", "close", "mid", "far", "very_far")
        bin_index = sum(distance >= threshold for threshold in thresholds)
        if len(thresholds) + 1 <= len(bin_names):
            return f"contact_{bin_names[bin_index]}"
        return f"contact_bin_{bin_index}"


def download_mmcif(
    pdb_id: str,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    sleep_seconds: float = 0.0,
) -> Path:
    """Download a compressed mmCIF file from RCSB if it is not already cached."""
    normalized = str(pdb_id).strip().lower()
    if not normalized:
        raise ValueError("pdb_id must be non-empty")
    out_path = Path(out_dir) / f"{normalized}.cif.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return out_path

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(RCSB_MMCIF_URL.format(pdb_id=normalized), timeout=timeout_seconds)
            response.raise_for_status()
            break
        except RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))

    if response is None:
        raise RuntimeError(f"Failed to download mmCIF for {normalized}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(out_path)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return out_path


class ProteinContactNetworkGraphicalizer(GraphicalizerMixin):
    """Fetch mmCIF files and convert requested proteins into PCN graphs."""

    def __init__(
        self,
        *,
        raw_mmcif_dir: str | Path | None = None,
        mmcif_dir: str | Path | None = None,
        graph_dir: str | Path | None = None,
        min_distance: float = 4.0,
        max_distance: float = 8.0,
        scale_edge_width_by_distance: bool = False,
        min_edge_width: float = 0.3,
        max_edge_width: float = 2.2,
        overwrite: bool = False,
        return_graphs: bool = True,
        download_timeout_seconds: int = 60,
        download_max_retries: int = 3,
        download_retry_backoff_seconds: float = 2.0,
        sleep_seconds: float = 0.0,
    ) -> None:
        resolved_mmcif_dir = mmcif_dir if mmcif_dir is not None else raw_mmcif_dir
        self.mmcif_dir = Path(resolved_mmcif_dir or "data/raw/mmcif")
        self.raw_mmcif_dir = self.mmcif_dir
        self.graph_dir = Path(graph_dir) if graph_dir is not None else None
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.scale_edge_width_by_distance = scale_edge_width_by_distance
        self.min_edge_width = min_edge_width
        self.max_edge_width = max_edge_width
        self.overwrite = overwrite
        self.return_graphs = return_graphs
        self.download_timeout_seconds = download_timeout_seconds
        self.download_max_retries = download_max_retries
        self.download_retry_backoff_seconds = download_retry_backoff_seconds
        self.sleep_seconds = sleep_seconds

    def graph_path_for(self, sample_id: str) -> Path:
        if self.graph_dir is None:
            raise ValueError("graph_dir must be set to resolve graph cache paths")
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
        return self.graph_dir / f"{safe_id}.pkl"

    def transform_record(self, record: ProteinChainRecord) -> nx.Graph:
        return protein_chain_record_to_pcn(
            record,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            scale_edge_width_by_distance=self.scale_edge_width_by_distance,
            min_edge_width=self.min_edge_width,
            max_edge_width=self.max_edge_width,
        )

    def transform_one(self, row: dict[str, str] | str | Path | ProteinChainRecord) -> tuple[nx.Graph | None, GraphBuildResult]:
        if isinstance(row, ProteinChainRecord):
            graph = self.transform_record(row)
            return graph, GraphBuildResult(row.sample_id, row.pdb_id, "", "created")

        normalized_row = _normalize_row(row)
        sample_id = normalized_row["sample_id"]
        pdb_id = normalized_row["pdb_id"]
        label_asym_id = normalized_row["label_asym_id"]
        auth_asym_id = normalized_row["auth_asym_id"]
        if not pdb_id:
            raise ValueError("Manifest row is missing pdb_id")
        if not sample_id:
            sample_id = f"{pdb_id}.{label_asym_id or auth_asym_id}" if (label_asym_id or auth_asym_id) else pdb_id

        graph_path: Path | None = None
        if self.graph_dir is not None:
            graph_path = self.graph_path_for(sample_id)
            if graph_path.exists() and graph_path.stat().st_size > 0 and not self.overwrite:
                graph = read_graph_pickle(graph_path) if self.return_graphs else None
                return graph, GraphBuildResult(sample_id, pdb_id, str(graph_path), "skipped_existing")

        mmcif_path = download_mmcif(
            pdb_id,
            self.mmcif_dir,
            overwrite=False,
            timeout_seconds=self.download_timeout_seconds,
            max_retries=self.download_max_retries,
            retry_backoff_seconds=self.download_retry_backoff_seconds,
            sleep_seconds=self.sleep_seconds,
        )
        record = extract_chain_record(
            mmcif_path,
            pdb_id=pdb_id,
            sample_id=sample_id,
            label_asym_id=label_asym_id,
            auth_asym_id=auth_asym_id,
        )
        graph = self.transform_record(record)
        if graph_path is not None:
            write_graph_pickle(graph, graph_path)
        return (graph if self.return_graphs else None), GraphBuildResult(
            sample_id=sample_id,
            pdb_id=pdb_id,
            graph_path=str(graph_path or ""),
            status="created",
        )

    def transform(self, X, y=None) -> list[nx.Graph]:
        rows = _rows_from_input(X)
        graphs: list[nx.Graph] = []
        for row in rows:
            graph, _ = self.transform_one(row)
            if graph is not None:
                graphs.append(graph)
        return graphs


class ProteinContactNetworkLoader:
    """Load protein contact network graphs from PDB tokens or manifest rows."""

    def __init__(self, **kwargs) -> None:
        self.graphicalizer = ProteinContactNetworkGraphicalizer(**kwargs)

    def graph_path_for(self, sample_id: str) -> Path:
        return self.graphicalizer.graph_path_for(sample_id)

    def transform_one(self, row: dict[str, str] | str | Path | ProteinChainRecord) -> tuple[nx.Graph | None, GraphBuildResult]:
        return self.graphicalizer.transform_one(row)

    def iter_graphs(self, X) -> Iterator[nx.Graph]:
        for row in _rows_from_input(X):
            graph, _ = self.transform_one(row)
            if graph is not None:
                yield graph

    def load(self, X) -> list[nx.Graph]:
        return list(self.iter_graphs(X))


def write_graph_pickle(graph: nx.Graph, path: str | Path) -> Path:
    """Atomically write a NetworkX graph with pickle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)
    return output_path


def read_graph_pickle(path: str | Path) -> nx.Graph:
    """Read a NetworkX graph written by `write_graph_pickle`."""
    with Path(path).open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path} did not contain a NetworkX graph")
    return graph


def _rows_from_input(X) -> list[dict[str, str] | ProteinChainRecord]:
    if isinstance(X, pd.DataFrame):
        return X.fillna("").astype(str).to_dict(orient="records")
    if isinstance(X, ProteinChainRecord):
        return [X]
    if isinstance(X, (str, Path)):
        return [_normalize_row(X)]
    if isinstance(X, Iterable):
        rows = []
        for item in X:
            if isinstance(item, ProteinChainRecord):
                rows.append(item)
            else:
                rows.append(_normalize_row(item))
        return rows
    raise TypeError("Unsupported input for ProteinContactNetworkGraphicalizer")


def _normalize_row(row: dict[str, object] | str | Path) -> dict[str, str]:
    if isinstance(row, dict):
        normalized = {key: "" if value is None else str(value) for key, value in row.items()}
        normalized.setdefault("sample_id", "")
        normalized.setdefault("pdb_id", "")
        normalized.setdefault("label_asym_id", "")
        normalized.setdefault("auth_asym_id", "")
        normalized["pdb_id"] = normalized["pdb_id"].strip().lower()
        normalized["sample_id"] = normalized["sample_id"].strip()
        normalized["label_asym_id"] = normalized["label_asym_id"].strip()
        normalized["auth_asym_id"] = normalized["auth_asym_id"].strip()
        return normalized
    pdb_id, chain = _split_pdb_token(str(row))
    return {"sample_id": f"{pdb_id}.{chain}" if chain else pdb_id, "pdb_id": pdb_id, "label_asym_id": chain, "auth_asym_id": ""}


def _split_pdb_token(token: str) -> tuple[str, str]:
    text = token.strip()
    if "." in text:
        pdb_id, chain = text.split(".", 1)
    elif "_" in text:
        pdb_id, chain = text.split("_", 1)
    else:
        pdb_id, chain = text, ""
    pdb_id = pdb_id.strip().lower()
    chain = chain.strip()
    if len(pdb_id) != 4:
        raise ValueError(f"Invalid PDB id {pdb_id!r}; expected a 4-character PDB accession")
    return pdb_id, chain


def _sequence_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), ""
    except ValueError:
        return 10**9, value


def _edge_width_from_distance(
    distance: float,
    *,
    min_distance: float,
    max_distance: float,
    min_edge_width: float,
    max_edge_width: float,
) -> float:
    clipped = min(max(distance, min_distance), max_distance)
    if max_distance == min_distance:
        return float(max_edge_width)
    closeness = 1.0 - ((clipped - min_distance) / (max_distance - min_distance))
    return float(min_edge_width + closeness * (max_edge_width - min_edge_width))
