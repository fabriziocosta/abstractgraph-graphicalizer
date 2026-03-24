"""Helpers for loading local PubChem assay exports and ZINC CSV exports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable, Iterator, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import requests

from abstractgraph_graphicalizer.chem.molecules import (
    MoleculeParseError,
    normalize_graph_schema,
    sdf_to_graphs,
    smiles_to_graph,
)

ZINC_250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)
DEFAULT_ZINC_TARGET_COLUMNS = ("logP", "qed", "SAS")
_ZINC_ROW_INDEX_COLUMN = "_zinc_row_index"


def bundled_pubchem_root() -> Path:
    """Return the tracked bundled PubChem dataset root."""
    return Path(__file__).resolve().parents[3] / "data" / "PUBCHEM"


def local_pubchem_root() -> Path:
    """Return the ignored local PubChem dataset root."""
    return Path(__file__).resolve().parents[3] / "data-local" / "PUBCHEM"


def pubchem_search_roots() -> list[Path]:
    """Return candidate PubChem dataset roots in preference order."""
    roots: list[Path] = []
    env_root = os.environ.get("ABSTRACTGRAPH_PUBCHEM_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend([local_pubchem_root(), bundled_pubchem_root()])
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def default_pubchem_root() -> Path:
    """Return the preferred local PubChem export directory."""
    for root in pubchem_search_roots():
        if root.exists():
            return root
    return bundled_pubchem_root()


def bundled_zinc_root() -> Path:
    """Return the tracked bundled ZINC dataset root."""
    return Path(__file__).resolve().parents[3] / "data" / "ZINC"


def local_zinc_root() -> Path:
    """Return the ignored local ZINC dataset root."""
    return Path(__file__).resolve().parents[3] / "data-local" / "ZINC"


def zinc_search_roots() -> list[Path]:
    """Return candidate ZINC dataset roots in preference order."""
    roots: list[Path] = []
    env_root = os.environ.get("ABSTRACTGRAPH_ZINC_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend([local_zinc_root(), bundled_zinc_root()])
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def default_zinc_root() -> Path:
    """Return the preferred local ZINC export directory."""
    for root in zinc_search_roots():
        if root.exists():
            return root
    return bundled_zinc_root()


def _normalize_assay_id(assay_id: str | int) -> str:
    raw = str(assay_id).strip()
    if raw.lower().startswith("aid"):
        raw = raw[3:]
    if not raw:
        raise ValueError("assay_id must not be empty")
    return raw


def _take_limit(graph_iter: Iterator[nx.Graph], limit: int | None) -> list[nx.Graph]:
    if limit is None:
        return list(graph_iter)
    graphs: list[nx.Graph] = []
    for graph in graph_iter:
        graphs.append(graph)
        if len(graphs) >= limit:
            break
    return graphs


def _count_sdf_records(path: Path) -> int:
    """Quickly count SDF records by scanning for `$$$$` separators."""
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"$$$$")
    return count


@dataclass(frozen=True)
class PubChemAssayPaths:
    """Resolved file paths for a local PubChem assay export."""

    assay_id: str
    active_path: Path
    inactive_path: Path


@dataclass(frozen=True)
class PubChemAssaySummary:
    """Summary metadata for one locally available PubChem assay export."""

    assay_id: str
    active_path: Path
    inactive_path: Path
    active_size_bytes: int
    inactive_size_bytes: int
    active_molecule_count: int
    inactive_molecule_count: int

    @property
    def total_size_bytes(self) -> int:
        return self.active_size_bytes + self.inactive_size_bytes

    @property
    def total_molecule_count(self) -> int:
        return self.active_molecule_count + self.inactive_molecule_count


@dataclass(frozen=True)
class ZINCDatasetPaths:
    """Resolved file path for one local ZINC CSV export."""

    dataset_name: str
    csv_path: Path


@dataclass(frozen=True)
class ZINCDatasetSummary:
    """Summary metadata for one locally available ZINC CSV export."""

    dataset_name: str
    csv_path: Path
    size_bytes: int
    molecule_count: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024


class SupervisedDatasetLoader:
    """Small dataset shaping helper for notebook-scale supervised workflows."""

    def __init__(
        self,
        load_func: Callable[[], tuple[Sequence[object], Sequence[object]]] | None = None,
        size: int | None = None,
        use_targets_list: Sequence[object] | None = None,
        use_equalized: bool = False,
        use_multiclass_to_binary: bool = False,
        use_regression_to_binary: bool = False,
        regression_to_binary_threshold: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.load_func = load_func
        self.size = size
        self.use_targets_list = use_targets_list
        self.use_equalized = use_equalized
        self.use_multiclass_to_binary = use_multiclass_to_binary
        self.use_regression_to_binary = use_regression_to_binary
        self.regression_to_binary_threshold = regression_to_binary_threshold
        self.random_state = random_state

    def _rng(self):
        return np.random.default_rng(self.random_state)

    @staticmethod
    def _restore_type(values: list[object], original):
        if isinstance(original, np.ndarray):
            return np.asarray(values, dtype=original.dtype if getattr(original, "dtype", None) is not None else None)
        return values

    def resize(self, data, targets, size: int):
        rng = self._rng()
        idxs = rng.choice(len(targets), size=int(size), replace=False)
        data_out = [data[idx] for idx in idxs]
        targets_out = [targets[idx] for idx in idxs]
        return self._restore_type(data_out, data), self._restore_type(targets_out, targets)

    def resize_equalized(self, data, targets, size: int):
        rng = self._rng()
        target_values = list(sorted(set(targets)))
        if not target_values:
            return data, targets
        per_target = int(size) // len(target_values)
        remainder = int(size) % len(target_values)
        sampled_indices: list[int] = []
        for target_index, target_value in enumerate(target_values):
            candidate_indices = [idx for idx, target in enumerate(targets) if target == target_value]
            take = per_target + (1 if target_index < remainder else 0)
            sampled_indices.extend(rng.choice(candidate_indices, size=take, replace=False).tolist())
        rng.shuffle(sampled_indices)
        data_out = [data[idx] for idx in sampled_indices]
        targets_out = [targets[idx] for idx in sampled_indices]
        return self._restore_type(data_out, data), self._restore_type(targets_out, targets)

    def equalize(self, data, targets):
        rng = self._rng()
        target_values = list(sorted(set(targets)))
        idxs_list = [[idx for idx in range(len(targets)) if targets[idx] == target_value] for target_value in target_values]
        min_size = min(len(idxs) for idxs in idxs_list)
        sampled = [rng.choice(idxs, size=min_size, replace=False).tolist() for idxs in idxs_list]
        data_out = [data[idx] for idxs in sampled for idx in idxs]
        targets_out = [targets[idx] for idxs in sampled for idx in idxs]
        return self._restore_type(data_out, data), self._restore_type(targets_out, targets)

    @staticmethod
    def binarize_multiclass(targets):
        targets_out = [target % 2 for target in targets]
        return np.asarray(targets_out) if isinstance(targets, np.ndarray) else targets_out

    def binarize_regression(self, targets):
        if self.regression_to_binary_threshold is None:
            raise ValueError("regression_to_binary_threshold must be set when use_regression_to_binary=True")
        targets_out = [target < self.regression_to_binary_threshold for target in targets]
        return np.asarray(targets_out) if isinstance(targets, np.ndarray) else targets_out

    def keep_target(self, data, targets):
        target_values = set(self.use_targets_list or [])
        idxs = [idx for idx, target in enumerate(targets) if target in target_values]
        data_out = [data[idx] for idx in idxs]
        targets_out = [targets[idx] for idx in idxs]
        return self._restore_type(data_out, data), self._restore_type(targets_out, targets)

    def load(self):
        if self.load_func is None:
            raise ValueError("load_func must be provided")
        data, targets = self.load_func()
        if self.use_targets_list is not None:
            data, targets = self.keep_target(data, targets)
        if self.use_equalized:
            data, targets = self.equalize(data, targets)
        if self.size is not None and self.size < len(targets):
            if self.use_equalized:
                data, targets = self.resize_equalized(data, targets, self.size)
            else:
                data, targets = self.resize(data, targets, self.size)
        if self.use_multiclass_to_binary:
            targets = self.binarize_multiclass(targets)
        if self.use_regression_to_binary:
            targets = self.binarize_regression(targets)
        return data, targets


SupervisedDataSetLoader = SupervisedDatasetLoader


class PubChemLoader:
    """Load local PubChem assay exports split into active/inactive SDF files.

    Expected file naming convention inside ``root``:

    - ``AID<assay_id>_active.sdf``
    - ``AID<assay_id>_inactive.sdf``

    The main public methods are:

    - ``resolve_paths(assay_id)`` to inspect the expected file locations
    - ``load_split(assay_id)`` to load the two graph lists separately
    - ``load(assay_id)`` to load a combined graph dataset plus binary targets
    """

    def __init__(self, root: str | Path | None = None, *, on_error: str = "raise") -> None:
        self.root = Path(root) if root is not None else default_pubchem_root()
        self.on_error = on_error

    def available_assay_ids(self) -> list[str]:
        assay_ids = {
            path.name.split("_")[0][3:]
            for path in self.root.glob("AID*_active.sdf")
            if (self.root / f"{path.name.split('_')[0]}_inactive.sdf").exists()
        }
        return sorted(assay_ids)

    def list_assays(self) -> list[PubChemAssaySummary]:
        """List locally available paired assay exports with sizes and record counts."""
        summaries: list[PubChemAssaySummary] = []
        for assay_id in self.available_assay_ids():
            paths = self.resolve_paths(assay_id)
            summaries.append(
                PubChemAssaySummary(
                    assay_id=assay_id,
                    active_path=paths.active_path,
                    inactive_path=paths.inactive_path,
                    active_size_bytes=paths.active_path.stat().st_size,
                    inactive_size_bytes=paths.inactive_path.stat().st_size,
                    active_molecule_count=_count_sdf_records(paths.active_path),
                    inactive_molecule_count=_count_sdf_records(paths.inactive_path),
                )
            )
        return summaries

    def format_assay_table(self, *, sort_by: str = "total_molecule_count", reverse: bool = True) -> str:
        """Return a formatted assay summary table for the current root."""
        summaries = self.list_assays()
        valid_sort_keys = {
            "assay_id",
            "active_molecule_count",
            "inactive_molecule_count",
            "total_molecule_count",
            "active_size_bytes",
            "inactive_size_bytes",
            "total_size_bytes",
        }
        if sort_by not in valid_sort_keys:
            raise ValueError(f"sort_by must be one of {sorted(valid_sort_keys)}")
        summaries = sorted(summaries, key=lambda assay: getattr(assay, sort_by), reverse=reverse)
        lines = [
            "{:<10} {:>12} {:>14} {:>11} {:>10}".format(
                "assay_id", "active_mols", "inactive_mols", "total_mols", "total_mb"
            ),
            "-" * 64,
        ]
        for assay in summaries:
            lines.append(
                f"{assay.assay_id:<10} "
                f"{assay.active_molecule_count:>12} "
                f"{assay.inactive_molecule_count:>14} "
                f"{assay.total_molecule_count:>11} "
                f"{assay.total_size_bytes / 1024 / 1024:>10.2f}"
            )
        return "\n".join(lines)

    def resolve_paths(self, assay_id: str | int) -> PubChemAssayPaths:
        normalized = _normalize_assay_id(assay_id)
        active_path = self.root / f"AID{normalized}_active.sdf"
        inactive_path = self.root / f"AID{normalized}_inactive.sdf"
        if not active_path.exists():
            raise FileNotFoundError(active_path)
        if not inactive_path.exists():
            raise FileNotFoundError(inactive_path)
        return PubChemAssayPaths(
            assay_id=normalized,
            active_path=active_path,
            inactive_path=inactive_path,
        )

    def _annotate_graphs(
        self,
        graphs: list[nx.Graph],
        *,
        assay_id: str,
        activity_label: str,
        target: int,
    ) -> list[nx.Graph]:
        for graph in graphs:
            graph.graph["pubchem_aid"] = assay_id
            graph.graph["pubchem_activity"] = activity_label
            graph.graph["target"] = target
        return graphs

    def load_split(
        self,
        assay_id: str | int,
        *,
        limit: int | None = None,
        limit_active: int | None = None,
        limit_inactive: int | None = None,
    ) -> tuple[list[nx.Graph], list[nx.Graph]]:
        paths = self.resolve_paths(assay_id)
        if limit_active is None:
            limit_active = limit
        if limit_inactive is None:
            limit_inactive = limit
        active_graphs = _take_limit(
            sdf_to_graphs(paths.active_path, on_error=self.on_error),
            limit_active,
        )
        inactive_graphs = _take_limit(
            sdf_to_graphs(paths.inactive_path, on_error=self.on_error),
            limit_inactive,
        )
        return (
            self._annotate_graphs(
                active_graphs,
                assay_id=paths.assay_id,
                activity_label="active",
                target=1,
            ),
            self._annotate_graphs(
                inactive_graphs,
                assay_id=paths.assay_id,
                activity_label="inactive",
                target=0,
            ),
        )

    def load(
        self,
        assay_id: str | int,
        *,
        limit: int | None = None,
        limit_active: int | None = None,
        limit_inactive: int | None = None,
    ) -> tuple[list[nx.Graph], list[int]]:
        active_graphs, inactive_graphs = self.load_split(
            assay_id,
            limit=limit,
            limit_active=limit_active,
            limit_inactive=limit_inactive,
        )
        graphs = inactive_graphs + active_graphs
        targets = [0] * len(inactive_graphs) + [1] * len(active_graphs)
        return graphs, targets


class ZINCLoader:
    """Load local ZINC CSV exports containing `smiles` rows and optional targets.

    Expected file naming convention inside ``root``:

    - ``<dataset_name>.csv`` such as ``zinc_250k.csv``

    The main public methods are:

    - ``resolve_paths(dataset_name)`` to inspect one dataset CSV path
    - ``load_frame(dataset_name)`` to read the CSV into a dataframe
    - ``load(dataset_name)`` to build molecule graphs plus aligned metadata
    """

    def __init__(self, root: str | Path | None = None, *, on_error: str = "raise") -> None:
        self.root = Path(root) if root is not None else default_zinc_root()
        self.on_error = on_error

    def _cache_root(self, dataset_name: str = "zinc_250k") -> Path:
        paths = self.resolve_paths(dataset_name)
        return self.root / "graph_corpus_cache" / paths.dataset_name

    def available_datasets(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.csv"))

    def list_datasets(self) -> list[ZINCDatasetSummary]:
        summaries: list[ZINCDatasetSummary] = []
        for dataset_name in self.available_datasets():
            paths = self.resolve_paths(dataset_name)
            frame = pd.read_csv(paths.csv_path)
            summaries.append(
                ZINCDatasetSummary(
                    dataset_name=dataset_name,
                    csv_path=paths.csv_path,
                    size_bytes=paths.csv_path.stat().st_size,
                    molecule_count=len(frame),
                )
            )
        return summaries

    def format_dataset_table(self, *, sort_by: str = "molecule_count", reverse: bool = True) -> str:
        summaries = self.list_datasets()
        valid_sort_keys = {
            "dataset_name",
            "molecule_count",
            "size_bytes",
            "size_mb",
        }
        if sort_by not in valid_sort_keys:
            raise ValueError(f"sort_by must be one of {sorted(valid_sort_keys)}")
        summaries = sorted(summaries, key=lambda dataset: getattr(dataset, sort_by), reverse=reverse)
        lines = [
            "{:<20} {:>12} {:>10}".format("dataset", "molecules", "size_mb"),
            "-" * 48,
        ]
        for dataset in summaries:
            lines.append(
                f"{dataset.dataset_name:<20} "
                f"{dataset.molecule_count:>12} "
                f"{dataset.size_mb:>10.2f}"
            )
        return "\n".join(lines)

    def resolve_paths(self, dataset_name: str = "zinc_250k") -> ZINCDatasetPaths:
        normalized = str(dataset_name).strip()
        if not normalized:
            raise ValueError("dataset_name must not be empty")
        if normalized.endswith(".csv"):
            normalized = normalized[:-4]
        csv_path = self.root / f"{normalized}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        return ZINCDatasetPaths(dataset_name=normalized, csv_path=csv_path)

    def load_frame(self, dataset_name: str = "zinc_250k", *, limit: int | None = None) -> pd.DataFrame:
        paths = self.resolve_paths(dataset_name)
        frame = pd.read_csv(paths.csv_path)
        if limit is not None:
            frame = frame.iloc[: int(limit)].reset_index(drop=True)
        return frame

    def _ensure_graph_corpus(
        self,
        dataset_name: str = "zinc_250k",
        *,
        force: bool = False,
    ) -> dict:
        paths = self.resolve_paths(dataset_name)
        cache_root = self._cache_root(paths.dataset_name)
        cache_root.mkdir(parents=True, exist_ok=True)
        return build_zinc_graph_corpus(cache_root, paths.csv_path, force=force)

    def _graph_from_row(self, row: dict, *, dataset_name: str, row_index: int) -> nx.Graph | None:
        smiles = row.get("smiles")
        if smiles is None or not str(smiles).strip():
            if self.on_error == "skip":
                return None
            raise MoleculeParseError("Missing SMILES", f"{dataset_name}[{row_index}]")
        try:
            graph = smiles_to_graph(str(smiles))
        except MoleculeParseError:
            if self.on_error == "skip":
                return None
            raise
        graph.graph["source"] = "zinc"
        graph.graph["input"] = f"{dataset_name}[{row_index}]"
        graph.graph["zinc_dataset"] = dataset_name
        for key, value in row.items():
            if pd.isna(value):
                continue
            graph.graph[key] = value
        return graph

    def load(
        self,
        dataset_name: str = "zinc_250k",
        *,
        limit: int | None = None,
        min_node_count: int | None = None,
        max_node_count: int | None = None,
    ) -> tuple[list[nx.Graph], pd.DataFrame]:
        self._ensure_graph_corpus(dataset_name)
        return load_zinc_graph_dataset(
            self._cache_root(dataset_name),
            max_molecules=limit,
            min_node_count=min_node_count,
            max_node_count=max_node_count,
        )


def load_pubchem_graph_dataset(
    root: str | Path | None = None,
    assay_id: str = "651610",
    dataset_size: int | None = None,
    max_node_count: int | None = None,
    use_equalized: bool = False,
    random_state: int = 0,
) -> tuple[list[nx.Graph], np.ndarray, pd.DataFrame]:
    """Load one PubChem assay with optional shaping and node-count filtering."""
    loader = PubChemLoader(root)

    def load_func():
        return loader.load(assay_id)

    graphs, targets = SupervisedDatasetLoader(
        load_func=load_func,
        size=dataset_size,
        use_equalized=use_equalized,
        random_state=random_state,
    ).load()
    graphs = np.asarray(graphs, dtype=object)
    targets = np.asarray(targets)
    metadata = pd.DataFrame(
        {
            "target": targets,
            "node_count": [graph.number_of_nodes() for graph in graphs],
            "edge_count": [graph.number_of_edges() for graph in graphs],
        }
    )
    if max_node_count is not None:
        keep = metadata["node_count"].to_numpy() <= int(max_node_count)
        graphs = graphs[keep]
        targets = targets[keep]
        metadata = metadata.loc[keep].reset_index(drop=True)
    return graphs.tolist(), targets, metadata


def download_zinc_dataset(
    dataset_dir: str | Path,
    url: str = ZINC_250K_URL,
    filename: str = "zinc_250k.csv",
    force: bool = False,
) -> Path:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / filename
    if csv_path.exists() and not force:
        return csv_path
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    csv_path.write_bytes(response.content)
    return csv_path


def _zinc_graph_bucket_path(dataset_dir: Path, node_count: int) -> Path:
    return dataset_dir / "graph_corpus" / f"graphs_nodes_{int(node_count):03d}.pkl"


def _normalize_zinc_cached_path(dataset_dir: Path, cached_path: Path | str) -> str:
    path = Path(cached_path).expanduser()
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(dataset_dir))
    except ValueError:
        return str(path)


def _resolve_zinc_bucket_path(dataset_dir: Path, cached_path: Path | str, node_count: int) -> Path:
    path = Path(cached_path).expanduser()
    if not path.is_absolute():
        return (dataset_dir / path).resolve()
    if path.exists():
        return path
    return _zinc_graph_bucket_path(dataset_dir, node_count)


def _normalize_zinc_corpus_manifest(dataset_dir: Path, manifest: dict) -> tuple[dict, bool]:
    normalized = dict(manifest)
    changed = False
    node_counts = [int(node_count) for node_count in normalized.get("node_counts", [])]
    if normalized.get("node_counts") != node_counts:
        normalized["node_counts"] = node_counts
        changed = True
    bucket_files = normalized.get("bucket_files")
    if bucket_files is None:
        normalized["bucket_files"] = {
            node_count: str(_zinc_graph_bucket_path(dataset_dir, node_count))
            for node_count in node_counts
        }
        changed = True
    else:
        normalized_bucket_files = {
            int(node_count): _normalize_zinc_cached_path(dataset_dir, bucket_path)
            for node_count, bucket_path in bucket_files.items()
        }
        if normalized_bucket_files != bucket_files:
            normalized["bucket_files"] = normalized_bucket_files
            changed = True
    csv_path = normalized.get("csv_path")
    if csv_path is not None:
        normalized_csv_path = _normalize_zinc_cached_path(dataset_dir, csv_path)
        if normalized_csv_path != csv_path:
            normalized["csv_path"] = normalized_csv_path
            changed = True
    return normalized, changed


def _normalize_legacy_graph_item(item: object) -> tuple[nx.Graph, dict]:
    if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError("Unsupported ZINC bucket item format.")
    graph, row = item
    if not isinstance(graph, nx.Graph):
        raise ValueError("Unsupported ZINC graph payload.")
    row_dict = dict(row)
    return normalize_graph_schema(graph, copy=False), row_dict


def _normalize_zinc_bucket_items(items: object) -> tuple[list[tuple[nx.Graph, dict]], bool]:
    if isinstance(items, list):
        normalized_items = [_normalize_legacy_graph_item(item) for item in items]
        changed = any(
            normalized_items[idx][0] is not items[idx][0] or normalized_items[idx][1] is not items[idx][1]
            for idx in range(len(normalized_items))
        )
        return normalized_items, changed
    if isinstance(items, dict) and "graphs" in items and "metadata" in items:
        graphs = list(items["graphs"])
        metadata = items["metadata"]
        if isinstance(metadata, pd.DataFrame):
            rows = metadata.to_dict(orient="records")
        elif isinstance(metadata, list):
            rows = [dict(row) for row in metadata]
        else:
            raise ValueError("Unsupported legacy ZINC bucket metadata format.")
        if len(graphs) != len(rows):
            raise ValueError("Legacy ZINC bucket graphs and metadata lengths differ.")
        return [(normalize_graph_schema(graph, copy=False), row) for graph, row in zip(graphs, rows)], True
    raise ValueError("Unsupported ZINC bucket format.")


def build_zinc_graph_corpus(
    dataset_dir: str | Path,
    csv_path: str | Path,
    force: bool = False,
) -> dict:
    """Convert a ZINC CSV table into cached graph buckets grouped by node count."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    corpus_dir = dataset_dir / "graph_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.pkl"
    if manifest_path.exists() and not force:
        with manifest_path.open("rb") as handle:
            manifest = pickle.load(handle)
        manifest, changed = _normalize_zinc_corpus_manifest(dataset_dir, manifest)
        if changed:
            with manifest_path.open("wb") as handle:
                pickle.dump(manifest, handle)
        return manifest

    frame = pd.read_csv(csv_path)
    loader = ZINCLoader(dataset_dir, on_error="skip")
    buckets: dict[int, list[tuple[nx.Graph, dict]]] = {}
    invalid_smiles_count = 0
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        graph = loader._graph_from_row(row, dataset_name=Path(csv_path).stem, row_index=row_index)
        if graph is None:
            invalid_smiles_count += 1
            continue
        node_count = graph.number_of_nodes()
        row_payload = dict(row)
        row_payload[_ZINC_ROW_INDEX_COLUMN] = row_index
        buckets.setdefault(node_count, []).append((graph, row_payload))

    total_graphs = 0
    node_counts = sorted(buckets)
    for node_count, items in buckets.items():
        bucket_path = _zinc_graph_bucket_path(dataset_dir, node_count)
        with bucket_path.open("wb") as handle:
            pickle.dump(items, handle)
        total_graphs += len(items)

    manifest = {
        "csv_path": _normalize_zinc_cached_path(dataset_dir, csv_path),
        "total_graphs": total_graphs,
        "invalid_smiles_count": invalid_smiles_count,
        "node_counts": node_counts,
        "bucket_files": {
            node_count: str(_zinc_graph_bucket_path(dataset_dir, node_count))
            for node_count in node_counts
        },
    }
    with manifest_path.open("wb") as handle:
        pickle.dump(manifest, handle)
    return manifest


def load_zinc_graph_dataset(
    dataset_dir: str | Path,
    max_molecules: int | None = 100_000,
    min_node_count: int | None = None,
    max_node_count: int | None = 40,
) -> tuple[list[nx.Graph], pd.DataFrame]:
    """Load cached ZINC graphs and metadata from a graph corpus directory."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "graph_corpus" / "manifest.pkl"
    with manifest_path.open("rb") as handle:
        manifest = pickle.load(handle)
    manifest, changed = _normalize_zinc_corpus_manifest(dataset_dir, manifest)
    if changed:
        with manifest_path.open("wb") as handle:
            pickle.dump(manifest, handle)

    selected_node_counts = [
        node_count
        for node_count in manifest["node_counts"]
        if (min_node_count is None or node_count >= int(min_node_count))
        and (max_node_count is None or node_count <= int(max_node_count))
    ]
    graphs: list[nx.Graph] = []
    metadata_rows: list[dict] = []
    for node_count in selected_node_counts:
        bucket_path = _resolve_zinc_bucket_path(dataset_dir, manifest["bucket_files"][node_count], node_count)
        with bucket_path.open("rb") as handle:
            items = pickle.load(handle)
        normalized_items, bucket_changed = _normalize_zinc_bucket_items(items)
        if bucket_changed:
            with bucket_path.open("wb") as handle:
                pickle.dump(normalized_items, handle)
        for graph, row in normalized_items:
            graphs.append(graph)
            metadata_rows.append(row)
            if max_molecules is not None and len(graphs) >= int(max_molecules):
                break
        if max_molecules is not None and len(graphs) >= int(max_molecules):
            break
    if metadata_rows and any(_ZINC_ROW_INDEX_COLUMN in row for row in metadata_rows):
        ordered = sorted(
            zip(graphs, metadata_rows),
            key=lambda item: int(item[1].get(_ZINC_ROW_INDEX_COLUMN, 10**12)),
        )
        graphs = [graph for graph, _ in ordered]
        metadata_rows = [dict(row) for _, row in ordered]
    for row in metadata_rows:
        row.pop(_ZINC_ROW_INDEX_COLUMN, None)
    return graphs, pd.DataFrame(metadata_rows)


def extract_zinc_targets(
    metadata: pd.DataFrame,
    target_columns: Sequence[str] = DEFAULT_ZINC_TARGET_COLUMNS,
) -> pd.DataFrame:
    """Extract requested target columns from ZINC metadata."""
    return metadata.loc[:, list(target_columns)].copy()
