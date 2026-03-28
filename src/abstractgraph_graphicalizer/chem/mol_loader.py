"""Shared helpers for molecular dataset loaders."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterator, Sequence, TypeVar

import networkx as nx
import numpy as np
import pandas as pd
import requests

from abstractgraph_graphicalizer.chem.molecules import (
    MoleculeParseError,
    normalize_graph_schema,
    smiles_to_graph,
)

DEFAULT_TABLE_SORT_BY = "molecule_count"
_ROW_INDEX_COLUMN = "_molecule_row_index"

DatasetPathsT = TypeVar("DatasetPathsT")
DatasetSummaryT = TypeVar("DatasetSummaryT")


@dataclass(frozen=True)
class GenericCSVDatasetPaths:
    """Resolved file path for one local CSV molecular dataset export."""

    dataset_name: str
    csv_path: Path


@dataclass(frozen=True)
class GenericCSVDatasetSummary:
    """Summary metadata for one locally available CSV molecular dataset export."""

    dataset_name: str
    csv_path: Path
    size_bytes: int
    molecule_count: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024


def search_roots(
    *,
    env_var: str,
    local_root: Callable[[], Path],
    bundled_root: Callable[[], Path],
) -> list[Path]:
    """Return candidate dataset roots in preference order."""
    roots: list[Path] = []
    env_root = os.environ.get(env_var)
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend([local_root(), bundled_root()])
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def default_root(
    *,
    env_var: str,
    local_root: Callable[[], Path],
    bundled_root: Callable[[], Path],
) -> Path:
    """Return the preferred dataset root based on environment and local files."""
    for root in search_roots(env_var=env_var, local_root=local_root, bundled_root=bundled_root):
        if root.exists():
            return root
    return bundled_root()


def download_dataset(
    dataset_dir: str | Path,
    *,
    url: str,
    filename: str,
    force: bool = False,
) -> Path:
    """Download one CSV dataset to the requested directory."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / filename
    if csv_path.exists() and not force:
        return csv_path
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    csv_path.write_bytes(response.content)
    return csv_path


def graph_corpus_dir(cache_root: Path) -> Path:
    """Return the graph corpus directory for a dataset cache root."""
    return cache_root / "graph_corpus"


def shared_cache_parent(dataset_dir: Path) -> Path:
    """Return the parent directory that should contain graph-corpus caches."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    parent = dataset_dir.parent
    if parent.name in {"data", "data-local"}:
        return parent / "graph_corpus_cache"
    return dataset_dir / "graph_corpus_cache"


def dataset_cache_root(dataset_dir: Path, dataset_name: str) -> Path:
    """Return the cache root for one dataset name."""
    return shared_cache_parent(dataset_dir) / str(dataset_name).strip()


def normalize_dataset_name(dataset_name: str, *, suffix: str = ".csv") -> str:
    """Normalize a user-facing dataset identifier."""
    normalized = str(dataset_name).strip()
    if not normalized:
        raise ValueError("dataset_name must not be empty")
    if suffix and normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized


def take_limit(graph_iter: Iterator[nx.Graph], limit: int | None) -> list[nx.Graph]:
    """Materialize at most `limit` graphs from an iterator."""
    if limit is None:
        return list(graph_iter)
    graphs: list[nx.Graph] = []
    for graph in graph_iter:
        graphs.append(graph)
        if len(graphs) >= limit:
            break
    return graphs


def csv_dataset_table(
    summaries: Sequence[GenericCSVDatasetSummary],
    *,
    sort_by: str = DEFAULT_TABLE_SORT_BY,
    reverse: bool = True,
) -> str:
    """Format a short tabular overview for CSV datasets."""
    valid_sort_keys = {
        "dataset_name",
        "molecule_count",
        "size_bytes",
        "size_mb",
    }
    if sort_by not in valid_sort_keys:
        raise ValueError(f"sort_by must be one of {sorted(valid_sort_keys)}")
    ordered = sorted(summaries, key=lambda dataset: getattr(dataset, sort_by), reverse=reverse)
    lines = [
        "{:<20} {:>12} {:>10}".format("dataset", "molecules", "size_mb"),
        "-" * 48,
    ]
    for dataset in ordered:
        lines.append(
            f"{dataset.dataset_name:<20} "
            f"{dataset.molecule_count:>12} "
            f"{dataset.size_mb:>10.2f}"
        )
    return "\n".join(lines)


def normalize_cached_path(cache_root: Path, cached_path: Path | str) -> str:
    """Persist paths relative to the cache root when possible."""
    path = Path(cached_path).expanduser()
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(cache_root))
    except ValueError:
        return str(path)


def dataset_graph_bucket_path(cache_root: Path, node_count: int) -> Path:
    """Return one node-count bucket path inside a graph corpus cache."""
    return graph_corpus_dir(cache_root) / f"graphs_nodes_{int(node_count):03d}.pkl"


def resolve_cache_root(dataset_dir: Path, csv_path: Path | str | None = None) -> Path:
    """Resolve a dataset directory or cache directory into a cache root."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if graph_corpus_dir(dataset_dir).exists():
        return dataset_dir
    if csv_path is not None:
        return dataset_cache_root(dataset_dir, Path(csv_path).stem)
    cache_parent = shared_cache_parent(dataset_dir)
    if cache_parent.exists():
        candidates = sorted(
            path.name
            for path in cache_parent.iterdir()
            if path.is_dir() and graph_corpus_dir(path).exists()
        )
        if len(candidates) == 1:
            return dataset_cache_root(dataset_dir, candidates[0])
    legacy_corpus_dir = dataset_dir / "graph_corpus"
    if legacy_corpus_dir.exists():
        return dataset_dir
    raise FileNotFoundError(
        "Could not infer a unique dataset cache. Pass a dataset root that contains exactly one "
        "cached dataset or use the loader `.load(...)` API."
    )


def resolve_bucket_path(cache_root: Path, cached_path: Path | str, node_count: int) -> Path:
    """Resolve one cached bucket path from a manifest entry."""
    path = Path(cached_path).expanduser()
    if not path.is_absolute():
        return (cache_root / path).resolve()
    if path.exists():
        return path
    return dataset_graph_bucket_path(cache_root, node_count)


def resolve_bucket_paths(
    cache_root: Path,
    cached_path: Path | str | Sequence[Path | str],
    node_count: int,
) -> list[Path]:
    """Resolve one or more cached bucket paths from a manifest entry."""
    if isinstance(cached_path, (list, tuple)):
        return [resolve_bucket_path(cache_root, path, node_count) for path in cached_path]
    return [resolve_bucket_path(cache_root, cached_path, node_count)]


def normalize_corpus_manifest(cache_root: Path, manifest: dict) -> tuple[dict, bool]:
    """Normalize a cached graph-corpus manifest in memory."""
    normalized = dict(manifest)
    changed = False
    node_counts = [int(node_count) for node_count in normalized.get("node_counts", [])]
    if normalized.get("node_counts") != node_counts:
        normalized["node_counts"] = node_counts
        changed = True
    bucket_files = normalized.get("bucket_files")
    if bucket_files is None:
        normalized["bucket_files"] = {
            node_count: str(dataset_graph_bucket_path(cache_root, node_count))
            for node_count in node_counts
        }
        changed = True
    else:
        normalized_bucket_files = {}
        for node_count, bucket_path in bucket_files.items():
            normalized_node_count = int(node_count)
            if isinstance(bucket_path, (list, tuple)):
                normalized_paths = [normalize_cached_path(cache_root, path) for path in bucket_path]
            else:
                normalized_paths = normalize_cached_path(cache_root, bucket_path)
            normalized_bucket_files[normalized_node_count] = normalized_paths
        if normalized_bucket_files != bucket_files:
            normalized["bucket_files"] = normalized_bucket_files
            changed = True
    csv_path = normalized.get("csv_path")
    if csv_path is not None:
        normalized_csv_path = normalize_cached_path(cache_root, csv_path)
        if normalized_csv_path != csv_path:
            normalized["csv_path"] = normalized_csv_path
            changed = True
    return normalized, changed


def normalize_legacy_graph_item(item: object) -> tuple[nx.Graph, dict]:
    """Normalize one cached `(graph, metadata)` pair."""
    if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError("Unsupported cached graph item format.")
    graph, row = item
    if not isinstance(graph, nx.Graph):
        raise ValueError("Unsupported cached graph payload.")
    row_dict = dict(row)
    return normalize_graph_schema(graph, copy=False), row_dict


def normalize_bucket_items(items: object) -> tuple[list[tuple[nx.Graph, dict]], bool]:
    """Normalize legacy cache bucket payloads into the current list-of-tuples format."""
    if isinstance(items, list):
        normalized_items = [normalize_legacy_graph_item(item) for item in items]
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
            raise ValueError("Unsupported legacy cached metadata format.")
        if len(graphs) != len(rows):
            raise ValueError("Legacy cached graphs and metadata lengths differ.")
        return [(normalize_graph_schema(graph, copy=False), row) for graph, row in zip(graphs, rows)], True
    raise ValueError("Unsupported cached graph bucket format.")


def build_graph_corpus(
    dataset_dir: str | Path,
    csv_path: str | Path,
    *,
    graph_builder: Callable[[dict, int], nx.Graph | None],
    force: bool = False,
) -> dict:
    """Convert a CSV table into cached graph buckets grouped by node count."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve()
    cache_root = dataset_cache_root(dataset_dir, csv_path.stem)
    corpus_dir = graph_corpus_dir(cache_root)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.pkl"
    if manifest_path.exists() and not force:
        with manifest_path.open("rb") as handle:
            manifest = pickle.load(handle)
        manifest, changed = normalize_corpus_manifest(cache_root, manifest)
        if changed:
            with manifest_path.open("wb") as handle:
                pickle.dump(manifest, handle)
        return manifest

    frame = pd.read_csv(csv_path)
    buckets: dict[int, list[tuple[nx.Graph, dict]]] = {}
    invalid_smiles_count = 0
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        graph = graph_builder(row, row_index)
        if graph is None:
            invalid_smiles_count += 1
            continue
        node_count = graph.number_of_nodes()
        row_payload = dict(row)
        row_payload[_ROW_INDEX_COLUMN] = row_index
        buckets.setdefault(node_count, []).append((graph, row_payload))

    total_graphs = 0
    node_counts = sorted(buckets)
    for node_count, items in buckets.items():
        bucket_path = dataset_graph_bucket_path(cache_root, node_count)
        with bucket_path.open("wb") as handle:
            pickle.dump(items, handle)
        total_graphs += len(items)

    manifest = {
        "dataset_name": csv_path.stem,
        "csv_path": normalize_cached_path(cache_root, csv_path),
        "total_graphs": total_graphs,
        "invalid_smiles_count": invalid_smiles_count,
        "node_counts": node_counts,
        "bucket_files": {
            node_count: str(dataset_graph_bucket_path(cache_root, node_count).relative_to(cache_root))
            for node_count in node_counts
        },
    }
    with manifest_path.open("wb") as handle:
        pickle.dump(manifest, handle)
    return manifest


def load_graph_dataset(
    dataset_dir: str | Path,
    *,
    max_molecules: int | None = 100_000,
    min_node_count: int | None = None,
    max_node_count: int | None = 40,
) -> tuple[list[nx.Graph], pd.DataFrame]:
    """Load cached graphs and metadata from a graph corpus directory."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    cache_root = resolve_cache_root(dataset_dir)
    manifest_path = graph_corpus_dir(cache_root) / "manifest.pkl"
    with manifest_path.open("rb") as handle:
        manifest = pickle.load(handle)
    manifest, changed = normalize_corpus_manifest(cache_root, manifest)
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
        bucket_paths = resolve_bucket_paths(cache_root, manifest["bucket_files"][node_count], node_count)
        for bucket_path in bucket_paths:
            with bucket_path.open("rb") as handle:
                items = pickle.load(handle)
            normalized_items, bucket_changed = normalize_bucket_items(items)
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
        if max_molecules is not None and len(graphs) >= int(max_molecules):
            break
    if metadata_rows and any(_ROW_INDEX_COLUMN in row for row in metadata_rows):
        ordered = sorted(
            zip(graphs, metadata_rows),
            key=lambda item: int(item[1].get(_ROW_INDEX_COLUMN, 10**12)),
        )
        graphs = [graph for graph, _ in ordered]
        metadata_rows = [dict(row) for _, row in ordered]
    for row in metadata_rows:
        row.pop(_ROW_INDEX_COLUMN, None)
    return graphs, pd.DataFrame(metadata_rows)


def extract_targets(metadata: pd.DataFrame, target_columns: Sequence[str]) -> pd.DataFrame:
    """Extract requested target columns from metadata."""
    return metadata.loc[:, list(target_columns)].copy()


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


class CSVMoleculeLoader(Generic[DatasetPathsT, DatasetSummaryT]):
    """Shared CSV-to-graph loader with graph-corpus caching."""

    def __init__(self, root: str | Path | None = None, *, on_error: str = "raise") -> None:
        self.root = Path(root) if root is not None else self.default_root()
        self.on_error = on_error

    @staticmethod
    def default_root() -> Path:
        raise NotImplementedError

    @property
    def default_dataset_name(self) -> str:
        raise NotImplementedError

    @property
    def graph_source_name(self) -> str:
        raise NotImplementedError

    @property
    def graph_dataset_key(self) -> str:
        raise NotImplementedError

    def auto_download_missing_default(self) -> bool:
        return False

    def ensure_default_dataset(self, *, force: bool = False) -> Path:
        raise FileNotFoundError("Automatic dataset download is not enabled for this loader.")

    def make_paths(self, dataset_name: str, csv_path: Path) -> DatasetPathsT:
        raise NotImplementedError

    def make_summary(self, dataset_name: str, csv_path: Path, frame: pd.DataFrame) -> DatasetSummaryT:
        raise NotImplementedError

    def _cache_root(self, dataset_name: str) -> Path:
        return dataset_cache_root(self.root, dataset_name)

    def available_datasets(self) -> list[str]:
        datasets = sorted(path.stem for path in self.root.glob("*.csv"))
        if not datasets and self.auto_download_missing_default():
            self.ensure_default_dataset()
            datasets = sorted(path.stem for path in self.root.glob("*.csv"))
        return datasets

    def list_datasets(self) -> list[DatasetSummaryT]:
        summaries: list[DatasetSummaryT] = []
        for dataset_name in self.available_datasets():
            paths = self.resolve_paths(dataset_name)
            frame = pd.read_csv(paths.csv_path)
            summaries.append(self.make_summary(paths.dataset_name, paths.csv_path, frame))
        return summaries

    def format_dataset_table(self, *, sort_by: str = DEFAULT_TABLE_SORT_BY, reverse: bool = True) -> str:
        return csv_dataset_table(self.list_datasets(), sort_by=sort_by, reverse=reverse)

    def resolve_paths(self, dataset_name: str | None = None) -> DatasetPathsT:
        normalized = normalize_dataset_name(dataset_name or self.default_dataset_name)
        csv_path = self.root / f"{normalized}.csv"
        if not csv_path.exists() and normalized == self.default_dataset_name and self.auto_download_missing_default():
            csv_path = self.ensure_default_dataset()
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        return self.make_paths(normalized, csv_path)

    def load_frame(self, dataset_name: str | None = None, *, limit: int | None = None) -> pd.DataFrame:
        paths = self.resolve_paths(dataset_name)
        frame = pd.read_csv(paths.csv_path)
        if limit is not None:
            frame = frame.iloc[: int(limit)].reset_index(drop=True)
        return frame

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
        graph.graph["source"] = self.graph_source_name
        graph.graph["input"] = f"{dataset_name}[{row_index}]"
        graph.graph[self.graph_dataset_key] = dataset_name
        for key, value in row.items():
            if pd.isna(value):
                continue
            graph.graph[key] = value
        return graph

    def _ensure_graph_corpus(self, dataset_name: str | None = None, *, force: bool = False) -> dict:
        paths = self.resolve_paths(dataset_name)
        self._cache_root(paths.dataset_name).mkdir(parents=True, exist_ok=True)
        return build_graph_corpus(
            self.root,
            paths.csv_path,
            graph_builder=lambda row, row_index: self._graph_from_row(
                row,
                dataset_name=paths.dataset_name,
                row_index=row_index,
            ),
            force=force,
        )

    def load(
        self,
        dataset_name: str | None = None,
        *,
        limit: int | None = None,
        min_node_count: int | None = None,
        max_node_count: int | None = None,
    ) -> tuple[list[nx.Graph], pd.DataFrame]:
        paths = self.resolve_paths(dataset_name)
        self._ensure_graph_corpus(paths.dataset_name)
        return load_graph_dataset(
            self._cache_root(paths.dataset_name),
            max_molecules=limit,
            min_node_count=min_node_count,
            max_node_count=max_node_count,
        )


class MolecularGraphSourceLoader:
    """Stream molecular graphs directly from raw source files.

    This loader is intentionally lightweight and complements the cached-corpus
    dataset loaders above. It is meant for single-pass or bounded-memory
    workflows such as online feature extraction or streaming model training.
    """

    _SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "Smiles")
    _SMILES_CSV_TYPES = frozenset({"smiles_csv", "csv_smiles", "zinc_csv"})

    def __init__(self, *, on_error: str = "skip", chunksize: int = 1024) -> None:
        self.on_error = str(on_error)
        self.chunksize = int(chunksize)
        if self.chunksize < 1:
            raise ValueError("chunksize must be >= 1")
        if self.on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be 'skip' or 'raise'")

    @staticmethod
    def _make_rng(random_state=None):
        if random_state is None:
            return np.random.default_rng()
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)

    @staticmethod
    def _normalize_limit(limit):
        if limit is None:
            return None
        if isinstance(limit, (int, np.integer)):
            if int(limit) < 0:
                raise ValueError("limit must be >= 0 when provided as an integer.")
            return int(limit)
        if isinstance(limit, float):
            if not 0.0 < float(limit) < 1.0:
                raise ValueError("float limit must be strictly between 0 and 1.")
            return float(limit)
        raise TypeError("limit must be None, int, or float.")

    @classmethod
    def _resolve_smiles_column(cls, columns) -> str:
        for column in cls._SMILES_COLUMN_CANDIDATES:
            if column in columns:
                return column
        raise ValueError(
            "SMILES CSV reader requires a smiles column named one of: "
            f"{', '.join(cls._SMILES_COLUMN_CANDIDATES)}."
        )

    @staticmethod
    def _coerce_metadata_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _graph_from_smiles_row(self, row: dict, *, dataset_name: str, row_index: int) -> nx.Graph | None:
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
        if graph is None:
            return None
        graph = normalize_graph_schema(graph, copy=False)
        graph.graph["source"] = dataset_name
        graph.graph["input"] = f"{dataset_name}[{row_index}]"
        graph.graph["smiles"] = str(smiles)
        graph.graph["name"] = f"{dataset_name}:{row_index}"
        for key, value in row.items():
            normalized = self._coerce_metadata_value(value)
            if normalized is None:
                continue
            graph.graph[key] = normalized
        return graph

    def _iter_smiles_csv_graphs(self, uri) -> Iterator[nx.Graph]:
        csv_path = Path(uri).expanduser().resolve()
        dataset_name = csv_path.stem
        row_index = 0
        for chunk in pd.read_csv(csv_path, chunksize=self.chunksize):
            smiles_column = self._resolve_smiles_column(chunk.columns)
            for row in chunk.to_dict(orient="records"):
                if smiles_column != "smiles" and "smiles" not in row:
                    row = dict(row)
                    row["smiles"] = row.get(smiles_column)
                graph = self._graph_from_smiles_row(
                    row,
                    dataset_name=dataset_name,
                    row_index=row_index,
                )
                row_index += 1
                if graph is None:
                    continue
                yield graph

    def iter_graphs(
        self,
        uri,
        source_type: str,
        *,
        limit=None,
        random_state=None,
        start_after_instance: int = 0,
    ) -> Iterator[nx.Graph]:
        normalized_type = str(source_type).strip().lower()
        if normalized_type not in self._SMILES_CSV_TYPES:
            available = ", ".join(sorted(self._SMILES_CSV_TYPES))
            raise ValueError(
                f"Unsupported molecular source type '{source_type}'. Available types: {available}."
            )
        start_after_instance = int(start_after_instance)
        if start_after_instance < 0:
            raise ValueError("start_after_instance must be >= 0")
        normalized_limit = self._normalize_limit(limit)
        rng = self._make_rng(random_state)
        yielded = 0
        for raw_index, graph in enumerate(self._iter_smiles_csv_graphs(uri)):
            if raw_index < start_after_instance:
                continue
            if normalized_limit is None:
                pass
            elif isinstance(normalized_limit, int):
                if yielded >= normalized_limit:
                    break
            else:
                if rng.random() > normalized_limit:
                    continue
            yielded += 1
            yield graph
