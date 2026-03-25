"""Helpers for loading local PubChem assay exports."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from abstractgraph_graphicalizer.chem.mol_loader import (
    SupervisedDataSetLoader,
    SupervisedDatasetLoader,
    dataset_cache_root,
    graph_corpus_dir,
    take_limit,
)
from abstractgraph_graphicalizer.chem.molecules import sdf_to_graphs


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


def _normalize_assay_id(assay_id: str | int) -> str:
    raw = str(assay_id).strip()
    if raw.lower().startswith("aid"):
        raw = raw[3:]
    if not raw:
        raise ValueError("assay_id must not be empty")
    return raw


def _count_sdf_records(path: Path) -> int:
    """Quickly count SDF records by scanning for `$$$$` separators."""
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"$$$$")
    return count


def _pubchem_cache_root(root: Path, assay_id: str) -> Path:
    return dataset_cache_root(root, f"AID{_normalize_assay_id(assay_id)}")


def _pubchem_cache_manifest_path(root: Path, assay_id: str) -> Path:
    return graph_corpus_dir(_pubchem_cache_root(root, assay_id)) / "manifest.pkl"


def _pubchem_cache_payload_path(root: Path, assay_id: str, activity_label: str) -> Path:
    normalized_activity = str(activity_label).strip().lower()
    if normalized_activity not in {"active", "inactive"}:
        raise ValueError("activity_label must be 'active' or 'inactive'")
    return graph_corpus_dir(_pubchem_cache_root(root, assay_id)) / f"{normalized_activity}.pkl"


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


class PubChemAssayLoader:
    """Load local PubChem assay exports split into active/inactive SDF files."""

    def __init__(self, root: str | Path | None = None, *, on_error: str = "raise") -> None:
        self.root = Path(root) if root is not None else default_pubchem_root()
        self.on_error = on_error

    def _cache_root(self, assay_id: str | int) -> Path:
        return _pubchem_cache_root(self.root, _normalize_assay_id(assay_id))

    def available_assay_ids(self) -> list[str]:
        assay_ids = {
            path.name.split("_")[0][3:]
            for path in self.root.glob("AID*_active.sdf")
            if (self.root / f"{path.name.split('_')[0]}_inactive.sdf").exists()
        }
        return sorted(assay_ids)

    def list_assays(self) -> list[PubChemAssaySummary]:
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

    def _build_cached_graphs(
        self,
        path: Path,
        *,
        assay_id: str,
        activity_label: str,
        target: int,
        limit: int | None,
    ) -> list[nx.Graph]:
        return self._annotate_graphs(
            take_limit(sdf_to_graphs(path, on_error=self.on_error), limit),
            assay_id=assay_id,
            activity_label=activity_label,
            target=target,
        )

    def _ensure_graph_corpus(self, assay_id: str | int, *, force: bool = False) -> dict:
        paths = self.resolve_paths(assay_id)
        cache_root = self._cache_root(paths.assay_id)
        corpus_dir = graph_corpus_dir(cache_root)
        corpus_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _pubchem_cache_manifest_path(self.root, paths.assay_id)
        active_payload_path = _pubchem_cache_payload_path(self.root, paths.assay_id, "active")
        inactive_payload_path = _pubchem_cache_payload_path(self.root, paths.assay_id, "inactive")
        if manifest_path.exists() and active_payload_path.exists() and inactive_payload_path.exists() and not force:
            with manifest_path.open("rb") as handle:
                return pickle.load(handle)

        active_graphs = self._build_cached_graphs(
            paths.active_path,
            assay_id=paths.assay_id,
            activity_label="active",
            target=1,
            limit=None,
        )
        inactive_graphs = self._build_cached_graphs(
            paths.inactive_path,
            assay_id=paths.assay_id,
            activity_label="inactive",
            target=0,
            limit=None,
        )
        with active_payload_path.open("wb") as handle:
            pickle.dump(active_graphs, handle)
        with inactive_payload_path.open("wb") as handle:
            pickle.dump(inactive_graphs, handle)
        manifest = {
            "assay_id": paths.assay_id,
            "active_sdf_path": str(paths.active_path.resolve()),
            "inactive_sdf_path": str(paths.inactive_path.resolve()),
            "active_graphs": str(active_payload_path.relative_to(cache_root)),
            "inactive_graphs": str(inactive_payload_path.relative_to(cache_root)),
            "active_count": len(active_graphs),
            "inactive_count": len(inactive_graphs),
        }
        with manifest_path.open("wb") as handle:
            pickle.dump(manifest, handle)
        return manifest

    def _load_cached_split(
        self,
        assay_id: str | int,
        *,
        limit_active: int | None = None,
        limit_inactive: int | None = None,
    ) -> tuple[list[nx.Graph], list[nx.Graph]]:
        normalized_assay_id = _normalize_assay_id(assay_id)
        self._ensure_graph_corpus(normalized_assay_id)
        active_payload_path = _pubchem_cache_payload_path(self.root, normalized_assay_id, "active")
        inactive_payload_path = _pubchem_cache_payload_path(self.root, normalized_assay_id, "inactive")
        with active_payload_path.open("rb") as handle:
            active_graphs = pickle.load(handle)
        with inactive_payload_path.open("rb") as handle:
            inactive_graphs = pickle.load(handle)
        return (
            list(active_graphs[:limit_active]) if limit_active is not None else list(active_graphs),
            list(inactive_graphs[:limit_inactive]) if limit_inactive is not None else list(inactive_graphs),
        )

    def load_split(
        self,
        assay_id: str | int,
        *,
        limit: int | None = None,
        limit_active: int | None = None,
        limit_inactive: int | None = None,
    ) -> tuple[list[nx.Graph], list[nx.Graph]]:
        normalized_assay_id = _normalize_assay_id(assay_id)
        if limit_active is None:
            limit_active = limit
        if limit_inactive is None:
            limit_inactive = limit
        return self._load_cached_split(
            normalized_assay_id,
            limit_active=limit_active,
            limit_inactive=limit_inactive,
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


def load_pubchem_graph_dataset(
    root: str | Path | None = None,
    assay_id: str = "651610",
    dataset_size: int | None = None,
    max_node_count: int | None = None,
    use_equalized: bool = False,
    random_state: int = 0,
) -> tuple[list[nx.Graph], np.ndarray, pd.DataFrame]:
    """Load one PubChem assay with optional shaping and node-count filtering."""
    loader = PubChemAssayLoader(root)

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
