"""Helpers for loading local PubChem assay exports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import networkx as nx

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
        limit_active: int | None = None,
        limit_inactive: int | None = None,
    ) -> tuple[list[nx.Graph], list[nx.Graph]]:
        paths = self.resolve_paths(assay_id)
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
        limit_active: int | None = None,
        limit_inactive: int | None = None,
    ) -> tuple[list[nx.Graph], list[int]]:
        active_graphs, inactive_graphs = self.load_split(
            assay_id,
            limit_active=limit_active,
            limit_inactive=limit_inactive,
        )
        graphs = inactive_graphs + active_graphs
        targets = [0] * len(inactive_graphs) + [1] * len(active_graphs)
        return graphs, targets
