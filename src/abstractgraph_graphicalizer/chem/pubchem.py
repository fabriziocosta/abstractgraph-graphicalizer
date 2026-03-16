"""Helpers for loading local PubChem assay exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import networkx as nx

from abstractgraph_graphicalizer.chem.molecules import sdf_to_graphs


def default_pubchem_root() -> Path:
    """Return the canonical local PubChem export directory."""
    return Path(__file__).resolve().parents[3] / "data" / "PUBCHEM"


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


@dataclass(frozen=True)
class PubChemAssayPaths:
    """Resolved file paths for a local PubChem assay export."""

    assay_id: str
    active_path: Path
    inactive_path: Path


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
