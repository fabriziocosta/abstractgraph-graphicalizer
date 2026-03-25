"""Helpers for loading local QM9 CSV exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from abstractgraph_graphicalizer.chem.mol_loader import (
    CSVMoleculeLoader,
    GenericCSVDatasetPaths,
    GenericCSVDatasetSummary,
    build_graph_corpus,
    default_root,
    download_dataset,
    extract_targets,
    load_graph_dataset,
    search_roots,
)

QM9_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.csv"
DEFAULT_QM9_TARGET_COLUMNS = (
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
)


def bundled_qm9_root() -> Path:
    """Return the tracked bundled QM9 dataset root."""
    return Path(__file__).resolve().parents[3] / "data" / "QM9"


def local_qm9_root() -> Path:
    """Return the ignored local QM9 dataset root."""
    return Path(__file__).resolve().parents[3] / "data-local" / "QM9"


def qm9_search_roots() -> list[Path]:
    """Return candidate QM9 dataset roots in preference order."""
    return search_roots(
        env_var="ABSTRACTGRAPH_QM9_ROOT",
        local_root=local_qm9_root,
        bundled_root=bundled_qm9_root,
    )


def default_qm9_root() -> Path:
    """Return the preferred local QM9 export directory."""
    return default_root(
        env_var="ABSTRACTGRAPH_QM9_ROOT",
        local_root=local_qm9_root,
        bundled_root=bundled_qm9_root,
    )


@dataclass(frozen=True)
class QM9DatasetPaths(GenericCSVDatasetPaths):
    """Resolved file path for one local QM9 CSV export."""


@dataclass(frozen=True)
class QM9DatasetSummary(GenericCSVDatasetSummary):
    """Summary metadata for one locally available QM9 CSV export."""


class QM9Loader(CSVMoleculeLoader[QM9DatasetPaths, QM9DatasetSummary]):
    """Load local QM9 CSV exports, downloading the default dataset if needed."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        on_error: str = "raise",
        auto_download: bool = True,
    ) -> None:
        super().__init__(root, on_error=on_error)
        self.auto_download = auto_download

    @staticmethod
    def default_root() -> Path:
        return default_qm9_root()

    @property
    def default_dataset_name(self) -> str:
        return "qm9"

    @property
    def graph_source_name(self) -> str:
        return "qm9"

    @property
    def graph_dataset_key(self) -> str:
        return "qm9_dataset"

    def auto_download_missing_default(self) -> bool:
        return self.auto_download

    def ensure_default_dataset(self, *, force: bool = False) -> Path:
        return download_qm9_dataset(self.root, force=force)

    def make_paths(self, dataset_name: str, csv_path: Path) -> QM9DatasetPaths:
        return QM9DatasetPaths(dataset_name=dataset_name, csv_path=csv_path)

    def make_summary(self, dataset_name: str, csv_path: Path, frame: pd.DataFrame) -> QM9DatasetSummary:
        return QM9DatasetSummary(
            dataset_name=dataset_name,
            csv_path=csv_path,
            size_bytes=csv_path.stat().st_size,
            molecule_count=len(frame),
        )


def download_qm9_dataset(
    dataset_dir: str | Path,
    url: str = QM9_URL,
    filename: str = "qm9.csv",
    force: bool = False,
) -> Path:
    """Download the default QM9 CSV export."""
    return download_dataset(dataset_dir, url=url, filename=filename, force=force)


def build_qm9_graph_corpus(
    dataset_dir: str | Path,
    csv_path: str | Path,
    force: bool = False,
) -> dict:
    """Convert a QM9 CSV table into cached graph buckets grouped by node count."""
    loader = QM9Loader(dataset_dir, on_error="skip", auto_download=False)
    dataset_name = Path(csv_path).stem
    return build_graph_corpus(
        dataset_dir,
        csv_path,
        graph_builder=lambda row, row_index: loader._graph_from_row(
            row,
            dataset_name=dataset_name,
            row_index=row_index,
        ),
        force=force,
    )


def load_qm9_graph_dataset(
    dataset_dir: str | Path,
    max_molecules: int | None = 100_000,
    min_node_count: int | None = None,
    max_node_count: int | None = 40,
):
    """Load cached QM9 graphs and metadata from a graph corpus directory."""
    return load_graph_dataset(
        dataset_dir,
        max_molecules=max_molecules,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
    )


def extract_qm9_targets(
    metadata,
    target_columns=DEFAULT_QM9_TARGET_COLUMNS,
):
    """Extract requested target columns from QM9 metadata."""
    return extract_targets(metadata, target_columns)
