"""Helpers for loading local ZINC CSV exports."""

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

ZINC_250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)
DEFAULT_ZINC_TARGET_COLUMNS = ("logP", "qed", "SAS")


def bundled_zinc_root() -> Path:
    """Return the tracked bundled ZINC dataset root."""
    return Path(__file__).resolve().parents[3] / "data" / "ZINC"


def local_zinc_root() -> Path:
    """Return the ignored local ZINC dataset root."""
    return Path(__file__).resolve().parents[3] / "data-local" / "ZINC"


def zinc_search_roots() -> list[Path]:
    """Return candidate ZINC dataset roots in preference order."""
    return search_roots(
        env_var="ABSTRACTGRAPH_ZINC_ROOT",
        local_root=local_zinc_root,
        bundled_root=bundled_zinc_root,
    )


def default_zinc_root() -> Path:
    """Return the preferred local ZINC export directory."""
    return default_root(
        env_var="ABSTRACTGRAPH_ZINC_ROOT",
        local_root=local_zinc_root,
        bundled_root=bundled_zinc_root,
    )


@dataclass(frozen=True)
class ZINCDatasetPaths(GenericCSVDatasetPaths):
    """Resolved file path for one local ZINC CSV export."""


@dataclass(frozen=True)
class ZINCDatasetSummary(GenericCSVDatasetSummary):
    """Summary metadata for one locally available ZINC CSV export."""


class ZINCLoader(CSVMoleculeLoader[ZINCDatasetPaths, ZINCDatasetSummary]):
    """Load local ZINC CSV exports containing `smiles` rows and optional targets."""

    @staticmethod
    def default_root() -> Path:
        return default_zinc_root()

    @property
    def default_dataset_name(self) -> str:
        return "zinc_250k"

    @property
    def graph_source_name(self) -> str:
        return "zinc"

    @property
    def graph_dataset_key(self) -> str:
        return "zinc_dataset"

    def make_paths(self, dataset_name: str, csv_path: Path) -> ZINCDatasetPaths:
        return ZINCDatasetPaths(dataset_name=dataset_name, csv_path=csv_path)

    def make_summary(self, dataset_name: str, csv_path: Path, frame: pd.DataFrame) -> ZINCDatasetSummary:
        return ZINCDatasetSummary(
            dataset_name=dataset_name,
            csv_path=csv_path,
            size_bytes=csv_path.stat().st_size,
            molecule_count=len(frame),
        )


def download_zinc_dataset(
    dataset_dir: str | Path,
    url: str = ZINC_250K_URL,
    filename: str = "zinc_250k.csv",
    force: bool = False,
) -> Path:
    """Download the default ZINC CSV export."""
    return download_dataset(dataset_dir, url=url, filename=filename, force=force)


def build_zinc_graph_corpus(
    dataset_dir: str | Path,
    csv_path: str | Path,
    force: bool = False,
) -> dict:
    """Convert a ZINC CSV table into cached graph buckets grouped by node count."""
    loader = ZINCLoader(dataset_dir, on_error="skip")
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


def load_zinc_graph_dataset(
    dataset_dir: str | Path,
    max_molecules: int | None = 100_000,
    min_node_count: int | None = None,
    max_node_count: int | None = 40,
):
    """Load cached ZINC graphs and metadata from a graph corpus directory."""
    return load_graph_dataset(
        dataset_dir,
        max_molecules=max_molecules,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
    )


def extract_zinc_targets(
    metadata,
    target_columns=DEFAULT_ZINC_TARGET_COLUMNS,
):
    """Extract requested target columns from ZINC metadata."""
    return extract_targets(metadata, target_columns)
