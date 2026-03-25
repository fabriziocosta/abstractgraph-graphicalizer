"""Mode-driven streaming loader for official GDB subsets."""

from __future__ import annotations

import gzip
import json
import pickle
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import networkx as nx
import pandas as pd
import requests

from abstractgraph_graphicalizer.chem.mol_loader import (
    dataset_cache_root,
    default_root,
    graph_corpus_dir,
    load_graph_dataset,
    normalize_cached_path,
    search_roots,
)
from abstractgraph_graphicalizer.chem.molecules import MoleculeParseError, smiles_to_graph


GDB_DOWNLOADS_PAGE = "https://gdb.unibe.ch/downloads/"
DEFAULT_GDB_MODE = "lead_like"
RECOMMENDED_LARGE_GDB_MODE = "50M"


@dataclass(frozen=True)
class GDBModeSpec:
    """Configuration for one user-facing GDB download mode."""

    mode: str
    family: str
    official_url: str
    archive_filename: str
    compression: str
    approx_molecules: int
    approx_compressed_size: str
    description: str


GDB_MODE_SPECS: dict[str, GDBModeSpec] = {
    "lead_like": GDBModeSpec(
        mode="lead_like",
        family="GDB-17",
        official_url="https://zenodo.org/record/5172018/files/GDB17.50000000LL.smi.gz?download=1&ref=gdb.unibe.ch",
        archive_filename="GDB17.50000000LL.smi.gz",
        compression="gz",
        approx_molecules=11_000_000,
        approx_compressed_size="75 MB",
        description="Lead-like GDB-17 subset (100-350 MW and 1-3 clogP). Recommended safe default.",
    ),
    "50M": GDBModeSpec(
        mode="50M",
        family="GDB-17",
        official_url="https://zenodo.org/records/5172018/files/GDB17.50000000.smi.gz?download=1&ref=gdb.unibe.ch",
        archive_filename="GDB17.50000000.smi.gz",
        compression="gz",
        approx_molecules=50_000_000,
        approx_compressed_size="484 MB",
        description="Practical large-scale 50 million molecule GDB-17 subset.",
    ),
    "lead_like_no_small_rings": GDBModeSpec(
        mode="lead_like_no_small_rings",
        family="GDB-17",
        official_url="https://zenodo.org/record/5172018/files/GDB17.50000000LLnoSR.smi.gz?download=1&ref=gdb.unibe.ch",
        archive_filename="GDB17.50000000LLnoSR.smi.gz",
        compression="gz",
        approx_molecules=800_000,
        approx_compressed_size="55 MB",
        description="Lead-like GDB-17 subset without small rings (3-4 ring atoms).",
    ),
    "1M": GDBModeSpec(
        mode="1M",
        family="GDB-13",
        official_url="https://zenodo.org/record/5172018/files/gdb13.1M.freq.ll.smi.gz?download=1&ref=gdb.unibe.ch",
        archive_filename="gdb13.1M.freq.ll.smi.gz",
        compression="gz",
        approx_molecules=1_000_000,
        approx_compressed_size="14.8 MB",
        description="Annotated 1 million molecule GDB-13 random sample.",
    ),
    "gdb13_full": GDBModeSpec(
        mode="gdb13_full",
        family="GDB-13",
        official_url="https://zenodo.org/record/5172018/files/gdb13.tgz?download=1&ref=gdb.unibe.ch",
        archive_filename="gdb13.tgz",
        compression="tar.gz",
        approx_molecules=977_468_314,
        approx_compressed_size="2.6 GB",
        description="Entire GDB-13 archive. Advanced multi-GB download, not recommended as default.",
    ),
}


def available_gdb_modes() -> list[str]:
    """Return the supported user-facing GDB modes."""
    return list(GDB_MODE_SPECS)


def resolve_gdb_mode(mode: str | None = None) -> GDBModeSpec:
    """Resolve a user-supplied GDB mode into a concrete source spec."""
    normalized = DEFAULT_GDB_MODE if mode is None else str(mode).strip()
    if not normalized or normalized == "auto":
        normalized = DEFAULT_GDB_MODE
    spec = GDB_MODE_SPECS.get(normalized)
    if spec is None:
        raise ValueError(
            f"Unsupported GDB mode {mode!r}. Available modes: {', '.join(available_gdb_modes())}"
        )
    return spec


@dataclass(frozen=True)
class GDBDownloadResult:
    """Resolved local archive paths and metadata for one GDB mode."""

    spec: GDBModeSpec
    root: Path
    archive_path: Path
    metadata_path: Path
    extracted_path: Path | None = None
    downloaded: bool = False
    defaulted_mode: bool = False


@dataclass(frozen=True)
class GDBRecord:
    """One SMILES record streamed from a GDB subset archive."""

    smiles: str
    row_index: int
    fields: tuple[str, ...] = ()
    annotation: str | None = None
    source_name: str | None = None


def bundled_gdb_root() -> Path:
    """Return the tracked bundled GDB dataset root."""
    return Path(__file__).resolve().parents[3] / "data" / "GDB"


def local_gdb_root() -> Path:
    """Return the ignored local GDB dataset root."""
    return Path(__file__).resolve().parents[3] / "data-local" / "GDB"


def gdb_search_roots() -> list[Path]:
    """Return candidate GDB roots in preference order."""
    return search_roots(
        env_var="ABSTRACTGRAPH_GDB_ROOT",
        local_root=local_gdb_root,
        bundled_root=bundled_gdb_root,
    )


def default_gdb_root() -> Path:
    """Return the preferred local GDB root."""
    return default_root(
        env_var="ABSTRACTGRAPH_GDB_ROOT",
        local_root=local_gdb_root,
        bundled_root=bundled_gdb_root,
    )


def gdb_dataset_name(mode: str | None = None) -> str:
    """Return the cache dataset name for one GDB mode."""
    spec = resolve_gdb_mode(mode)
    return f"gdb_{spec.mode}"


def gdb_cache_root(root: str | Path, mode: str | None = None) -> Path:
    """Return the graph-corpus cache root for one GDB mode."""
    return dataset_cache_root(Path(root).expanduser().resolve(), gdb_dataset_name(mode))


@dataclass(frozen=True)
class GDBDatasetPaths:
    """Resolved local paths for one GDB mode."""

    mode: str
    spec: GDBModeSpec
    archive_path: Path
    metadata_path: Path
    extracted_path: Path | None
    cache_root: Path


@dataclass(frozen=True)
class GDBDatasetSummary:
    """Static and local summary metadata for one supported GDB mode."""

    mode: str
    family: str
    description: str
    archive_filename: str
    official_url: str
    compression: str
    approx_molecules: int
    approx_compressed_size: str
    archive_path: Path
    metadata_path: Path
    downloaded: bool


def gdb_archive_path(root: str | Path, mode: str | None = None) -> Path:
    """Return the target archive path for one GDB mode."""
    spec = resolve_gdb_mode(mode)
    root_path = Path(root).expanduser().resolve()
    return root_path / spec.archive_filename


def gdb_metadata_path(root: str | Path, mode: str | None = None) -> Path:
    """Return the metadata file path for one GDB mode."""
    archive_path = gdb_archive_path(root, mode)
    return archive_path.parent / f"{archive_path.name}.metadata.json"


def gdb_extracted_path(root: str | Path, mode: str | None = None) -> Path:
    """Return the default extracted-path target for one GDB mode."""
    archive_path = gdb_archive_path(root, mode)
    if archive_path.name.endswith(".tar.gz"):
        return archive_path.parent / archive_path.name[: -len(".tar.gz")]
    if archive_path.suffix == ".tgz":
        return archive_path.parent / archive_path.stem
    if archive_path.suffix == ".gz":
        return archive_path.with_suffix("")
    return archive_path


def _metadata_payload(result: GDBDownloadResult) -> dict:
    payload = asdict(result.spec)
    payload.update(
        {
            "mode": result.spec.mode,
            "root": str(result.root),
            "archive_path": str(result.archive_path),
            "metadata_path": str(result.metadata_path),
            "extracted_path": str(result.extracted_path) if result.extracted_path is not None else None,
            "downloaded": result.downloaded,
            "defaulted_mode": result.defaulted_mode,
            "source_page": GDB_DOWNLOADS_PAGE,
        }
    )
    return payload


def write_gdb_metadata(result: GDBDownloadResult) -> Path:
    """Persist local metadata for one downloaded or resolved GDB mode."""
    result.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.metadata_path.write_text(json.dumps(_metadata_payload(result), indent=2, sort_keys=True) + "\n")
    return result.metadata_path


def read_gdb_metadata(metadata_path: str | Path) -> dict:
    """Load persisted GDB metadata from disk."""
    return json.loads(Path(metadata_path).expanduser().read_text())


def download_gdb_archive(
    root: str | Path,
    mode: str | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> GDBDownloadResult:
    """Download one official GDB archive in streaming mode."""
    defaulted_mode = mode is None or not str(mode).strip() or str(mode).strip() == "auto"
    spec = resolve_gdb_mode(mode)
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    archive_path = root_path / spec.archive_filename
    metadata_path = gdb_metadata_path(root_path, spec.mode)
    extracted_path = gdb_extracted_path(root_path, spec.mode)
    if archive_path.exists() and not force:
        result = GDBDownloadResult(
            spec=spec,
            root=root_path,
            archive_path=archive_path,
            metadata_path=metadata_path,
            extracted_path=extracted_path if extracted_path.exists() else None,
            downloaded=False,
            defaulted_mode=defaulted_mode,
        )
        write_gdb_metadata(result)
        return result

    if verbose:
        selected_mode = f"{DEFAULT_GDB_MODE} (default)" if defaulted_mode else spec.mode
        print(
            "Downloading official GDB subset "
            f"{selected_mode}: {spec.description} "
            f"Approx molecules={spec.approx_molecules:,}, compressed size={spec.approx_compressed_size}."
        )

    response = requests.get(spec.official_url, stream=True, timeout=60)
    response.raise_for_status()
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    try:
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        part_path.replace(archive_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    result = GDBDownloadResult(
        spec=spec,
        root=root_path,
        archive_path=archive_path,
        metadata_path=metadata_path,
        extracted_path=extracted_path if extracted_path.exists() else None,
        downloaded=True,
        defaulted_mode=defaulted_mode,
    )
    write_gdb_metadata(result)
    return result


def ensure_gdb_extracted(
    root: str | Path,
    mode: str | None = None,
    *,
    force: bool = False,
) -> GDBDownloadResult:
    """Extract a downloaded GDB archive when an extracted view is requested."""
    result = download_gdb_archive(root, mode, force=False, verbose=False)
    extracted_path = gdb_extracted_path(result.root, result.spec.mode)
    if extracted_path.exists() and not force:
        updated = GDBDownloadResult(**{**result.__dict__, "extracted_path": extracted_path})
        write_gdb_metadata(updated)
        return updated

    if result.spec.compression == "gz":
        with gzip.open(result.archive_path, "rb") as src, extracted_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    elif result.spec.compression == "tar.gz":
        extracted_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(result.archive_path, "r:gz") as archive:
            archive.extractall(extracted_path)
    else:
        raise ValueError(f"Unsupported GDB compression {result.spec.compression!r}")

    updated = GDBDownloadResult(**{**result.__dict__, "extracted_path": extracted_path})
    write_gdb_metadata(updated)
    return updated


def explain_gdb_mode(mode: str | None = None) -> str:
    """Return a short user-facing explanation for a selected GDB mode."""
    spec = resolve_gdb_mode(mode)
    return (
        f"{spec.mode}: {spec.description} "
        f"(family={spec.family}, molecules~{spec.approx_molecules:,}, size~{spec.approx_compressed_size})"
    )


def invalid_gdb_mode_message(mode: str | None) -> str:
    """Return a clear validation error for an unsupported GDB mode."""
    return f"Unsupported GDB mode {mode!r}. Available modes: {', '.join(available_gdb_modes())}"


def infer_gdb_compression(path: str | Path) -> str | None:
    """Infer the compression mode from a GDB path suffix."""
    path_str = str(path)
    if path_str.endswith(".tar.gz") or path_str.endswith(".tgz"):
        return "tar.gz"
    if path_str.endswith(".gz"):
        return "gz"
    return None


def _iter_text_lines(path: Path, compression: str | None) -> Iterator[tuple[str, str | None]]:
    if compression == "tar.gz":
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                for raw_line in stream:
                    yield raw_line.decode("utf-8"), member.name
    elif compression == "gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield line, path.name
    else:
        with path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                yield line, path.name


def iter_gdb_records(
    path: str | Path,
    *,
    compression: str | None = None,
) -> Iterator[GDBRecord]:
    """Stream GDB SMILES records line by line without loading the full archive."""
    resolved_path = Path(path).expanduser().resolve()
    selected_compression = infer_gdb_compression(resolved_path) if compression is None else compression
    row_index = 0
    for line, source_name in _iter_text_lines(resolved_path, selected_compression):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        smiles = parts[0]
        fields = tuple(parts[1:])
        annotation = " ".join(fields) if fields else None
        yield GDBRecord(
            smiles=smiles,
            row_index=row_index,
            fields=fields,
            annotation=annotation,
            source_name=source_name,
        )
        row_index += 1


def _bucket_chunk_path(cache_root: Path, node_count: int, chunk_index: int) -> Path:
    return graph_corpus_dir(cache_root) / f"graphs_nodes_{int(node_count):03d}_chunk_{int(chunk_index):05d}.pkl"


def build_gdb_graph_corpus(
    dataset_dir: str | Path,
    *,
    mode: str | None = None,
    force: bool = False,
    decompress: bool = False,
    chunk_size: int = 10_000,
    on_error: str = "raise",
    verbose: bool = True,
) -> dict:
    """Build a cached graph corpus for one streamed GDB mode."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    loader = GDBLoader(dataset_dir, on_error=on_error, auto_download=True, verbose=verbose)
    paths = loader.resolve_paths(mode, download=True, decompress=decompress, force=force, verbose=verbose)
    cache_root = paths.cache_root
    corpus_dir = graph_corpus_dir(cache_root)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.pkl"
    if manifest_path.exists() and not force:
        with manifest_path.open("rb") as handle:
            return pickle.load(handle)

    dataset_key = gdb_dataset_name(paths.mode)
    bucket_buffers: dict[int, list[tuple[nx.Graph, dict]]] = {}
    bucket_files: dict[int, list[str]] = {}
    bucket_chunk_indices: dict[int, int] = {}
    total_graphs = 0
    invalid_smiles_count = 0

    def flush_bucket(node_count: int) -> None:
        items = bucket_buffers.get(node_count)
        if not items:
            return
        chunk_index = bucket_chunk_indices.get(node_count, 0)
        bucket_path = _bucket_chunk_path(cache_root, node_count, chunk_index)
        with bucket_path.open("wb") as handle:
            pickle.dump(items, handle)
        bucket_files.setdefault(node_count, []).append(normalize_cached_path(cache_root, bucket_path))
        bucket_chunk_indices[node_count] = chunk_index + 1
        bucket_buffers[node_count] = []

    source_path = paths.extracted_path if decompress and paths.extracted_path is not None else paths.archive_path
    for record in loader.iter_smiles_records(paths.mode, download=True, decompress=decompress, force=False, verbose=False):
        try:
            graph = loader._graph_from_record(record, spec=paths.spec)
        except MoleculeParseError:
            if on_error == "skip":
                invalid_smiles_count += 1
                continue
            raise
        node_count = graph.number_of_nodes()
        row_payload = {
            "smiles": record.smiles,
            "gdb_mode": paths.spec.mode,
            "gdb_family": paths.spec.family,
            "gdb_annotation": record.annotation,
            "gdb_source_name": record.source_name,
            "_molecule_row_index": record.row_index,
        }
        bucket_buffers.setdefault(node_count, []).append((graph, row_payload))
        total_graphs += 1
        if len(bucket_buffers[node_count]) >= chunk_size:
            flush_bucket(node_count)

    for node_count in sorted(bucket_buffers):
        flush_bucket(node_count)

    manifest = {
        "dataset_name": dataset_key,
        "mode": paths.spec.mode,
        "family": paths.spec.family,
        "description": paths.spec.description,
        "official_url": paths.spec.official_url,
        "archive_filename": paths.spec.archive_filename,
        "archive_path": normalize_cached_path(cache_root, paths.archive_path),
        "metadata_path": normalize_cached_path(cache_root, paths.metadata_path),
        "extracted_path": (
            normalize_cached_path(cache_root, paths.extracted_path) if paths.extracted_path is not None else None
        ),
        "reader_path": normalize_cached_path(cache_root, source_path),
        "compression": paths.spec.compression,
        "approx_molecules": paths.spec.approx_molecules,
        "approx_compressed_size": paths.spec.approx_compressed_size,
        "total_graphs": total_graphs,
        "invalid_smiles_count": invalid_smiles_count,
        "node_counts": sorted(bucket_files),
        "bucket_files": bucket_files,
    }
    with manifest_path.open("wb") as handle:
        pickle.dump(manifest, handle)
    return manifest


def load_gdb_graph_dataset(
    dataset_dir: str | Path,
    *,
    mode: str | None = None,
    max_molecules: int | None = 100_000,
    min_node_count: int | None = None,
    max_node_count: int | None = 40,
):
    """Load cached GDB graphs and metadata from a graph corpus directory."""
    target_dir = gdb_cache_root(dataset_dir, mode) if mode is not None else dataset_dir
    return load_graph_dataset(
        target_dir,
        max_molecules=max_molecules,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
    )


class GDBLoader:
    """Load official GDB subsets via explicit user-facing modes."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        on_error: str = "raise",
        auto_download: bool = True,
        default_mode: str = DEFAULT_GDB_MODE,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else self.default_root()
        self.on_error = on_error
        self.auto_download = auto_download
        self.default_mode = resolve_gdb_mode(default_mode).mode
        self.verbose = verbose

    @staticmethod
    def default_root() -> Path:
        return default_gdb_root()

    def available_modes(self) -> list[str]:
        return available_gdb_modes()

    def list_modes(self) -> list[GDBDatasetSummary]:
        summaries: list[GDBDatasetSummary] = []
        for mode_name in self.available_modes():
            spec = resolve_gdb_mode(mode_name)
            archive_path = gdb_archive_path(self.root, spec.mode)
            summaries.append(
                GDBDatasetSummary(
                    mode=spec.mode,
                    family=spec.family,
                    description=spec.description,
                    archive_filename=spec.archive_filename,
                    official_url=spec.official_url,
                    compression=spec.compression,
                    approx_molecules=spec.approx_molecules,
                    approx_compressed_size=spec.approx_compressed_size,
                    archive_path=archive_path,
                    metadata_path=gdb_metadata_path(self.root, spec.mode),
                    downloaded=archive_path.exists(),
                )
            )
        return summaries

    def format_mode_table(self) -> str:
        lines = [
            "{:<28} {:<8} {:>14} {:>10} {:>11}".format("mode", "family", "molecules", "size", "downloaded"),
            "-" * 78,
        ]
        for summary in self.list_modes():
            lines.append(
                "{:<28} {:<8} {:>14} {:>10} {:>11}".format(
                    summary.mode,
                    summary.family,
                    f"{summary.approx_molecules:,}",
                    summary.approx_compressed_size,
                    "yes" if summary.downloaded else "no",
                )
            )
        return "\n".join(lines)

    def describe_mode(self, mode: str | None = None) -> str:
        selected_mode = self.default_mode if mode is None else mode
        return explain_gdb_mode(selected_mode)

    def resolve_paths(
        self,
        mode: str | None = None,
        *,
        download: bool | None = None,
        decompress: bool = False,
        force: bool = False,
        verbose: bool | None = None,
    ) -> GDBDatasetPaths:
        requested_mode = mode
        selected_mode = self.default_mode if mode is None else mode
        spec = resolve_gdb_mode(selected_mode)
        should_download = self.auto_download if download is None else download
        should_verbose = self.verbose if verbose is None else verbose
        archive_path = gdb_archive_path(self.root, spec.mode)
        metadata_path = gdb_metadata_path(self.root, spec.mode)
        extracted_path: Path | None = None
        if archive_path.exists() or should_download:
            result: GDBDownloadResult
            if decompress:
                result = ensure_gdb_extracted(self.root, requested_mode, force=force)
            else:
                result = download_gdb_archive(self.root, requested_mode, force=force, verbose=should_verbose)
            extracted_path = result.extracted_path
            archive_path = result.archive_path
            metadata_path = result.metadata_path
        elif not archive_path.exists():
            raise FileNotFoundError(
                f"GDB mode {spec.mode!r} is not available at {archive_path}. "
                f"Available modes: {', '.join(self.available_modes())}"
            )
        return GDBDatasetPaths(
            mode=spec.mode,
            spec=spec,
            archive_path=archive_path,
            metadata_path=metadata_path,
            extracted_path=extracted_path,
            cache_root=gdb_cache_root(self.root, spec.mode),
        )

    def ensure_mode(self, mode: str | None = None, **kwargs) -> GDBDatasetPaths:
        return self.resolve_paths(mode, **kwargs)

    def iter_smiles_records(
        self,
        mode: str | None = None,
        *,
        download: bool | None = None,
        decompress: bool = False,
        force: bool = False,
        verbose: bool | None = None,
    ) -> Iterator[GDBRecord]:
        paths = self.resolve_paths(
            mode,
            download=download,
            decompress=decompress,
            force=force,
            verbose=verbose,
        )
        source_path = paths.extracted_path if decompress and paths.extracted_path is not None else paths.archive_path
        compression = None if source_path == paths.extracted_path else paths.spec.compression
        return iter_gdb_records(source_path, compression=compression)

    def _graph_from_record(self, record: GDBRecord, *, spec: GDBModeSpec) -> nx.Graph:
        try:
            graph = smiles_to_graph(record.smiles)
        except MoleculeParseError:
            if self.on_error == "skip":
                raise
            raise
        graph.graph["source"] = "gdb"
        graph.graph["input"] = f"{spec.mode}[{record.row_index}]"
        graph.graph["gdb_mode"] = spec.mode
        graph.graph["gdb_family"] = spec.family
        graph.graph["gdb_record_index"] = record.row_index
        if record.annotation is not None:
            graph.graph["gdb_annotation"] = record.annotation
        if record.source_name is not None:
            graph.graph["gdb_source_name"] = record.source_name
        return graph

    def iter_graphs(
        self,
        mode: str | None = None,
        *,
        limit: int | None = None,
        download: bool | None = None,
        decompress: bool = False,
        force: bool = False,
        verbose: bool | None = None,
    ) -> Iterator[nx.Graph]:
        count = 0
        selected_mode = self.default_mode if mode is None else mode
        spec = resolve_gdb_mode(selected_mode)
        for record in self.iter_smiles_records(
            mode,
            download=download,
            decompress=decompress,
            force=force,
            verbose=verbose,
        ):
            try:
                graph = self._graph_from_record(record, spec=spec)
            except MoleculeParseError:
                if self.on_error == "skip":
                    continue
                raise
            yield graph
            count += 1
            if limit is not None and count >= int(limit):
                break

    def load(
        self,
        mode: str | None = None,
        *,
        limit: int | None = None,
        min_node_count: int | None = None,
        max_node_count: int | None = None,
        force: bool = False,
        decompress: bool = False,
        chunk_size: int = 10_000,
        verbose: bool | None = None,
    ) -> tuple[list[nx.Graph], pd.DataFrame]:
        selected_mode = self.default_mode if mode is None else mode
        spec = resolve_gdb_mode(selected_mode)
        build_gdb_graph_corpus(
            self.root,
            mode=mode,
            force=force,
            decompress=decompress,
            chunk_size=chunk_size,
            on_error=self.on_error,
            verbose=self.verbose if verbose is None else verbose,
        )
        return load_gdb_graph_dataset(
            self.root,
            mode=spec.mode,
            max_molecules=limit,
            min_node_count=min_node_count,
            max_node_count=max_node_count,
        )
