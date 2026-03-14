"""RNA graphicalizers."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import networkx as nx

from abstractgraph_graphicalizer.core import GraphicalizerMixin


RNA_COMPLEMENT_MAP = {"A": "U", "U": "A", "C": "G", "G": "C", "T": "A", "N": "N"}
_SUPPORTED_ON_ERROR = {"raise", "sequence"}


def normalize_rna_sequence(sequence: str) -> str:
    """Uppercase a sequence and normalize DNA thymine to RNA uracil."""
    return sequence.upper().replace("T", "U")


def read_fasta(source: str | Path | Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """Yield `(header, sequence)` pairs from a FASTA path or iterable."""
    if not isinstance(source, (str, Path)):
        for header, sequence in source:
            yield str(header), normalize_rna_sequence(str(sequence))
        return

    header: str | None = None
    chunks: list[str] = []
    for raw_line in Path(source).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, normalize_rna_sequence("".join(chunks))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        yield header, normalize_rna_sequence("".join(chunks))


def _annotate_secondary_structure(graph: nx.Graph, dotbracket: str, *, mode: str = "standard") -> None:
    for node_idx, symbol in zip(graph.nodes(), dotbracket):
        if symbol in "()":
            graph.nodes[node_idx]["secondary_structure"] = "stem"
            graph.nodes[node_idx]["non_stem"] = "S"
            if mode == "non_stem":
                graph.nodes[node_idx]["label"] = "S"
        else:
            graph.nodes[node_idx]["secondary_structure"] = "unpaired"
            graph.nodes[node_idx]["non_stem"] = graph.nodes[node_idx]["label"]


def sequence_dotbracket_to_graph(seq_info: str, seq_struct: str) -> nx.Graph:
    """Convert an RNA sequence and dot-bracket structure into a graph."""
    sequence = normalize_rna_sequence(seq_info)
    if len(sequence) != len(seq_struct):
        raise ValueError("sequence and dot-bracket structure must have the same length")

    graph = nx.Graph()
    lifo: list[int] = []
    for index, (base, symbol) in enumerate(zip(sequence, seq_struct)):
        graph.add_node(index, label=base, position=index)
        if index > 0:
            graph.add_edge(index - 1, index, label="bk", type="backbone", weight=1.0)
        if symbol == "(":
            lifo.append(index)
        elif symbol == ")":
            if not lifo:
                raise ValueError("unbalanced dot-bracket structure")
            paired = lifo.pop()
            graph.add_edge(index, paired, label="bp", type="basepair", weight=1.0)
        elif symbol != ".":
            raise ValueError(f"unsupported dot-bracket symbol {symbol!r}")
    if lifo:
        raise ValueError("unbalanced dot-bracket structure")
    return graph


def seq_struct_to_graph(
    header: str,
    sequence: str,
    dotbracket: str,
    *,
    mode: str = "standard",
) -> nx.Graph:
    """Build an RNA graph from sequence plus dot-bracket structure."""
    graph = sequence_dotbracket_to_graph(sequence, dotbracket)
    graph.graph["source"] = "rna_structure"
    graph.graph["id"] = header
    graph.graph["sequence"] = normalize_rna_sequence(sequence)
    graph.graph["structure"] = dotbracket
    _annotate_secondary_structure(graph, dotbracket, mode=mode)
    return graph


def seq_to_graph(header: str, sequence: str, *, mode: str = "standard") -> nx.Graph:
    """Build an RNA backbone graph with no base-pair structure."""
    structure = "." * len(sequence)
    graph = seq_struct_to_graph(header, sequence, structure, mode=mode)
    graph.graph["source"] = "rna_sequence"
    return graph


def rnafold_wrapper(sequence: str, *, executable: str = "RNAfold", flags: str = "--noPS") -> tuple[str, str]:
    """Run RNAfold and return normalized sequence plus dot-bracket structure."""
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"{executable} is not available on PATH")
    normalized = normalize_rna_sequence(sequence)
    result = subprocess.run(
        [executable, *flags.split()],
        input=f"{normalized}\n",
        capture_output=True,
        check=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("RNAfold output is incomplete")
    seq_info = lines[0]
    seq_struct = lines[1].split()[0]
    return seq_info, seq_struct


def rnafold_to_graphs(
    iterable: Iterable[tuple[str, str]],
    *,
    mode: str = "standard",
    on_error: str = "sequence",
) -> list[nx.Graph]:
    """Fold RNA sequences with RNAfold when available."""
    if on_error not in _SUPPORTED_ON_ERROR:
        raise ValueError(f"Unsupported on_error mode {on_error!r}")

    graphs: list[nx.Graph] = []
    for header, sequence in iterable:
        try:
            _, dotbracket = rnafold_wrapper(sequence)
            graph = seq_struct_to_graph(header, sequence, dotbracket, mode=mode)
            graph.graph["source"] = "rnafold"
        except Exception:
            if on_error == "raise":
                raise
            graph = seq_to_graph(header, sequence, mode=mode)
        graphs.append(graph)
    return graphs


def make_reverse_complement_kmer(kmer: str, complement_map: dict[str, str]) -> str:
    complement = "".join(complement_map.get(base, "N") for base in normalize_rna_sequence(kmer))
    return complement[::-1]


def split_kmers(sequence: str, k: int) -> list[str]:
    normalized = normalize_rna_sequence(sequence)
    return [normalized[i * k : (i + 1) * k] for i in range(len(normalized) // k)]


def find_all_occurrences_of_reverse_complement(
    kmer: str,
    sequence: str,
    complement_map: dict[str, str],
) -> list[int]:
    reverse_complement = make_reverse_complement_kmer(kmer, complement_map)
    return [match.start() for match in re.finditer(reverse_complement, normalize_rna_sequence(sequence))]


def make_offset_reverse_complement_kmer_graph(
    sequence: str,
    *,
    k: int,
    offset: int,
    complement_map: dict[str, str],
) -> nx.Graph:
    normalized = normalize_rna_sequence(sequence)[offset:]
    kmers = split_kmers(normalized, k)
    occurrences = [
        find_all_occurrences_of_reverse_complement(kmer, normalized, complement_map) for kmer in kmers
    ]

    graph = nx.Graph()
    for index, base in enumerate(normalized):
        graph.add_node(index + offset, label=base)
    for index in range(len(normalized) - 1):
        graph.add_edge(index + offset, index + offset + 1, label="bk", weight=1.0)

    for kmer_index, starts in enumerate(occurrences):
        start = kmer_index * k
        for end in starts:
            for j in range(k):
                source = start + j + offset
                target = end + k - j + offset - 1
                if graph.has_node(source) and graph.has_node(target) and target > source + k:
                    graph.add_edge(source, target, label="rc", weight=1.0 - 1.0 / k)
    graph.graph["source"] = "reverse_complement"
    return graph


def make_reverse_complement_kmer_graph(
    sequence: str,
    *,
    k: int,
    complement_map: dict[str, str] | None = None,
) -> nx.Graph:
    complement_map = complement_map or RNA_COMPLEMENT_MAP
    graphs = [
        make_offset_reverse_complement_kmer_graph(
            sequence,
            k=k,
            offset=offset,
            complement_map=complement_map,
        )
        for offset in range(k)
    ]
    out = graphs[0]
    for graph in graphs[1:]:
        out = nx.compose(out, graph)
    return out


def make_reverse_complement_graph(
    sequence: str,
    *,
    min_k: int,
    max_k: int,
    complement_map: dict[str, str] | None = None,
) -> nx.Graph:
    complement_map = complement_map or RNA_COMPLEMENT_MAP
    graphs = [
        make_reverse_complement_kmer_graph(sequence, k=k, complement_map=complement_map)
        for k in range(min_k, max_k + 1)
    ]
    out = graphs[0]
    for graph in graphs[1:]:
        out = nx.compose(out, graph)
    return out


class RNASequenceGraphicalizer(GraphicalizerMixin):
    """Convert `(header, sequence)` pairs into RNA backbone graphs."""

    def __init__(self, *, mode: str = "standard") -> None:
        self.mode = mode

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [seq_to_graph(header, sequence, mode=self.mode) for header, sequence in X]


class RNAFoldGraphicalizer(GraphicalizerMixin):
    """Convert `(header, sequence)` pairs into RNAfold graphs when possible."""

    def __init__(self, *, mode: str = "standard", on_error: str = "sequence") -> None:
        self.mode = mode
        self.on_error = on_error

    def transform(self, X, y=None) -> list[nx.Graph]:
        return rnafold_to_graphs(X, mode=self.mode, on_error=self.on_error)


class SequenceReverseComplementGraphicalizer(GraphicalizerMixin):
    """Build reverse-complement interaction graphs from raw sequences."""

    def __init__(
        self,
        *,
        min_k: int,
        max_k: int,
        complement_map: dict[str, str] | None = None,
    ) -> None:
        self.min_k = min_k
        self.max_k = max_k
        self.complement_map = complement_map or RNA_COMPLEMENT_MAP

    def transform(self, X, y=None) -> list[nx.Graph]:
        return [
            make_reverse_complement_graph(
                sequence,
                min_k=self.min_k,
                max_k=self.max_k,
                complement_map=self.complement_map,
            )
            for sequence in X
        ]
