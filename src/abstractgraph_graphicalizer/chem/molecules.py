"""RDKit-backed chemistry graphicalizers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from abstractgraph_graphicalizer.core import GraphicalizerMixin

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception as exc:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Draw = None  # type: ignore[assignment]
    _RDKIT_IMPORT_ERROR = exc
else:  # pragma: no cover
    _RDKIT_IMPORT_ERROR = None


def _require_rdkit() -> None:
    if Chem is None:
        raise ImportError(
            "RDKit is required for chemistry graphicalizers. "
            "Install the 'chem' extra for abstractgraph-graphicalizer."
        ) from _RDKIT_IMPORT_ERROR


@dataclass
class MoleculeParseError(ValueError):
    """Raised when a molecule input cannot be parsed."""

    message: str
    source: str | None = None

    def __str__(self) -> str:
        if self.source is None:
            return self.message
        return f"{self.message}: {self.source}"


CHEM_NODE_SCHEMA = {
    "label": "Atomic symbol string, for example 'C' or 'O'.",
    "atomic_num": "Atomic number.",
    "formal_charge": "Formal charge as an integer.",
    "aromatic": "Whether the atom is aromatic.",
}

CHEM_EDGE_SCHEMA = {
    "label": "Bond type label string: 'single', 'double', 'triple', or 'aromatic'.",
    "bond_order": "Numeric bond order as a float.",
    "bond_type": "Original RDKit bond type string.",
    "aromatic": "Whether the bond is aromatic.",
}

_SUPPORTED_ON_ERROR = {"raise", "skip"}
_BOND_LABEL_TO_TYPE = {
    "single": "SINGLE",
    "double": "DOUBLE",
    "triple": "TRIPLE",
    "aromatic": "AROMATIC",
    "1": "SINGLE",
    "2": "DOUBLE",
    "3": "TRIPLE",
    "4": "AROMATIC",
    "singlebond": "SINGLE",
    "doublebond": "DOUBLE",
    "triplebond": "TRIPLE",
    "aromaticbond": "AROMATIC",
    "single bond": "SINGLE",
    "double bond": "DOUBLE",
    "triple bond": "TRIPLE",
    "aromatic bond": "AROMATIC",
}
_LEGACY_BOND_LABEL_MAP = {
    "1": "single",
    "2": "double",
    "3": "triple",
    "4": "aromatic",
    "single": "single",
    "double": "double",
    "triple": "triple",
    "aromatic": "aromatic",
    "singlebond": "single",
    "doublebond": "double",
    "triplebond": "triple",
    "aromaticbond": "aromatic",
    "single bond": "single",
    "double bond": "double",
    "triple bond": "triple",
    "aromatic bond": "aromatic",
    "singlebondtype": "single",
    "doublebondtype": "double",
    "triplebondtype": "triple",
    "aromaticbondtype": "aromatic",
}


def _normalize_on_error(on_error: str) -> str:
    if on_error not in _SUPPORTED_ON_ERROR:
        raise ValueError(
            f"Unsupported on_error mode {on_error!r}. "
            f"Expected one of {sorted(_SUPPORTED_ON_ERROR)}."
        )
    return on_error


def _bond_label_from_rdkit(bond) -> str:
    if bond.GetIsAromatic():
        return "aromatic"
    bond_type = str(bond.GetBondType()).lower()
    return bond_type


def _bond_type_from_label(label: object):
    _require_rdkit()
    normalized = str(label).strip().lower()
    bond_type_name = _BOND_LABEL_TO_TYPE.get(normalized)
    if bond_type_name is None:
        raise MoleculeParseError("Unsupported bond label", str(label))
    return getattr(Chem.BondType, bond_type_name)


def normalize_bond_label(label: object, *, aromatic: bool = False) -> str:
    """Map legacy chemistry edge labels to the canonical graphicalizer schema."""
    if aromatic:
        return "aromatic"
    normalized = str(label).strip()
    if not normalized:
        return "single"
    mapped = _LEGACY_BOND_LABEL_MAP.get(normalized.lower())
    if mapped is not None:
        return mapped
    raise MoleculeParseError("Unsupported bond label", normalized)


def normalize_graph_schema(graph: nx.Graph, *, copy: bool = True) -> nx.Graph:
    """Return a graph with canonical chemistry edge labels and metadata."""
    normalized_graph = graph.copy() if copy else graph
    for _, _, data in normalized_graph.edges(data=True):
        label = normalize_bond_label(data.get("label", "single"), aromatic=bool(data.get("aromatic", False)))
        data["label"] = label
        if "bond_order" not in data:
            data["bond_order"] = {
                "single": 1.0,
                "double": 2.0,
                "triple": 3.0,
                "aromatic": 1.5,
            }[label]
        if "bond_type" not in data:
            data["bond_type"] = _BOND_LABEL_TO_TYPE[label]
        data["aromatic"] = bool(data.get("aromatic", False) or label == "aromatic")
    return normalized_graph


def rdmol_to_graph(mol) -> nx.Graph:
    """Convert an RDKit molecule to a labeled NetworkX graph."""
    _require_rdkit()
    if mol is None:
        raise MoleculeParseError("RDKit molecule is None")

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            label=atom.GetSymbol(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            aromatic=bool(atom.GetIsAromatic()),
        )
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=_bond_label_from_rdkit(bond),
            bond_order=float(bond.GetBondTypeAsDouble()),
            bond_type=str(bond.GetBondType()),
            aromatic=bool(bond.GetIsAromatic()),
        )
    return graph


def smiles_to_graph(smiles: str) -> nx.Graph:
    """Convert a single SMILES string to a labeled NetworkX graph."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeParseError("Could not parse SMILES", smiles)
    graph = rdmol_to_graph(mol)
    graph.graph["source"] = "smiles"
    graph.graph["input"] = smiles
    return graph


def smiles_list_to_graphs(
    smiles_list: Sequence[str],
    *,
    on_error: str = "raise",
) -> list[nx.Graph]:
    """Convert a sequence of SMILES strings to graphs."""
    on_error = _normalize_on_error(on_error)
    graphs: list[nx.Graph] = []
    for index, smiles in enumerate(smiles_list):
        try:
            graphs.append(smiles_to_graph(smiles))
        except MoleculeParseError as exc:
            if on_error == "skip":
                continue
            raise MoleculeParseError(exc.message, exc.source or f"smiles[{index}]") from exc
    return graphs


def sdf_to_graphs(path: str | Path, *, on_error: str = "raise") -> Iterator[nx.Graph]:
    """Yield graphs from an SDF file."""
    _require_rdkit()
    on_error = _normalize_on_error(on_error)
    path = Path(path)
    supplier = Chem.SDMolSupplier(str(path))
    for index, mol in enumerate(supplier):
        if mol is None:
            if on_error == "skip":
                continue
            raise MoleculeParseError("Could not parse SDF record", f"{path}[{index}]")
        graph = rdmol_to_graph(mol)
        graph.graph["source"] = "sdf"
        graph.graph["input"] = f"{path}[{index}]"
        yield graph


def smi_to_graphs(path: str | Path, *, on_error: str = "raise") -> Iterator[nx.Graph]:
    """Yield graphs from a .smi file."""
    _require_rdkit()
    on_error = _normalize_on_error(on_error)
    path = Path(path)
    for index, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        smiles = line.split()[0]
        try:
            graph = smiles_to_graph(smiles)
        except MoleculeParseError as exc:
            if on_error == "skip":
                continue
            raise MoleculeParseError("Could not parse SMI record", f"{path}:{index}") from exc
        graph.graph["source"] = "smi"
        graph.graph["input"] = f"{path}:{index}"
        yield graph


def graph_to_rdmol(graph: nx.Graph):
    """Convert a labeled NetworkX graph back to an RDKit molecule."""
    _require_rdkit()
    graph = normalize_graph_schema(graph, copy=True)
    mol = Chem.RWMol(Chem.MolFromSmiles(""))
    atom_index: dict[int, int] = {}

    for node, data in graph.nodes(data=True):
        label = data.get("label")
        if not label:
            raise MoleculeParseError("Graph node is missing 'label'", str(node))
        atom_index[node] = mol.AddAtom(Chem.Atom(str(label)))

    for source, target, data in graph.edges(data=True):
        label = data.get("label", "single")
        bond_type = _bond_type_from_label(label)
        mol.AddBond(atom_index[source], atom_index[target], bond_type)

    return mol.GetMol()


def draw_molecule(molecule, *, size: tuple[int, int] = (500, 300)):
    """Return an RDKit molecule image for an RDKit mol or NetworkX graph."""
    _require_rdkit()
    mol = graph_to_rdmol(molecule) if isinstance(molecule, nx.Graph) else molecule
    if mol is None:
        raise MoleculeParseError("Cannot draw empty molecule")
    return Draw.MolToImage(mol, size=size)


def draw_graph(graph: nx.Graph, *, ax=None):
    """Draw a labeled NetworkX graph and return the matplotlib axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(graph, seed=0)
    node_labels = {node: data.get("label", node) for node, data in graph.nodes(data=True)}
    edge_labels = {
        (source, target): data.get("label", "")
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        labels=node_labels,
        node_color="#dde7f0",
        edgecolors="#264653",
        linewidths=1.0,
    )
    nx.draw_networkx_edge_labels(graph, pos=pos, ax=ax, edge_labels=edge_labels)
    ax.set_axis_off()
    return ax


def draw_molecules(
    molecules: Sequence[object],
    *,
    n_graphs_per_line: int = 4,
    titles: Sequence[str] | None = None,
    size: tuple[int, int] = (3, 2),
    show: bool = True,
):
    """Draw a grid of RDKit molecules or molecule graphs.

    Args:
        molecules: Sequence of RDKit mols or NetworkX molecule graphs.
        n_graphs_per_line: Number of items per row.
        titles: Optional per-molecule titles.
        size: Size of each subplot in inches.
        show: If True, call ``plt.show()``.

    Returns:
        Matplotlib figure containing the image grid.
    """
    molecules = list(molecules)
    n = len(molecules)
    cols = max(1, int(n_graphs_per_line))
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows))
    axes_list = list(np.atleast_1d(axes).ravel())
    for idx, ax in enumerate(axes_list):
        if idx >= n:
            ax.axis("off")
            continue
        image = draw_molecule(molecules[idx], size=(500, 300))
        ax.imshow(np.asarray(image))
        ax.axis("off")
        if titles is not None and idx < len(titles):
            ax.set_title(str(titles[idx]))
    fig.tight_layout()
    if show:
        plt.show()
    return fig


class MoleculeGraphicalizer(GraphicalizerMixin):
    """Batch molecule graphicalizer for SMILES strings."""

    def __init__(self, *, on_error: str = "raise") -> None:
        self.on_error = _normalize_on_error(on_error)

    def read_sdf(self, path: str | Path) -> list[nx.Graph]:
        return list(sdf_to_graphs(path, on_error=self.on_error))

    def read_smi(self, path: str | Path) -> list[nx.Graph]:
        return list(smi_to_graphs(path, on_error=self.on_error))

    def transform(self, X: Sequence[str], y=None) -> list[nx.Graph]:
        return smiles_list_to_graphs(X, on_error=self.on_error)

    def inverse_transform(self, graphs: Iterable[nx.Graph]) -> list[str]:
        _require_rdkit()
        smiles: list[str] = []
        for graph in graphs:
            smiles.append(Chem.MolToSmiles(graph_to_rdmol(graph)))
        return smiles
