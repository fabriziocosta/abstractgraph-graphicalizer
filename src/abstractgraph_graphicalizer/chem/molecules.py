"""RDKit-backed chemistry graphicalizers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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
            label=str(int(bond.GetBondTypeAsDouble())),
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
    return rdmol_to_graph(mol)


def smiles_list_to_graphs(
    smiles_list: Sequence[str],
    *,
    on_error: str = "raise",
) -> list[nx.Graph]:
    """Convert a sequence of SMILES strings to graphs."""
    graphs: list[nx.Graph] = []
    for smiles in smiles_list:
        try:
            graphs.append(smiles_to_graph(smiles))
        except MoleculeParseError:
            if on_error == "skip":
                continue
            raise
    return graphs


def sdf_to_graphs(path: str | Path, *, on_error: str = "raise") -> Iterator[nx.Graph]:
    """Yield graphs from an SDF file."""
    _require_rdkit()
    supplier = Chem.SDMolSupplier(str(path))
    for index, mol in enumerate(supplier):
        if mol is None:
            if on_error == "skip":
                continue
            raise MoleculeParseError("Could not parse SDF record", f"{path}[{index}]")
        yield rdmol_to_graph(mol)


def smi_to_graphs(path: str | Path, *, on_error: str = "raise") -> Iterator[nx.Graph]:
    """Yield graphs from a .smi file."""
    _require_rdkit()
    supplier = Chem.SmilesMolSupplier(str(path), titleLine=False)
    for index, mol in enumerate(supplier):
        if mol is None:
            if on_error == "skip":
                continue
            raise MoleculeParseError("Could not parse SMI record", f"{path}[{index}]")
        yield rdmol_to_graph(mol)


def graph_to_rdmol(graph: nx.Graph):
    """Convert a labeled NetworkX graph back to an RDKit molecule."""
    _require_rdkit()
    mol = Chem.RWMol(Chem.MolFromSmiles(""))
    atom_index: dict[int, int] = {}

    for node, data in graph.nodes(data=True):
        label = data.get("label")
        if not label:
            raise MoleculeParseError("Graph node is missing 'label'", str(node))
        atom_index[node] = mol.AddAtom(Chem.Atom(str(label)))

    for source, target, data in graph.edges(data=True):
        label = str(data.get("label", "1"))
        if label == "1":
            bond_type = Chem.BondType.SINGLE
        elif label == "2":
            bond_type = Chem.BondType.DOUBLE
        elif label == "3":
            bond_type = Chem.BondType.TRIPLE
        elif label == "4":
            bond_type = Chem.BondType.AROMATIC
        else:
            raise MoleculeParseError("Unsupported bond label", label)
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
    nx.draw_networkx(graph, pos=pos, ax=ax, labels=node_labels, node_color="#dde7f0")
    nx.draw_networkx_edge_labels(graph, pos=pos, ax=ax, edge_labels=edge_labels)
    ax.set_axis_off()
    return ax


class MoleculeGraphicalizer(GraphicalizerMixin):
    """Batch molecule graphicalizer for SMILES strings."""

    def __init__(self, *, on_error: str = "raise") -> None:
        self.on_error = on_error

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


class SmilesMolecularGraphicalizer(MoleculeGraphicalizer):
    """Compatibility alias for CoCoGraPE-style naming."""

    pass
