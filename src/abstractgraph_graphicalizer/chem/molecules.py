"""RDKit-backed chemistry graphicalizers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import networkx as nx
from PIL import Image

from abstractgraph_graphicalizer.core import GraphicalizerMixin

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception as exc:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Draw = None  # type: ignore[assignment]
    rdMolDraw2D = None  # type: ignore[assignment]
    _RDKIT_IMPORT_ERROR = exc
else:  # pragma: no cover
    _RDKIT_IMPORT_ERROR = None


def _require_rdkit() -> None:
    global Chem, Draw, rdMolDraw2D, _RDKIT_IMPORT_ERROR
    if Chem is None or Draw is None or rdMolDraw2D is None:
        try:
            Chem = importlib.import_module("rdkit.Chem")
            Draw = importlib.import_module("rdkit.Chem.Draw")
            rdMolDraw2D = importlib.import_module("rdkit.Chem.Draw.rdMolDraw2D")
            _RDKIT_IMPORT_ERROR = None
        except Exception as exc:  # pragma: no cover
            _RDKIT_IMPORT_ERROR = exc
    if Chem is None or Draw is None or rdMolDraw2D is None:
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


def graph_to_rdmol(graph: nx.Graph, *, return_index_maps: bool = False):
    """Convert a labeled NetworkX graph back to an RDKit molecule."""
    _require_rdkit()
    graph = normalize_graph_schema(graph, copy=True)
    mol = Chem.RWMol(Chem.MolFromSmiles(""))
    atom_index: dict[int, int] = {}
    bond_index: dict[tuple[int, int], int] = {}

    for node, data in graph.nodes(data=True):
        label = data.get("label")
        if not label:
            raise MoleculeParseError("Graph node is missing 'label'", str(node))
        atom_index[node] = mol.AddAtom(Chem.Atom(str(label)))

    for source, target, data in graph.edges(data=True):
        label = data.get("label", "single")
        bond_type = _bond_type_from_label(label)
        mol.AddBond(atom_index[source], atom_index[target], bond_type)
        rd_bond_idx = mol.GetBondBetweenAtoms(atom_index[source], atom_index[target]).GetIdx()
        bond_index[(source, target)] = rd_bond_idx
        bond_index[(target, source)] = rd_bond_idx

    out = mol.GetMol()
    if return_index_maps:
        return out, atom_index, bond_index
    return out


def _graph_node_scores(graph: nx.Graph, atom_scores: Mapping[object, float] | None) -> dict[object, float]:
    if atom_scores is not None:
        return {node: float(value) for node, value in atom_scores.items()}
    out: dict[object, float] = {}
    for node, data in graph.nodes(data=True):
        if "importance" in data:
            out[node] = float(data["importance"])
    return out


def _graph_bond_scores(
    graph: nx.Graph,
    bond_scores: Mapping[object, float] | None,
    *,
    normalized_atom_scores: Mapping[object, float] | None = None,
) -> dict[tuple[object, object], float]:
    out: dict[tuple[object, object], float] = {}
    if bond_scores is not None:
        for key, value in bond_scores.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            source, target = key
            out[(source, target)] = float(value)
            out[(target, source)] = float(value)
        return out
    if normalized_atom_scores:
        for source, target in graph.edges():
            score = min(
                float(normalized_atom_scores.get(source, 0.0)),
                float(normalized_atom_scores.get(target, 0.0)),
            )
            if score > 0.0:
                out[(source, target)] = score
                out[(target, source)] = score
        return out
    for source, target, data in graph.edges(data=True):
        if "importance" in data:
            out[(source, target)] = float(data["importance"])
            out[(target, source)] = float(data["importance"])
    return out


def _normalize_positive_scores(scores: Mapping[object, float]) -> dict[object, float]:
    positive = {key: max(0.0, float(value)) for key, value in scores.items() if float(value) > 0.0}
    if not positive:
        return {}
    vmax = max(positive.values())
    if vmax <= 0:
        return {}
    return {key: value / vmax for key, value in positive.items()}


def _visualize_scores(
    scores: Mapping[object, float],
    *,
    floor: float = 0.18,
    gamma: float = 0.6,
) -> dict[object, float]:
    floor = float(np.clip(floor, 0.0, 1.0))
    gamma = max(1e-9, float(gamma))
    out: dict[object, float] = {}
    for key, value in scores.items():
        norm_value = float(np.clip(value, 0.0, 1.0))
        out[key] = floor + (1.0 - floor) * float(np.power(norm_value, gamma))
    return out


def _score_to_rgb(score: float, cmap_name: str) -> tuple[float, float, float]:
    rgba = colormaps.get_cmap(cmap_name)(float(np.clip(score, 0.0, 1.0)))
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]))


def draw_molecule(
    molecule,
    *,
    size: tuple[int, int] = (500, 300),
    atom_scores: Mapping[object, float] | None = None,
    bond_scores: Mapping[object, float] | None = None,
    cmap: str = "YlOrRd",
    glow: bool = True,
):
    """Return an RDKit molecule image for an RDKit mol or NetworkX graph.

    When scores are provided, atoms and bonds are highlighted using RDKit's
    highlight rendering. For NetworkX graphs, node/edge ``importance`` fields
    are used by default when explicit score mappings are not passed.
    """
    _require_rdkit()
    atom_highlights: dict[int, tuple[float, float, float]] = {}
    atom_radii: dict[int, float] = {}
    bond_highlights: dict[int, tuple[float, float, float]] = {}

    if isinstance(molecule, nx.Graph):
        mol, atom_index, bond_index = graph_to_rdmol(molecule, return_index_maps=True)
        normalized_atom_scores = _normalize_positive_scores(_graph_node_scores(molecule, atom_scores))
        visual_atom_scores = _visualize_scores(normalized_atom_scores)
        normalized_bond_scores = _graph_bond_scores(
            molecule,
            bond_scores,
            normalized_atom_scores=visual_atom_scores if bond_scores is None else None,
        )
        if bond_scores is not None:
            normalized_bond_scores = _normalize_positive_scores(normalized_bond_scores)
            normalized_bond_scores = _visualize_scores(normalized_bond_scores)
        for node, score in visual_atom_scores.items():
            atom_idx = atom_index.get(node)
            if atom_idx is None:
                continue
            atom_highlights[atom_idx] = _score_to_rgb(score, cmap)
            atom_radii[atom_idx] = 0.14 + (0.22 * score if glow else 0.10 * score)
        for edge, score in normalized_bond_scores.items():
            bond_idx = bond_index.get(edge)
            if bond_idx is None:
                continue
            bond_highlights[bond_idx] = _score_to_rgb(score, cmap)
    else:
        mol = molecule
        normalized_atom_scores = _visualize_scores(_normalize_positive_scores(atom_scores or {}))
        normalized_bond_scores = _visualize_scores(_normalize_positive_scores(bond_scores or {}))
        for atom_idx, score in normalized_atom_scores.items():
            atom_highlights[int(atom_idx)] = _score_to_rgb(score, cmap)
            atom_radii[int(atom_idx)] = 0.14 + (0.22 * score if glow else 0.10 * score)
        for bond_idx, score in normalized_bond_scores.items():
            bond_highlights[int(bond_idx)] = _score_to_rgb(score, cmap)

    if mol is None:
        raise MoleculeParseError("Cannot draw empty molecule")
    if not atom_highlights and not bond_highlights:
        return Draw.MolToImage(mol, size=size)

    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(int(size[0]), int(size[1]))
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    if hasattr(options, "fillHighlights"):
        options.fillHighlights = True
    if hasattr(options, "continuousHighlight"):
        options.continuousHighlight = bool(glow)
    if hasattr(options, "atomHighlightsAreCircles"):
        options.atomHighlightsAreCircles = True
    if hasattr(options, "highlightBondWidthMultiplier"):
        options.highlightBondWidthMultiplier = 16 if glow else 8

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=sorted(atom_highlights.keys()),
        highlightAtomColors=atom_highlights,
        highlightAtomRadii=atom_radii,
        highlightBonds=sorted(bond_highlights.keys()),
        highlightBondColors=bond_highlights,
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


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
