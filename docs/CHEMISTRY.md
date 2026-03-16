# Chemistry Graph Schema

`abstractgraph_graphicalizer.chem` converts RDKit molecules, SMILES, `.smi`,
and `.sdf` inputs into labeled `networkx.Graph` objects.

It also includes a small `PubChemLoader` wrapper for local PubChem assay
exports stored as paired active/inactive SDF files.

## Canonical node attributes

- `label`: atomic symbol string, such as `C`, `O`, or `N`
- `atomic_num`: atomic number
- `formal_charge`: integer formal charge
- `aromatic`: boolean aromaticity flag

## Canonical edge attributes

- `label`: bond label string, one of `single`, `double`, `triple`, `aromatic`
- `bond_order`: numeric bond order as a float
- `bond_type`: original RDKit bond type string
- `aromatic`: boolean aromaticity flag

## Graph-level metadata

When available, graph metadata is attached under `graph.graph`:

- `source`: input family, for example `smiles`, `smi`, or `sdf`
- `input`: original input string or source location
- `pubchem_aid`: assay identifier for graphs loaded through `PubChemLoader`
- `pubchem_activity`: `active` or `inactive` for graphs loaded through
  `PubChemLoader`
- `target`: binary target label for graphs loaded through `PubChemLoader`

## Error handling

Batch helpers accept `on_error="raise"` or `on_error="skip"`.

- `raise`: fail fast with `MoleculeParseError`
- `skip`: drop invalid records and continue

Invalid `on_error` modes raise `ValueError`.

## Round-trip support

`graph_to_rdmol` supports the canonical edge labels above and also accepts the
legacy numeric labels `1`, `2`, `3`, and `4` for compatibility with older
graphs.

## PubChem Assay Exports

`PubChemLoader` standardizes the common local export layout:

- `AID<assay_id>_active.sdf`
- `AID<assay_id>_inactive.sdf`

The bundled Git-tracked sample location is:

- `abstractgraph-graphicalizer/data/PUBCHEM/`

For larger local assay collections, the preferred ignored location is:

- `abstractgraph-graphicalizer/data-local/PUBCHEM/`

Resolution order is:

1. the explicit `root=` passed to `PubChemLoader(...)`
2. `ABSTRACTGRAPH_PUBCHEM_ROOT`
3. `abstractgraph-graphicalizer/data-local/PUBCHEM/`
4. `abstractgraph-graphicalizer/data/PUBCHEM/`

The bundled dataset is intentionally a small Git-safe subset for examples and
tests. Larger local assay exports should live in `data-local/PUBCHEM/` or an
external directory pointed to by `ABSTRACTGRAPH_PUBCHEM_ROOT`.

To inspect the currently resolved root and the available assay file sizes:

```python
loader = PubChemLoader()
print(loader.format_assay_table())
```

Typical usage:

```python
from abstractgraph_graphicalizer.chem import PubChemLoader

loader = PubChemLoader()
graphs, targets = loader.load("624249")
```

If you want the two classes separately:

```python
active_graphs, inactive_graphs = loader.load_split("624249")
```
