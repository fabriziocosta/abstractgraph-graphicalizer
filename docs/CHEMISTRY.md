# Chemistry Graph Schema

`abstractgraph_graphicalizer.chem` converts RDKit molecules, SMILES, `.smi`,
and `.sdf` inputs into labeled `networkx.Graph` objects.

It also includes maintained dataset helpers for local PubChem assay exports and
local ZINC CSV exports, including graph-corpus caching and notebook-scale
supervised dataset shaping.

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
- `zinc_dataset`: dataset identifier for graphs loaded through `ZINCLoader`
- any non-null CSV column from a ZINC row is copied onto `graph.graph`

## Error handling

Batch helpers accept `on_error="raise"` or `on_error="skip"`.

- `raise`: fail fast with `MoleculeParseError`
- `skip`: drop invalid records and continue

Invalid `on_error` modes raise `ValueError`.

## Round-trip support

`graph_to_rdmol` supports the canonical edge labels above and also accepts the
legacy numeric labels `1`, `2`, `3`, and `4` for compatibility with older
graphs. `normalize_graph_schema(...)` upgrades legacy graph edge labels to the
canonical schema in-memory.

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

For notebook-sized shaped datasets:

```python
from abstractgraph_graphicalizer.chem import load_pubchem_graph_dataset

graphs, targets, metadata = load_pubchem_graph_dataset(
    assay_id="624249",
    dataset_size=256,
    max_node_count=40,
    use_equalized=False,
)
```

## ZINC CSV Exports

`ZINCLoader` standardizes local CSV exports such as:

- `zinc_250k.csv`
- `zinc_small.csv`

The bundled Git-tracked sample location is:

- `abstractgraph-graphicalizer/data/ZINC/`

For larger local ZINC datasets, the preferred ignored location is:

- `abstractgraph-graphicalizer/data-local/ZINC/`

Resolution order is:

1. the explicit `root=` passed to `ZINCLoader(...)`
2. `ABSTRACTGRAPH_ZINC_ROOT`
3. `abstractgraph-graphicalizer/data-local/ZINC/`
4. `abstractgraph-graphicalizer/data/ZINC/`

Typical usage:

```python
from abstractgraph_graphicalizer.chem import ZINCLoader

loader = ZINCLoader()
graphs, metadata = loader.load("zinc_250k", limit=128)
```

To inspect the available local CSV datasets:

```python
loader = ZINCLoader()
print(loader.format_dataset_table())
```

To build and reuse a cached graph corpus bucketed by node count:

```python
from abstractgraph_graphicalizer.chem import (
    build_zinc_graph_corpus,
    download_zinc_dataset,
    extract_zinc_targets,
    load_zinc_graph_dataset,
)

csv_path = download_zinc_dataset("/tmp/zinc")
manifest = build_zinc_graph_corpus("/tmp/zinc", csv_path)
graphs, metadata = load_zinc_graph_dataset("/tmp/zinc", min_node_count=10, max_node_count=15)
targets = extract_zinc_targets(metadata)
```

The chemistry package also exports `SupervisedDatasetLoader` and the legacy
camel-case alias `SupervisedDataSetLoader` for notebook-scale resizing,
equalization, and binary-target shaping.
