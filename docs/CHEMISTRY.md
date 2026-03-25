# Chemistry Graph Schema

`abstractgraph_graphicalizer.chem` converts RDKit molecules, SMILES, `.smi`,
and `.sdf` inputs into labeled `networkx.Graph` objects.

It also includes maintained dataset helpers for local PubChem assay exports,
local ZINC CSV exports, local QM9 CSV exports, and official GDB subset
archives, including graph-corpus caching and notebook-scale supervised dataset
shaping.

Loader module layout:

- `chem/mol_loader.py`: shared CSV-loader and graph-corpus cache helpers
- `chem/pubchem.py`: PubChem-specific assay loader
- `chem/zinc.py`: ZINC-specific CSV loader
- `chem/qm9.py`: QM9-specific CSV loader
- `chem/gdb.py`: GDB mode definitions, archive download helpers, streaming readers, and loader/cache integration

Conceptual split:

- `PubChemAssayLoader` is assay-centric. It manages a collection of bioassays,
  and each assay is a distinct binary prediction task with `active` and
  `inactive` molecule splits.
- `ZINCLoader` and `QM9Loader` are dataset-centric. They each manage one
  tabular molecular dataset at a time, where targets are metadata columns
  attached to each molecule row rather than assay-defined activity splits.

This distinction is intentional. PubChem is not treated as "just another CSV
dataset" because the primary loading unit is the assay, not the file row.

For bundled roots under `data/<DATASET>/` and ignored roots under
`data-local/<DATASET>/`, cached graph corpora are unified under the shared
sibling directory `data/graph_corpus_cache/` or `data-local/graph_corpus_cache/`
rather than nested separately inside each dataset folder.

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
- `pubchem_aid`: assay identifier for graphs loaded through `PubChemAssayLoader`
- `pubchem_activity`: `active` or `inactive` for graphs loaded through
  `PubChemAssayLoader`
- `target`: binary target label for graphs loaded through `PubChemAssayLoader`
- `zinc_dataset`: dataset identifier for graphs loaded through `ZINCLoader`
- `qm9_dataset`: dataset identifier for graphs loaded through `QM9Loader`
- `gdb_mode`: selected GDB subset mode for graphs loaded through `GDBLoader`
- `gdb_family`: source family for graphs loaded through `GDBLoader`
- any non-null CSV column from a ZINC row is copied onto `graph.graph`
- any non-null CSV column from a QM9 row is copied onto `graph.graph`

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

`PubChemAssayLoader` standardizes the common local export layout:

- `AID<assay_id>_active.sdf`
- `AID<assay_id>_inactive.sdf`

The bundled Git-tracked sample location is:

- `abstractgraph-graphicalizer/data/PUBCHEM/`

For larger local assay collections, the preferred ignored location is:

- `abstractgraph-graphicalizer/data-local/PUBCHEM/`

Resolution order is:

1. the explicit `root=` passed to `PubChemAssayLoader(...)`
2. `ABSTRACTGRAPH_PUBCHEM_ROOT`
3. `abstractgraph-graphicalizer/data-local/PUBCHEM/`
4. `abstractgraph-graphicalizer/data/PUBCHEM/`

The bundled dataset is intentionally a small Git-safe subset for examples and
tests. Larger local assay exports should live in `data-local/PUBCHEM/` or an
external directory pointed to by `ABSTRACTGRAPH_PUBCHEM_ROOT`.

To inspect the currently resolved root and the available assay file sizes:

```python
loader = PubChemAssayLoader()
print(loader.format_assay_table())
```

Typical usage:

```python
from abstractgraph_graphicalizer.chem import PubChemAssayLoader

loader = PubChemAssayLoader()
graphs, targets = loader.load("624249")
```

The important assumption is that one assay corresponds to one supervised binary
task. If you switch from assay `624249` to another assay, you are loading a new
task, not just another shard of the same dataset.

Converted PubChem graphs are cached under:

- `graph_corpus_cache/AID<assay_id>/graph_corpus/`

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

Unlike PubChem, ZINC is treated as one molecular dataset per CSV export. There
is no assay split built into the interface; any downstream target selection or
task definition is derived from metadata columns.

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

This writes the cached corpus under:

- `graph_corpus_cache/<dataset_name>/graph_corpus/`

## QM9 CSV Exports

`QM9Loader` standardizes local QM9 CSV exports such as:

- `qm9.csv`

The bundled Git-tracked sample location is:

- `abstractgraph-graphicalizer/data/QM9/`

For larger local QM9 datasets, the preferred ignored location is:

- `abstractgraph-graphicalizer/data-local/QM9/`

Resolution order is:

1. the explicit `root=` passed to `QM9Loader(...)`
2. `ABSTRACTGRAPH_QM9_ROOT`
3. `abstractgraph-graphicalizer/data-local/QM9/`
4. `abstractgraph-graphicalizer/data/QM9/`

If the default `qm9.csv` is requested and not present, `QM9Loader` can
download it automatically.

Typical usage:

```python
from abstractgraph_graphicalizer.chem import QM9Loader

loader = QM9Loader()
graphs, metadata = loader.load("qm9", limit=256)
```

Like ZINC, QM9 is treated as a single molecular property dataset. The standard
QM9 targets are numeric columns carried in the metadata rather than assay-style
binary activity labels.

To inspect the available local QM9 CSV datasets:

```python
loader = QM9Loader()
print(loader.format_dataset_table())
```

To build and reuse a cached graph corpus bucketed by node count:

```python
from abstractgraph_graphicalizer.chem import (
    build_qm9_graph_corpus,
    download_qm9_dataset,
    extract_qm9_targets,
    load_qm9_graph_dataset,
)

csv_path = download_qm9_dataset("/tmp/qm9")
manifest = build_qm9_graph_corpus("/tmp/qm9", csv_path)
graphs, metadata = load_qm9_graph_dataset("/tmp/qm9", min_node_count=3, max_node_count=12)
targets = extract_qm9_targets(metadata)
```

This writes the cached corpus under:

- `graph_corpus_cache/<dataset_name>/graph_corpus/`

## GDB Official Subsets

`GDBLoader` treats GDB as a family of curated downloadable subsets, not as one
monolithic default dataset.

Supported user-facing modes currently include:

- `lead_like`: safe default and recommended starting point
- `50M`: recommended large-scale experiment setting
- `1M`: smaller GDB-13 sample
- `lead_like_no_small_rings`: smaller filtered lead-like subset
- `gdb13_full`: advanced multi-GB full GDB-13 archive

The official source page is:

- `https://gdb.unibe.ch/downloads/`

Resolution order is:

1. the explicit `root=` passed to `GDBLoader(...)`
2. `ABSTRACTGRAPH_GDB_ROOT`
3. `abstractgraph-graphicalizer/data-local/GDB/`
4. `abstractgraph-graphicalizer/data/GDB/`

Important defaults:

- If no mode is given, `GDBLoader` uses `lead_like`
- It never silently upgrades to a larger mode
- `50M` is a good explicit large-scale option
- `gdb13_full` is supported as an advanced workflow, not as the normal default

Typical usage:

```python
from abstractgraph_graphicalizer.chem import GDBLoader

loader = GDBLoader()
print(loader.format_mode_table())
print(loader.describe_mode("50M"))

graphs, metadata = loader.load("lead_like", limit=256)
```

The loader downloads the official compressed archive, records local metadata
about the selected mode, and streams SMILES line by line when building the
graph corpus.

For direct archive access and streaming record iteration:

```python
from abstractgraph_graphicalizer.chem import download_gdb_archive, iter_gdb_records

result = download_gdb_archive("/tmp/gdb", mode="lead_like")
for record in iter_gdb_records(result.archive_path):
    print(record.smiles)
    break
```

Converted GDB graphs are cached under:

- `graph_corpus_cache/gdb_<mode>/graph_corpus/`

Large modes should be treated as streaming-style workflows. Avoid assuming the
full archive will fit comfortably in memory.

The chemistry package also exports `SupervisedDatasetLoader` and the legacy
camel-case alias `SupervisedDataSetLoader` for notebook-scale resizing,
equalization, and binary-target shaping.
