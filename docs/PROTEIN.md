# Protein Graphicalizers

The protein backend converts PDBx/mmCIF protein-chain data into labeled
`networkx` protein contact networks.

## Intended Input

Use this backend for:

- local plain or gzipped mmCIF files
- RCSB PDB IDs such as `1ubq`
- PDB chain tokens such as `1ubq.A`
- manifest rows with `sample_id`, `pdb_id`, `label_asym_id`, and `auth_asym_id`
- already parsed `ProteinChainRecord` objects

## Main Entrypoints

- `extract_chain_record`
- `protein_chain_record_to_pcn`
- `download_mmcif`
- `ProteinContactNetworkGraphicalizer`
- `ProteinContactNetworkLoader`
- `ProteinLabelGraphicalizer`
- `label_protein_contact_graph`

## Output Schema

The output is a plain undirected `networkx.Graph` whose nodes represent
C-alpha residues and whose edges represent contacts inside the configured
distance window.

Graph-level metadata includes:

- `source = "protein_contact_network"`
- `sample_id`
- `pdb_id`
- `label_asym_id`
- `auth_asym_id`
- `min_distance`
- `max_distance`
- `edge_distance_key = "distance"`

Node attributes include:

- `label`: residue name
- `residue_name`
- `residue_index`
- `label_seq_id`
- `auth_seq_id`
- `insertion_code`
- `chain_id`
- `auth_chain_id`

Edge attributes include:

- `distance`: measured C-alpha distance
- optional `edge_width` for distance-aware display

Edges intentionally do not persist a `label` attribute. Downstream
representation wrappers should derive generic or distance-bin labels when
needed.

## On-The-Fly Labeling

`ProteinLabelGraphicalizer` copies raw cached PCN graphs and derives the labels
needed by graph vectorizers without mutating the cached graph. It can map
residue labels into reduced amino-acid alphabets such as `hp2`, `charge4`,
`physchem5`, `chem7`, and `dayhoff6`. It also derives edge labels from the
stored C-alpha `distance`; with no threshold every edge is labeled `contact`,
and with one threshold edges become `contact_close` or `contact_far`.

## Example

```python
from abstractgraph_graphicalizer.protein import ProteinContactNetworkLoader

loader = ProteinContactNetworkLoader(
    mmcif_dir="data/raw/mmcif",
    graph_dir="data/processed/graphs",
    return_graphs=True,
)
graphs = loader.load(["1ubq.A"])

labeler = ProteinLabelGraphicalizer(alphabet="physchem5", edge_distance_thresholds=[5.0])
graphs_for_vectorizer = labeler.transform(graphs)
```
