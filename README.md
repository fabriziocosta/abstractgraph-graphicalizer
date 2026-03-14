# abstractgraph-graphicalizer

`abstractgraph-graphicalizer` owns raw-data-to-NetworkX conversion for the
AbstractGraph ecosystem.

Wave 1 includes:

- attention-driven token/preimage induction migrated from `abstractgraph`
- chemistry conversion from SMILES or SDF to labeled `networkx` graphs
- drawing helpers for molecules and graph views
- lightweight sequence and vector-neighborhood graphicalizers ported from CoCoGraPE
- graph annotation and data-matrix graphicalizers ported from CoCoGraPE

Planned future backends include graph, image, RNA, and text graphicalizers.

## Ecosystem

This repo is part of the AbstractGraph ecosystem:

- `abstractgraph`
- `abstractgraph-ml`
- `abstractgraph-generative`
- `abstractgraph-graphicalizer`

## Package layout

- `src/abstractgraph_graphicalizer/attention/`
  attention-driven graph induction and clustering
- `src/abstractgraph_graphicalizer/chem/`
  molecule conversion and drawing helpers
- `src/abstractgraph_graphicalizer/core/`
  shared graphicalizer protocols
- `src/abstractgraph_graphicalizer/data/`
  data-matrix and feature-correlation graphicalizers
- `src/abstractgraph_graphicalizer/graph/`
  sequence, vector-neighborhood, graph-annotation, and product graphicalizers
- `src/abstractgraph_graphicalizer/image/`
  reserved namespace for future image graphicalizers
- `src/abstractgraph_graphicalizer/rna/`
  reserved namespace for future RNA graphicalizers
- `src/abstractgraph_graphicalizer/text/`
  reserved namespace for future text graphicalizers

## Install

Core install:

```bash
python -m pip install -e .
```

Chemistry extras:

```bash
python -m pip install -e '.[chem]'
```

## Chemistry API

Main entrypoints:

- `smiles_to_graph`
- `smiles_list_to_graphs`
- `smi_to_graphs`
- `sdf_to_graphs`
- `rdmol_to_graph`
- `graph_to_rdmol`
- `draw_molecule`
- `draw_graph`
- `MoleculeGraphicalizer`

Canonical chemistry schema:

- node `label`: atomic symbol
- edge `label`: `single`, `double`, `triple`, or `aromatic`
- extra node and edge metadata is documented in
  [docs/CHEMISTRY.md](docs/CHEMISTRY.md)

Batch helpers use `on_error="raise"` by default. Set `on_error="skip"` to
drop invalid records instead.

## Graph API

The next focused CoCoGraPE port adds lightweight graphicalizers for sequence
and vector inputs:

- `sequence_to_graph`
- `string_to_graph`
- `SequenceGraphicalizer`
- `StringGraphicalizer`
- `mutual_nearest_neighbour_graph`
- `MutualNearestNeighbourGraphicalizer`
- `NearestNeighborVectorGraphicalizer`
- `normalized_laplacian_svd`
- `annotate_normalized_laplacian_svd`
- `NormalizedLaplacianSVDGraphGraphicalizer`
- `NodeEmbedderGraphGraphicalizer`
- `product_graph`
- `ProductGraphGraphicalizer`

## Data API

The data backend now includes:

- `data_matrix_to_feature_graph`
- `DataMatrixGraphicalizer`
- `data_to_graph`
- `FeatureCorrelationGraphicalizer`

## Validation

```bash
python scripts/smoke_test.py
```
