# abstractgraph-graphicalizer

`abstractgraph-graphicalizer` owns raw-data-to-NetworkX conversion for the
AbstractGraph ecosystem.

Wave 1 includes:

- attention-driven token/preimage induction migrated from `abstractgraph`
- chemistry conversion from SMILES or SDF to labeled `networkx` graphs
- drawing helpers for molecules and graph views

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
- `src/abstractgraph_graphicalizer/graph/`
  reserved namespace for future graph graphicalizers
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

## Validation

```bash
python scripts/smoke_test.py
```
