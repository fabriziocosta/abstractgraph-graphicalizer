# abstractgraph-graphicalizer Organization

This document covers code organization, local setup, validation, and supporting
documentation for `abstractgraph-graphicalizer`.

For the semantic role of this repository, see [../README.md](../README.md).

## Package Layout

- `src/abstractgraph_graphicalizer/attention/`
- `src/abstractgraph_graphicalizer/chem/`
- `src/abstractgraph_graphicalizer/core/`
- `src/abstractgraph_graphicalizer/data/`
- `src/abstractgraph_graphicalizer/graph/`
- `src/abstractgraph_graphicalizer/image/`
- `src/abstractgraph_graphicalizer/protein/`
- `src/abstractgraph_graphicalizer/rna/`
- `src/abstractgraph_graphicalizer/text/`

## Documentation

- [ATTENTION.md](ATTENTION.md)
- [CHEMISTRY.md](CHEMISTRY.md)
- [GRAPH.md](GRAPH.md)
- [DATA.md](DATA.md)
- [RNA.md](RNA.md)
- [PROTEIN.md](PROTEIN.md)
- [IMAGE.md](IMAGE.md)

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
