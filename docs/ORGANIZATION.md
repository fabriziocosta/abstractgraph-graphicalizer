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

Standalone editable install:

```bash
python -m pip install -e .
```

Inside the `abstractgraph-ecosystem` superproject:

```bash
python -m pip install -e repos/abstractgraph-graphicalizer --no-deps
```

Chemistry extras:

```bash
python -m pip install -e '.[chem]'
```

## Dependencies

Runtime dependencies declared in `pyproject.toml`:

- `networkx`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `Pillow`
- `requests`
- `torch`

Optional chemistry dependency:

- `rdkit`

## Caveats

- This package intentionally does not depend on the core `abstractgraph`
  package. It converts raw domains into NetworkX graphs that can then be used by
  the rest of the ecosystem.
- `torch` is a default dependency because attention graphicalizers need tensor
  inputs and model outputs.
- RDKit can be easiest to install from conda-forge. Use the `chem` extra only
  when chemistry graphicalizers are needed.
- Install with `--no-deps` only in a shared ecosystem environment where runtime
  dependencies are already managed.

## Validation

```bash
python scripts/smoke_test.py
```
