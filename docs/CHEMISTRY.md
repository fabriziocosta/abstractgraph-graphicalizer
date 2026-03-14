# Chemistry Graph Schema

`abstractgraph_graphicalizer.chem` converts RDKit molecules, SMILES, `.smi`,
and `.sdf` inputs into labeled `networkx.Graph` objects.

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

## Error handling

Batch helpers accept `on_error="raise"` or `on_error="skip"`.

- `raise`: fail fast with `MoleculeParseError`
- `skip`: drop invalid records and continue

Invalid `on_error` modes raise `ValueError`.

## Round-trip support

`graph_to_rdmol` supports the canonical edge labels above and also accepts the
legacy numeric labels `1`, `2`, `3`, and `4` for compatibility with older
graphs.
