# abstractgraph-graphicalizer

`abstractgraph-graphicalizer` converts raw, structured, or weakly structured
data into labeled `networkx` graphs that can then be used directly or handed
off to the rest of the AbstractGraph ecosystem.

The role of this package is narrower than `abstractgraph` itself. It does not
focus on decomposition operators, hashing, or downstream learning. It focuses
on the earlier step: how to take a molecule, a token sequence, a data matrix,
an RNA structure, or segmented image objects and turn that input into a graph
with meaningful node and edge attributes.

## Ecosystem

This repo is part of the AbstractGraph ecosystem:

- `abstractgraph`
- `abstractgraph-ml`
- `abstractgraph-generative`
- `abstractgraph-graphicalizer`

## Converter Families

### Attention graphicalizers

The attention backend turns token-level numeric inputs into preimage graphs by
learning token embeddings and extracting robust co-clustering structure from
attention patterns. It is meant for inputs such as per-token feature matrices,
embedding sequences, or other array-like instances where each row represents a
token or local part of an example. The main entrypoints are
`AbstractGraphPreprocessor` and `ImageNodeClusterer` in
`abstractgraph_graphicalizer.attention`.

### Chemistry graphicalizers

The chemistry backend turns molecular representations into labeled graphs with
atoms as nodes and bonds as edges. It accepts SMILES strings, iterables of
SMILES, `.smi` files, `.sdf` files, and RDKit molecule objects, and it also
supports conversion back from compatible graphs into RDKit molecules. This is
the right backend for small molecules, cheminformatics preprocessing, and
molecular visualization. The main entrypoints are `smiles_to_graph`,
`sdf_to_graphs`, `graph_to_rdmol`, `draw_molecule`, and `draw_graph` in
`abstractgraph_graphicalizer.chem`. The canonical chemistry schema is
documented in [docs/CHEMISTRY.md](docs/CHEMISTRY.md).

### Graph graphicalizers

The graph backend provides lightweight converters and graph enrichers for data
that is already sequence-like, vector-like, or graph-like. It can build path
graphs from strings or token sequences, local neighborhood graphs from vector
datasets or instance matrices, and annotated graphs from existing graphs using
operations such as normalized-Laplacian embeddings or graph products. This
backend is useful when the input is not raw text or molecules but already has a
clear combinatorial or geometric interpretation. The main entrypoints live in
`abstractgraph_graphicalizer.graph`, including `sequence_to_graph`,
`StringGraphicalizer`, `MutualNearestNeighbourGraphicalizer`,
`NormalizedLaplacianSVDGraphGraphicalizer`, and `ProductGraphGraphicalizer`.

### Data graphicalizers

The data backend turns tabular or matrix-valued numeric data into feature
graphs. It is designed for inputs such as dense data matrices, samples by
feature arrays, or datasets where feature correlations and relative importance
can be used to induce graph structure. Depending on the use case, it can create
feature-dependency graphs directly from a matrix or fit a correlation template
and instantiate sample-specific graphs. The main entrypoints are
`data_matrix_to_feature_graph`, `DataMatrixGraphicalizer`, `data_to_graph`, and
`FeatureCorrelationGraphicalizer` in `abstractgraph_graphicalizer.data`.

### RNA graphicalizers

The RNA backend converts sequence and structure information into graphs whose
nodes are nucleotides and whose edges capture backbone connectivity, base-pair
links, or reverse-complement interactions. It can work from plain RNA/DNA
sequences, FASTA records, explicit dot-bracket structures, or RNAfold output
when `RNAfold` is installed. This backend is intended for biological sequence
and secondary-structure workflows rather than general text processing. The main
entrypoints are `sequence_dotbracket_to_graph`, `seq_struct_to_graph`,
`RNASequenceGraphicalizer`, `RNAFoldGraphicalizer`, and
`SequenceReverseComplementGraphicalizer` in
`abstractgraph_graphicalizer.rna`.

### Image graphicalizers

The image backend builds scene graphs from images plus precomputed segment
descriptions. It is designed for cases where object proposals, masks, or
bounding boxes already exist and the remaining task is to derive a graph of
geometric relations such as overlap, containment, proximity, left-of, and
above. It does not currently own the full detector/classifier stack; instead it
focuses on graph construction, visualization, and loading utilities around
segmented image inputs. The main entrypoints are
`extract_geometric_relations_graph`, `ImageSegmentGraphicalizer`,
`visualize_scene_graph_on_image`, and `load_images` in
`abstractgraph_graphicalizer.image`.

## Package Layout

- `src/abstractgraph_graphicalizer/attention/`
- `src/abstractgraph_graphicalizer/chem/`
- `src/abstractgraph_graphicalizer/core/`
- `src/abstractgraph_graphicalizer/data/`
- `src/abstractgraph_graphicalizer/graph/`
- `src/abstractgraph_graphicalizer/image/`
- `src/abstractgraph_graphicalizer/rna/`
- `src/abstractgraph_graphicalizer/text/`

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
