# Graph Graphicalizers

The graph backend contains lightweight converters and enrichers for data that
is already sequence-like, vector-like, or graph-like.

## Intended input

Use this backend when the input already has a clear combinatorial form:

- strings or token sequences
- instance matrices where local neighborhoods should become graphs
- existing graphs that need node annotations or graph products

## Main entrypoints

- `sequence_to_graph`
- `string_to_graph`
- `SequenceGraphicalizer`
- `StringGraphicalizer`
- `mutual_nearest_neighbour_graph`
- `MutualNearestNeighbourGraphicalizer`
- `NearestNeighborVectorGraphicalizer`
- `normalized_laplacian_svd`
- `NormalizedLaplacianSVDGraphGraphicalizer`
- `NodeEmbedderGraphGraphicalizer`
- `product_graph`
- `ProductGraphGraphicalizer`

## Output idea

Depending on the converter, the output is either:

- a path graph induced from an ordered sequence
- a neighborhood graph induced from vector similarity
- an existing graph enriched with new node attributes
- a graph transformed through a Cartesian product construction

The backend is useful when a graph is either directly implied by the data or
should be derived from a local geometric or spectral view of the data.
