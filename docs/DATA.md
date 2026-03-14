# Data Graphicalizers

The data backend converts numeric matrices and tabular feature arrays into
graphs whose nodes represent features and whose edges represent inferred
relationships such as correlation or importance-guided dependency.

## Intended input

Use this backend for:

- dense data matrices
- samples-by-features arrays
- feature tables where correlation structure is meaningful

This backend is about inducing graph structure from statistics of the data
matrix rather than from explicit symbolic or domain-specific structure.

## Main entrypoints

- `data_matrix_to_feature_graph`
- `DataMatrixGraphicalizer`
- `data_to_graph`
- `FeatureCorrelationGraphicalizer`

## Output idea

The output graph usually has one node per feature. Edges connect features that
appear strongly related under the chosen construction. In the correlation-based
templates, the learned template graph can then be instantiated per sample,
attaching sample-specific feature values as node attributes.

## When to use it

Use this backend when the graph should summarize statistical structure in a
matrix-valued dataset, rather than preserve an original graph supplied by the
input domain.
