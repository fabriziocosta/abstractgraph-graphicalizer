# Attention Graphicalizers

This backend is retained for compatibility and lightweight attention-derived
graph extraction. For new learned graphicalization work on transformer/token
embeddings, prefer the graph interpretation bottleneck backend described in
[BOTTLENECK.md](BOTTLENECK.md).

The attention backend converts token-level numeric inputs into labeled
`networkx` graphs by learning token embeddings and extracting robust
co-clustering structure from attention.

## Intended input

Use this backend when each instance is a 2D array-like object of shape
`(n_tokens, n_features)`, or any equivalent token-by-feature representation.
Typical inputs include embedding sequences, patch/token descriptors, or other
pre-segmented local feature matrices.

## Main entrypoints

- `AbstractGraphPreprocessor`
- `ImageNodeClusterer`

## Output idea

The output is a plain `networkx.Graph` whose nodes correspond to tokens or
local parts of the input instance. Node attributes typically include learned
embeddings and optional labels. Edges encode attention-derived structural
relationships rather than explicit symbolic adjacency supplied by the user.

## When to use it

Use this backend when the graph should be induced from learned interactions in
the input representation, rather than read directly from a molecule, sequence,
or explicit adjacency structure.
