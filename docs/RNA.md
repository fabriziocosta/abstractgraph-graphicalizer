# RNA Graphicalizers

The RNA backend converts RNA or RNA-like sequence information into graphs with
nucleotides as nodes and structural relations as edges.

## Intended input

Use this backend for:

- raw RNA or DNA sequences
- FASTA records
- explicit dot-bracket structures
- sequence plus RNAfold-derived secondary structure
- reverse-complement interaction analysis

DNA inputs are normalized into RNA-style sequences by replacing `T` with `U`.

## Main entrypoints

- `sequence_dotbracket_to_graph`
- `seq_struct_to_graph`
- `seq_to_graph`
- `rnafold_to_graphs`
- `make_reverse_complement_graph`
- `read_fasta`
- `RNASequenceGraphicalizer`
- `RNAFoldGraphicalizer`
- `SequenceReverseComplementGraphicalizer`

## Output idea

The output graph uses nucleotide positions as nodes. Backbone adjacency becomes
one edge family, while base-pair or reverse-complement relations become other
edge families with their own labels. Graph-level metadata records the sequence,
structure, and source of the construction when available.

## When to use it

Use this backend when the graph should reflect biological sequence structure or
secondary-structure constraints rather than generic token order alone.
