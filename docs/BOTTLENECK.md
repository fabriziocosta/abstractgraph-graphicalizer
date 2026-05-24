# Graph Interpretation Bottleneck

The bottleneck backend learns sparse reusable graph structure from transformer
or token embeddings. It is the preferred learned graphicalization path for
weakly structured inputs.

## Intended input

Use this backend when each instance is a token-by-feature matrix:

```python
{
    "token_embeddings": Tensor[n_tokens, d_model],
    "tokens": list,
    "input_id": str,
    "metadata": dict,
}
```

Plain NumPy arrays or tensors of shape `(n_tokens, n_features)` are also
accepted by `BottleneckGraphicalizer`.

## Main entrypoints

- `GraphInterpretationBottleneck`
- `BottleneckGraphicalizer`
- `BottleneckOutput`
- `bottleneck_output_to_networkx`

Synthetic benchmark helpers:

- `HiddenAutomaton`
- `sample_hidden_automaton`
- `generate_automaton_sequences`
- `train_tiny_sequence_transformer`
- `extract_sequence_embeddings`
- `evaluate_automaton_recovery`

## Architecture

The model maps dense token embeddings into a finite vocabulary of learnable
global node prototypes. Tokens are softly assigned to prototypes, producing
instance-specific node embeddings with globally reusable node identities.

The backend then:

1. masks a subset of induced graph nodes,
2. predicts sparse directed edges between prototypes,
3. applies residual graph message passing,
4. reconstructs masked node embeddings,
5. reconstructs masked token embeddings only from reconstructed graph nodes.

The objective combines masked node reconstruction, masked token reconstruction,
edge sparsity, edge binarization, and assignment entropy. This encourages the
graph to become a compressed predictive relational structure rather than a
similarity graph.

## Output schema

`BottleneckGraphicalizer.transform(X)` returns NetworkX graphs. By default it
returns `nx.DiGraph`; pass `output_graph="undirected"` to emit `nx.Graph`.

Node attributes include:

- `embedding`
- `node_type`
- `prototype_id`
- `assignment_mass`
- `label`

Edge attributes include:

- `weight`
- `probability`
- `edge_type`
- `label`

Graph attributes include:

- `source="graph_interpretation_bottleneck"`
- `assignments`
- `token_to_nodes`
- `active_prototype_ids`
- optional `tokens`
- optional `input_id`
- optional `metadata`

Only active prototypes are emitted by default. A prototype is active when its
assignment mass exceeds `active_mass_threshold`.

## Minimal usage

```python
import numpy as np
from abstractgraph_graphicalizer.bottleneck import BottleneckGraphicalizer

X = [np.random.randn(12, 16), np.random.randn(10, 16)]

graphicalizer = BottleneckGraphicalizer(
    d_model=64,
    num_prototypes=32,
    n_epochs=5,
)
graphs = graphicalizer.fit_transform(X)
```

For precomputed transformer embeddings, disable the internal linear encoder:

```python
graphicalizer = BottleneckGraphicalizer(
    d_model=768,
    num_prototypes=128,
    use_encoder=False,
)
graphs = graphicalizer.fit_transform(embedding_sequences)
```

## Hidden automaton benchmark

The synthetic benchmark samples a hidden probabilistic finite-state automaton,
generates observable symbol sequences, trains a tiny sequence transformer, then
graphicalizes the resulting token embeddings. Hidden state traces are retained
only for evaluation.

The recovery utilities report diagnostic clustering and edge metrics. These
metrics are meant for experiments and smoke tests; exact automaton recovery is
the research hypothesis, not a guaranteed behavior of every short run.

See `notebooks/examples/example_bottleneck_hidden_automaton.ipynb` for an
executable visualization notebook with side-by-side plots of the true automaton,
the learned prototype graph, and the learned graph collapsed back to inferred
hidden states.
