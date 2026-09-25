# Text Graphicalizers

The text backend converts spaCy dependency parses into directed NetworkX
graphs. Tokens become nodes, grammatical dependencies become edges, and token
order is retained with `next` edges.

Install the optional spaCy dependency and a language model:

```bash
python -m pip install -e '.[text]'
python -m spacy download en_core_web_sm
```

Create and display a graph:

```python
from abstractgraph_graphicalizer.text import (
    display_dependency,
    sentence_dependency_graph,
)

graph = sentence_dependency_graph("The quick fox jumps.")
display_dependency([graph])
```

The `nlp` parameter can be used to supply an already loaded spaCy pipeline.
`render_dependency_displacy` provides spaCy's HTML/Jupyter dependency view.
