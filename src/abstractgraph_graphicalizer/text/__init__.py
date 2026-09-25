"""Text graphicalizers."""

from abstractgraph_graphicalizer.text.nlp_dependency import (
    display_dependency,
    render_dependency_displacy,
    sentence_dependency_graph,
)

__all__ = [
    "sentence_dependency_graph",
    "render_dependency_displacy",
    "display_dependency",
]
