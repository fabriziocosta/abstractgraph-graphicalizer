"""
Dependency parsing utilities that produce NetworkX graphs.

This module provides a small helper to turn a sentence into a dependency
parse graph using spaCy. Nodes correspond to tokens; edges point from head to
dependent and carry the dependency label.

Usage:
    >>> import networkx as nx
    >>> from abstractgraph_graphicalizer.text import sentence_dependency_graph
    >>> G = sentence_dependency_graph("The quick brown fox jumps over the lazy dog.")
    >>> list(G.nodes(data=True))[:3]
    [(0, {'text': 'The', 'lemma': 'the', 'pos': 'DET', 'tag': 'DT', 'dep': 'det', 'idx': 0, 'char_start': 0, 'char_end': 3}), ...]

Notes:
- Requires spaCy with a pipeline that includes the dependency parser
  (e.g., "en_core_web_sm"). If the model is not installed, you can install it
  with: `python -m spacy download en_core_web_sm`.
- By default, a synthetic ROOT node (-1) is added and connected to each
  sentence root.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import networkx as nx

# Lazy cache of loaded spaCy pipelines to avoid re-loading on each call.
_NLP_CACHE: Dict[str, object] = {}


def _ensure_spacy_model(lang_model: str, auto_install: bool = False):
    """Return a loaded spaCy pipeline for `lang_model`.

    Args:
        lang_model: spaCy model name, e.g. "en_core_web_sm".
        auto_install: If True, attempt to download the model when missing.

    Returns:
        A loaded spaCy `Language` pipeline.

    Raises:
        ImportError: If spaCy is not installed, or the model is missing and
            `auto_install` is False.
    """
    if lang_model in _NLP_CACHE:
        return _NLP_CACHE[lang_model]

    try:
        import spacy  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time env issue
        raise ImportError(
            "spaCy is required for dependency parsing. Install with `pip install spacy`"
        ) from exc

    try:
        nlp = spacy.load(lang_model, disable=())
    except Exception:
        if not auto_install:
            raise ImportError(
                f"spaCy model '{lang_model}' is not installed. "
                f"Install with: python -m spacy download {lang_model}"
            )
        # Try to download then load
        try:
            from spacy.cli import download  # type: ignore

            download(lang_model)
            nlp = spacy.load(lang_model, disable=())
        except Exception as exc:  # pragma: no cover - network/env dependent
            raise ImportError(
                f"Failed to auto-install spaCy model '{lang_model}'. "
                "Install manually with: python -m spacy download "
                f"{lang_model}"
            ) from exc

    _NLP_CACHE[lang_model] = nlp
    return nlp


def sentence_dependency_graph(
    sentence: str,
    nlp: Optional[object] = None,
    *,
    lang_model: str = "en_core_web_sm",
    include_root: bool = True,
    include_punct: bool = True,
    as_undirected: bool = False,
    auto_install_model: bool = False,
    label_separator: str = "|",
) -> nx.Graph:
    """Build a NetworkX graph for the dependency parse of `sentence`.

    Nodes are integer token indices (0..N-1) with attributes:
      - text, lemma, pos, tag, dep, idx, char_start, char_end, label
        where `label` is a convenience string built as
        f"{pos}{label_separator}{text}" (e.g., "NOUN|dog").
    Edges point from head -> dependent with attributes:
      - `dep`: dependency relation label (e.g., "nsubj", "amod", "ROOT").
      - `label`: a copy of `dep` for display utilities that expect 'label'.
    Additionally, sequential token edges (i -> i+1) are added with dep='next'
    to capture linear order; visualization ignores these by default.
    If `include_root` is True, a synthetic node -1 labeled 'ROOT' is added and
    connected to each sentence root with edge dep='ROOT'.

    Args:
        sentence: Input sentence text. Multiple sentences are supported; a
            single graph is returned with one ROOT connection per sentence.
        nlp: Optional preloaded spaCy `Language`. If None, loads `lang_model`.
        lang_model: spaCy model name to load when `nlp` is None.
        include_root: Whether to add a synthetic ROOT node (-1).
        include_punct: Whether to include punctuation tokens.
        as_undirected: If True, return an undirected graph; otherwise DiGraph.
        auto_install_model: If True and the model is missing, attempt to
            auto-download it via `spacy.cli.download`.
        label_separator: Separator used to build the node `label` attribute as
            POS + separator + Token (default: "|").

    Returns:
        A NetworkX `DiGraph` (or `Graph` if `as_undirected=True`).
    """
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("sentence must be a non-empty string")

    if nlp is None:
        nlp = _ensure_spacy_model(lang_model, auto_install=auto_install_model)

    # Defer spacy import typing references to runtime
    try:
        import spacy  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "spaCy is required for dependency parsing. Install with `pip install spacy`"
        ) from exc

    # Build either directed or undirected graph
    G: nx.Graph
    if as_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    doc = nlp(sentence)  # type: ignore[operator]

    # Add tokens as nodes with attributes
    for token in doc:  # type: ignore[attr-defined]
        if not include_punct and token.is_punct:
            continue
        G.add_node(
            int(token.i),
            text=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            idx=int(token.i),
            char_start=int(token.idx),
            char_end=int(token.idx + len(token)),
            label=f"{token.pos_}{label_separator}{token.text}",
        )

    # Add dependency edges head -> child
    for token in doc:  # type: ignore[attr-defined]
        if not include_punct and token.is_punct:
            continue
        head = token.head
        if head is token:  # root token of a sentence
            continue
        if (not include_punct) and head.is_punct:
            # If the head is punctuation and we dropped it, skip the edge; the
            # token will remain as an isolated node unless a higher head exists.
            continue
        if G.has_node(int(head.i)) and G.has_node(int(token.i)):
            dep_lbl = token.dep_
            G.add_edge(int(head.i), int(token.i), dep=dep_lbl, label=dep_lbl)

    # Synthetic ROOT node connected to sentence roots
    if include_root:
        G.add_node(
            -1,
            text="ROOT",
            lemma="ROOT",
            pos="ROOT",
            tag="ROOT",
            dep="ROOT",
            idx=-1,
            label=f"ROOT{label_separator}ROOT",
        )
        for sent in doc.sents:  # type: ignore[attr-defined]
            root = sent.root
            if (not include_punct) and root.is_punct:
                continue
            if G.has_node(int(root.i)):
                G.add_edge(-1, int(root.i), dep="ROOT", label="ROOT")

    # Add linear order edges (next) between subsequent tokens present in G
    token_nodes = [n for n in sorted(G.nodes()) if isinstance(n, int) and n >= 0]
    token_nodes.sort()
    for i in range(len(token_nodes) - 1):
        u, v = token_nodes[i], token_nodes[i + 1]
        if not G.has_edge(u, v):
            G.add_edge(u, v, dep="next", label="next")

    # Attach graph-level metadata
    G.graph["text"] = sentence
    G.graph["lang_model"] = lang_model if nlp is None else getattr(nlp, "meta", {}).get("name", lang_model)  # type: ignore
    G.graph["directed"] = not as_undirected
    return G


__all__ = [
    "sentence_dependency_graph",
    "display_dependency",            # grid display for NetworkX graphs
    "render_dependency_displacy",    # HTML/Jupyter displaCy helper
]


def render_dependency_displacy(
    text: str,
    nlp: Optional[object] = None,
    *,
    lang_model: str = "en_core_web_sm",
    auto_install_model: bool = False,
    compact: bool = True,
    distance: int = 200,
    color: str = "#000000",
    bg: str = "#ffffff",
    jupyter: Optional[bool] = None,
    return_html: bool = False,
    options: Optional[dict] = None,
) -> Optional[str]:
    """Render the spaCy dependency parse of `text` using displaCy.

    This helper loads a spaCy model (with optional auto-install), parses the
    input, and renders the dependency arcs. In a notebook, it displays inline;
    otherwise you can request the HTML string.

    Args:
        text: Input text to parse and visualize.
        nlp: Optional preloaded spaCy `Language`. If None, loads `lang_model`.
        lang_model: spaCy model name to load when `nlp` is None.
        auto_install_model: If True and the model is missing, attempt to
            auto-download it via `spacy.cli.download`.
        compact: Use a compact layout (reduced spacing between words/arcs).
        distance: Base distance between words, in pixels.
        color: Text/arc color (CSS color string).
        bg: Background color (CSS color string).
        jupyter: Force Jupyter rendering behavior. If None, inferred from
            `return_html` (HTML when True, Jupyter when False).
        return_html: If True, return the HTML string instead of displaying.
        options: Extra displaCy options to merge with defaults.

    Returns:
        HTML string if `return_html=True`, otherwise None.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    if nlp is None:
        nlp = _ensure_spacy_model(lang_model, auto_install=auto_install_model)

    try:
        from spacy import displacy  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "spaCy is required for dependency visualization. Install with `pip install spacy`"
        ) from exc

    doc = nlp(text)  # type: ignore[operator]

    # Default displaCy options; caller can override via `options`.
    displacy_options = {
        "compact": bool(compact),
        "distance": int(distance),
        "color": str(color),
        "bg": str(bg),
    }
    if options:
        displacy_options.update(options)

    # Decide rendering mode.
    if jupyter is None:
        jupyter = not return_html

    if return_html:
        return displacy.render(doc, style="dep", jupyter=False, options=displacy_options)
    else:
        # In notebooks, this displays inline. In plain terminals, it has no effect.
        displacy.render(doc, style="dep", jupyter=bool(jupyter), options=displacy_options)
        return None


def _linear_token_layout(G: nx.Graph, *, spacing: float = 1.6):
    """Compute a 1D layout for dependency graphs using token order.

    - Places tokens along the x-axis by their `idx` (or node key) at y=0.
    - Places ROOT (-1) above the center if present.

    Args:
        G: Dependency graph from `sentence_dependency_graph`.

    Returns:
        Dict[node, (x, y)]: Position mapping for drawing.
    """
    # Identify token nodes (exclude synthetic root -1 if present)
    tokens = [n for n in G.nodes if n != -1]
    # Sort by explicit idx if present; else by node id
    def sort_key(n):
        d = G.nodes[n]
        return d.get("idx", n)

    tokens.sort(key=sort_key)
    pos = {}
    for i, n in enumerate(tokens):
        pos[n] = (i * float(spacing), 0.0)

    if G.has_node(-1):
        if tokens:
            mid_x = (pos[tokens[0]][0] + pos[tokens[-1]][0]) / 2.0
        else:
            mid_x = 0.0
        pos[-1] = (mid_x, 0.8)
    return pos


def display_dependency(
    graphs,
    n_graphs_per_line: int = 1,
    *,
    show_root: bool = False,
    spacing: float = 2.2,
    text_size: int = 11,
    dep_label_size: int = 9,
    pos_size: int = 9,
    category_size: int = 9,
    arrow_size: int = 12,
    table: bool = True,
    anchor_offset: float = 0.12,
    draw_source_stem: bool = True,
):
    """Display a grid of dependency graphs produced by `sentence_dependency_graph`.

    Draws each graph with tokens as text only (no node markers), arranged
    along a baseline with ample spacing. Curved dependency arcs (head → dependent)
    are drawn above the baseline. When `table=True`, four aligned rows are shown:
      Label (incoming dependency label for each token), Token, POS (fine tag),
      Category (coarse POS). This mimics the provided example figure.

    The semicircle endpoints are placed on a horizontal anchor line above the
    label row at `y = y_label + anchor_offset` (or above the token baseline if
    `table=False`). A short vertical stem is drawn at the source (optional) and
    a short arrow is drawn at the target, pointing to the token center.

    Args:
        graphs: Iterable of NetworkX graphs (DiGraph) from `sentence_dependency_graph`.
        n_graphs_per_line: Number of columns in the output grid.

    Returns:
        The created Matplotlib Figure.
    """
    import math
    from matplotlib import pyplot as plt
    import networkx as nx  # re-import for type hints in notebooks

    graphs = list(graphs)
    n = len(graphs)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title("No graphs to display")
        plt.show()
        return fig

    cols = max(1, int(n_graphs_per_line))
    rows = math.ceil(n / cols)
    figsize = (cols * 6.5, rows * 3.2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Normalize axes to a list
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    for i in range(rows * cols):
        ax = axes_list[i]
        if i >= n:
            ax.axis("off")
            continue
        G = graphs[i]
        ax.axis("off")
        # Title: sentence text if available
        title = G.graph.get("text", None)
        if title:
            # Truncate long titles for readability
            t = title if len(title) <= 120 else title[:117] + "..."
            ax.set_title(t, fontsize=9)

        # Layout with wider spacing
        pos = _linear_token_layout(G, spacing=spacing)

        # Ordered tokens and attributes
        tokens = [n for n in G.nodes if n != -1]
        tokens.sort(key=lambda n: G.nodes[n].get("idx", n))
        words = [G.nodes[n].get("text", str(n)) for n in tokens]
        tags = [G.nodes[n].get("tag", G.nodes[n].get("pos", "")) for n in tokens]
        coarse = [G.nodes[n].get("pos", "") for n in tokens]

        # Incoming dependency label for each token (edge head->token). Ignore 'next'.
        incoming_dep = []
        for v in tokens:
            dep_label = ""
            for u in G.predecessors(v):
                ed = G.get_edge_data(u, v, default={})
                d = ed.get("dep") if isinstance(ed, dict) else None
                if d and d != "next":
                    dep_label = d
                    if dep_label != "ROOT":
                        break
            if not dep_label and G.has_node(-1) and G.has_edge(-1, v):
                dep_label = G.edges[-1, v].get("dep", "ROOT")
            incoming_dep.append(dep_label)

        # Row y-positions
        y_token = 0.0
        y_label = 0.45 if table else 0.30
        y_pos = -0.35
        y_coarse = -0.70

        # Draw left-hand headers
        if table and tokens:
            xs = [pos[t][0] for t in tokens]
            x_left = min(xs) - spacing * 0.8
            hdr = dict(ha="right", va="center", fontsize=pos_size)
            ax.text(x_left, y_label, "Label", **hdr)
            ax.text(x_left, y_token, "Token", **hdr)
            ax.text(x_left, y_pos, "POS", **hdr)
            ax.text(x_left, y_coarse, "Category", **hdr)

        # Draw per-token columns
        for i, n_tok in enumerate(tokens):
            x = pos[n_tok][0]
            if table and incoming_dep[i]:
                ax.text(x, y_label, incoming_dep[i], ha="center", va="bottom", fontsize=dep_label_size, color="#555555")
            # Emphasize content words
            is_content = G.nodes[n_tok].get("pos", "") in {"NOUN", "VERB", "PROPN", "ADJ", "AUX"}
            ax.text(x, y_token, words[i], ha="center", va="center", fontsize=text_size, fontweight=("bold" if is_content else "normal"))
            if table:
                ax.text(x, y_pos, tags[i], ha="center", va="center", fontsize=pos_size, color="#444444")
                ax.text(x, y_coarse, coarse[i], ha="center", va="center", fontsize=category_size, color="#444444")

        edge_color = "#555555"

        # Draw edges as semicircles above the baseline
        from matplotlib.patches import Arc
        max_dx = 0.0
        # Anchor line above the label (or baseline if table is False)
        anchor_y = (y_label + float(anchor_offset)) if table else (y_token + max(0.25, float(anchor_offset)))
        edges_iter = list(G.edges(data=True))
        # Optionally hide synthetic ROOT edges
        if not show_root:
            edges_iter = [
                (u, v, d)
                for (u, v, d) in edges_iter
                if not (u == -1 or v == -1 or d.get("dep") == "ROOT")
            ]
        # Always hide linear order 'next' edges in the visualization
        edges_iter = [(u, v, d) for (u, v, d) in edges_iter if d.get("dep") != "next"]

        for u, v, data in edges_iter:
            dep = data.get("dep", "")
            x1 = pos.get(u, (0.0, 0.0))[0]
            x2 = pos.get(v, (0.0, 0.0))[0]
            if x1 == x2:
                continue
            left, right = (x1, x2) if x1 < x2 else (x2, x1)
            width = abs(right - left)
            max_dx = max(max_dx, width)
            center_x = (left + right) / 2.0
            # Semicircle: width == height; draw from 0° to 180° above the anchor line
            arc = Arc((center_x, anchor_y), width=width, height=width, angle=0.0, theta1=0.0, theta2=180.0, color=edge_color, linewidth=1.2)
            ax.add_patch(arc)

            # Optional source vertical stem down to the token line
            if draw_source_stem:
                ax.plot([x1, x1], [anchor_y, y_token], color=edge_color, linewidth=1.0)

            # Arrowhead ends at the anchor line (stops above the token)
            head_len = max(0.08, min(0.14, width * 0.08))
            ax.annotate(
                "",
                xy=(x2, anchor_y),
                xytext=(x2, anchor_y + head_len),
                arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=1.2),
            )

            # If table is disabled, place arc label at the top midpoint as fallback
            if dep and not table:
                mid_x = center_x
                arc_height = anchor_y + (width / 2.0) + 0.05
                ax.text(mid_x, arc_height, str(dep), ha="center", va="bottom", fontsize=dep_label_size, color=edge_color)

        # Set limits to ensure enough space for text and arcs
        if pos:
            xs = [x for x, y in pos.values()]
            if xs:
                xmin, xmax = min(xs), max(xs)
                pad = spacing * 0.8
                ax.set_xlim(xmin - pad, xmax + pad)
        # Y limits: leave room for the tallest semicircle and the label row
        y_top = anchor_y + (max_dx / 2.0) + 0.35
        ax.set_ylim(-1.0, y_top)

    fig.tight_layout()
    plt.show()
    return fig
