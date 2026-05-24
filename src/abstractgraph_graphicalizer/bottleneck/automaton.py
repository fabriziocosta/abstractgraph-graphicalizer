"""Synthetic hidden-automaton benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from abstractgraph_graphicalizer.bottleneck.model import TinySequenceTransformer


@dataclass(frozen=True)
class HiddenAutomaton:
    """Probabilistic finite-state automaton with hidden states."""

    transition_probs: np.ndarray
    emission_probs: np.ndarray
    start_probs: np.ndarray
    symbols: tuple[Any, ...]

    @property
    def n_states(self) -> int:
        return int(self.transition_probs.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.emission_probs.shape[1])

    def transition_graph(self, threshold: float = 0.0) -> nx.DiGraph:
        graph = nx.DiGraph()
        for state in range(self.n_states):
            graph.add_node(state, label=state)
        for src in range(self.n_states):
            for dst in range(self.n_states):
                prob = float(self.transition_probs[src, dst])
                if prob > threshold:
                    graph.add_edge(src, dst, probability=prob, weight=prob)
        return graph


def sample_hidden_automaton(
    *,
    n_states: int = 4,
    vocab_size: int = 6,
    transition_concentration: float = 0.3,
    emission_concentration: float = 0.3,
    random_state: int | None = None,
) -> HiddenAutomaton:
    """Sample a hidden probabilistic finite-state automaton."""

    rng = np.random.default_rng(random_state)
    transition_probs = rng.dirichlet(
        np.full(int(n_states), float(transition_concentration)),
        size=int(n_states),
    )
    emission_probs = rng.dirichlet(
        np.full(int(vocab_size), float(emission_concentration)),
        size=int(n_states),
    )
    start_probs = rng.dirichlet(np.ones(int(n_states)))
    symbols = tuple(range(int(vocab_size)))
    return HiddenAutomaton(
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        start_probs=start_probs,
        symbols=symbols,
    )


def generate_automaton_sequences(
    automaton: HiddenAutomaton,
    *,
    n_sequences: int = 32,
    length: int = 24,
    random_state: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Generate observable symbol sequences and hidden state traces."""

    rng = np.random.default_rng(random_state)
    sequences: list[np.ndarray] = []
    hidden_states: list[np.ndarray] = []
    for _ in range(int(n_sequences)):
        state = int(rng.choice(automaton.n_states, p=automaton.start_probs))
        seq: list[int] = []
        states: list[int] = []
        for _step in range(int(length)):
            states.append(state)
            symbol = int(rng.choice(automaton.vocab_size, p=automaton.emission_probs[state]))
            seq.append(symbol)
            state = int(rng.choice(automaton.n_states, p=automaton.transition_probs[state]))
        sequences.append(np.asarray(seq, dtype=int))
        hidden_states.append(np.asarray(states, dtype=int))
    return {"sequences": sequences, "hidden_states": hidden_states}


def train_tiny_sequence_transformer(
    sequences: list[np.ndarray],
    *,
    vocab_size: int,
    d_model: int = 64,
    n_heads: int = 4,
    num_layers: int = 2,
    n_epochs: int = 5,
    lr: float = 1e-3,
    device: str = "auto",
) -> TinySequenceTransformer:
    """Train the tiny sequence transformer with next-token prediction."""

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
    if device in {"cpu", "cuda"}:
        torch_device = torch.device(device)
    model = TinySequenceTransformer(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_heads=int(n_heads),
        num_layers=int(num_layers),
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    for _ in range(int(n_epochs)):
        for sequence in sequences:
            tensor = torch.as_tensor(sequence, dtype=torch.long, device=torch_device)
            if tensor.numel() < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor[:-1].unsqueeze(0))[0]
            loss = F.cross_entropy(logits, tensor[1:])
            loss.backward()
            optimizer.step()
    return model


def extract_sequence_embeddings(
    model: TinySequenceTransformer,
    sequences: list[np.ndarray],
    *,
    device: str = "auto",
) -> list[np.ndarray]:
    """Encode symbol sequences into transformer token embeddings."""

    torch_device = next(model.parameters()).device
    if device in {"cpu", "cuda"}:
        torch_device = torch.device(device)
        model = model.to(torch_device)
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for sequence in sequences:
            tensor = torch.as_tensor(sequence, dtype=torch.long, device=torch_device)
            encoded = model.encode(tensor.unsqueeze(0))[0]
            embeddings.append(encoded.detach().cpu().numpy())
    return embeddings


def _flatten_label_sequences(label_sequences: list[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(label_sequences, np.ndarray) and label_sequences.ndim == 1:
        return label_sequences.astype(int)
    return np.concatenate([np.asarray(x).reshape(-1) for x in label_sequences]).astype(int)


def state_assignment_diagnostics(
    learned_assignments: list[np.ndarray] | np.ndarray,
    hidden_states: list[np.ndarray] | np.ndarray,
) -> dict[str, Any]:
    """Evaluate learned node assignments against hidden automaton states."""

    pred = _flatten_label_sequences(learned_assignments)
    truth = _flatten_label_sequences(hidden_states)
    if pred.shape[0] != truth.shape[0]:
        raise ValueError("learned_assignments and hidden_states must have the same total length")
    if pred.size == 0:
        raise ValueError("Cannot evaluate empty assignments")

    pred_labels = np.unique(pred)
    true_labels = np.unique(truth)
    counts = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    for row, pred_label in enumerate(pred_labels):
        pred_mask = pred == pred_label
        for col, true_label in enumerate(true_labels):
            counts[row, col] = int(np.sum(pred_mask & (truth == true_label)))

    row_ind, col_ind = linear_sum_assignment(-counts)
    matched = int(counts[row_ind, col_ind].sum())
    clustering_accuracy = matched / float(pred.size)
    purity = sum(int(counts[row].max()) for row in range(counts.shape[0])) / float(pred.size)
    majority_state = {
        int(pred_labels[row]): int(true_labels[int(np.argmax(counts[row]))])
        for row in range(counts.shape[0])
    }
    matched_state = {
        int(pred_labels[row]): int(true_labels[col])
        for row, col in zip(row_ind, col_ind)
    }

    prototypes_per_state = {int(state): 0 for state in true_labels}
    prototype_state_entropy: dict[int, float] = {}
    for row, pred_label in enumerate(pred_labels):
        state = majority_state[int(pred_label)]
        prototypes_per_state[state] = prototypes_per_state.get(state, 0) + 1
        row_total = float(counts[row].sum())
        if row_total <= 0.0:
            prototype_state_entropy[int(pred_label)] = 0.0
            continue
        probs = counts[row] / row_total
        nz = probs[probs > 0]
        prototype_state_entropy[int(pred_label)] = float(-(nz * np.log(nz)).sum())

    return {
        "metrics": {
            "clustering_accuracy": float(clustering_accuracy),
            "purity": float(purity),
            "ari": float(adjusted_rand_score(truth, pred)),
            "nmi": float(normalized_mutual_info_score(truth, pred)),
            "n_learned_nodes": float(len(pred_labels)),
            "n_true_states": float(len(true_labels)),
            "mean_prototype_state_entropy": float(np.mean(list(prototype_state_entropy.values()))),
            "max_prototypes_per_state": float(max(prototypes_per_state.values()) if prototypes_per_state else 0),
        },
        "contingency": counts,
        "learned_labels": pred_labels,
        "true_labels": true_labels,
        "majority_state": majority_state,
        "matched_state": matched_state,
        "prototypes_per_state": prototypes_per_state,
        "prototype_state_entropy": prototype_state_entropy,
    }


def transition_graph_from_assignments(
    learned_assignments: list[np.ndarray] | np.ndarray,
    *,
    tokens: list[np.ndarray] | np.ndarray | None = None,
    top_k_per_source: int | None = None,
    min_count: int = 1,
) -> nx.DiGraph:
    """Build a prototype transition graph from consecutive token assignments."""

    sequences = (
        [np.asarray(learned_assignments).reshape(-1)]
        if isinstance(learned_assignments, np.ndarray) and learned_assignments.ndim == 1
        else [np.asarray(x).reshape(-1) for x in learned_assignments]
    )
    token_sequences = None
    if tokens is not None:
        token_sequences = (
            [np.asarray(tokens).reshape(-1)]
            if isinstance(tokens, np.ndarray) and tokens.ndim == 1
            else [np.asarray(x).reshape(-1) for x in tokens]
        )
        if len(token_sequences) != len(sequences):
            raise ValueError("tokens must have the same number of sequences as learned_assignments")
    nodes: set[int] = set()
    edge_counts: dict[tuple[int, int], int] = {}
    edge_symbols: dict[tuple[int, int], dict[Any, int]] = {}
    outgoing_counts: dict[int, int] = {}
    for sequence_idx, sequence in enumerate(sequences):
        values = [int(x) for x in sequence]
        nodes.update(values)
        symbols = None if token_sequences is None else token_sequences[sequence_idx]
        for pos, (src, dst) in enumerate(zip(values[:-1], values[1:])):
            edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1
            outgoing_counts[src] = outgoing_counts.get(src, 0) + 1
            if symbols is not None:
                symbol = symbols[pos + 1].item() if hasattr(symbols[pos + 1], "item") else symbols[pos + 1]
                symbol_counts = edge_symbols.setdefault((src, dst), {})
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    graph = nx.DiGraph()
    for node in sorted(nodes):
        graph.add_node(node, prototype_id=node, label=node)

    outgoing: dict[int, list[tuple[float, int, int]]] = {}
    for (src, dst), count in edge_counts.items():
        if count < int(min_count):
            continue
        probability = count / float(max(1, outgoing_counts.get(src, 0)))
        outgoing.setdefault(src, []).append((probability, dst, count))

    for src, candidates in outgoing.items():
        selected = sorted(candidates, reverse=True)
        if top_k_per_source is not None:
            selected = selected[: int(top_k_per_source)]
        for probability, dst, count in selected:
            symbol_counts = edge_symbols.get((src, dst), {})
            top_symbol = None
            if symbol_counts:
                top_symbol = max(symbol_counts.items(), key=lambda item: item[1])[0]
            graph.add_edge(
                src,
                dst,
                probability=float(probability),
                count=int(count),
                weight=float(probability),
                edge_type="assignment_transition",
                edge_label_counts=dict(symbol_counts),
                top_symbol=top_symbol,
                label="assignment_transition",
            )
    graph.graph["source"] = "bottleneck_assignment_transitions"
    return graph


def aggregate_bottleneck_graphs(
    graphs: list[nx.Graph],
    *,
    min_edge_frequency: float = 0.0,
    top_k_per_source: int | None = None,
) -> nx.DiGraph:
    """Aggregate predicted bottleneck edge graphs over a dataset."""

    n_graphs = max(1, len(graphs))
    nodes: set[int] = set()
    node_mass: dict[int, float] = {}
    node_seen: dict[int, int] = {}
    edge_score: dict[tuple[int, int], float] = {}
    edge_seen: dict[tuple[int, int], int] = {}

    for graph in graphs:
        for _, attrs in graph.nodes(data=True):
            proto = int(attrs.get("prototype_id"))
            nodes.add(proto)
            node_mass[proto] = node_mass.get(proto, 0.0) + float(attrs.get("assignment_mass", 0.0))
            node_seen[proto] = node_seen.get(proto, 0) + 1
        for u, v, attrs in graph.edges(data=True):
            src = int(graph.nodes[u].get("prototype_id", u))
            dst = int(graph.nodes[v].get("prototype_id", v))
            edge = (src, dst)
            edge_score[edge] = edge_score.get(edge, 0.0) + float(
                attrs.get("probability", attrs.get("weight", 1.0))
            )
            edge_seen[edge] = edge_seen.get(edge, 0) + 1

    out = nx.DiGraph()
    for proto in sorted(nodes):
        out.add_node(
            proto,
            prototype_id=proto,
            assignment_mass=node_mass.get(proto, 0.0) / float(max(1, node_seen.get(proto, 0))),
            label=proto,
        )

    outgoing: dict[int, list[tuple[float, int, float, float]]] = {}
    for (src, dst), total in edge_score.items():
        frequency = edge_seen[(src, dst)] / float(n_graphs)
        if frequency < float(min_edge_frequency):
            continue
        probability = total / float(max(1, edge_seen[(src, dst)]))
        score = probability * frequency
        outgoing.setdefault(src, []).append((score, dst, probability, frequency))

    for src, candidates in outgoing.items():
        selected = sorted(candidates, reverse=True)
        if top_k_per_source is not None:
            selected = selected[: int(top_k_per_source)]
        for score, dst, probability, frequency in selected:
            out.add_edge(
                src,
                dst,
                probability=float(probability),
                frequency=float(frequency),
                weight=float(score),
                edge_type="predicted_bottleneck_edge",
                label="predicted_bottleneck_edge",
            )
    out.graph["source"] = "aggregated_graph_interpretation_bottleneck"
    out.graph["graph_kind"] = "aggregated_predicted_bottleneck_edges"
    return out


def collapse_graph_to_states(
    graph: nx.Graph,
    prototype_to_state: dict[int, int],
    n_states: int,
    *,
    top_k_per_source: int | None = None,
) -> nx.DiGraph:
    """Collapse a prototype graph into hidden-state space using a post-hoc mapping."""

    collapsed = nx.DiGraph()
    for state in range(int(n_states)):
        collapsed.add_node(state, label=state)

    edge_scores: dict[tuple[int, int], float] = {}
    edge_counts: dict[tuple[int, int], int] = {}
    for u, v, attrs in graph.edges(data=True):
        src_proto = int(graph.nodes[u].get("prototype_id", u))
        dst_proto = int(graph.nodes[v].get("prototype_id", v))
        if src_proto not in prototype_to_state or dst_proto not in prototype_to_state:
            continue
        src_state = int(prototype_to_state[src_proto])
        dst_state = int(prototype_to_state[dst_proto])
        score = float(attrs.get("probability", attrs.get("weight", 1.0)))
        edge = (src_state, dst_state)
        edge_scores[edge] = edge_scores.get(edge, 0.0) + score
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    outgoing: dict[int, list[tuple[float, int]]] = {}
    for (src, dst), total in edge_scores.items():
        score = total / float(max(1, edge_counts[(src, dst)]))
        outgoing.setdefault(src, []).append((score, dst))

    for src, candidates in outgoing.items():
        selected = sorted(candidates, reverse=True)
        if top_k_per_source is not None:
            selected = selected[: int(top_k_per_source)]
        for score, dst in selected:
            collapsed.add_edge(src, dst, probability=float(score), weight=float(score))
    collapsed.graph["source"] = "collapsed_bottleneck_graph"
    return collapsed


def edge_recovery_diagnostics(
    true_graph: nx.Graph,
    score_graph: nx.Graph,
    *,
    n_states: int | None = None,
    probability_key: str = "probability",
) -> dict[str, float | bool]:
    """Compare a scored recovered graph against a true state-transition graph."""

    if n_states is None:
        nodes = sorted(set(true_graph.nodes()) | set(score_graph.nodes()))
    else:
        nodes = list(range(int(n_states)))
    true_edges = set(true_graph.edges())
    predicted_edges = set(score_graph.edges())
    y_true: list[int] = []
    y_score: list[float] = []
    for src in nodes:
        for dst in nodes:
            y_true.append(1 if (src, dst) in true_edges else 0)
            y_score.append(
                float(score_graph.edges[src, dst].get(probability_key, 1.0))
                if score_graph.has_edge(src, dst)
                else 0.0
            )

    overlap = len(true_edges & predicted_edges)
    if len(set(y_true)) < 2:
        auroc = 0.5
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auroc = 0.5
    if not np.isfinite(auroc):
        auroc = 0.5
    return {
        "edge_precision": overlap / float(max(1, len(predicted_edges))),
        "edge_recall": overlap / float(max(1, len(true_edges))),
        "edge_auroc": auroc,
        "edge_symmetric_difference": float(len(true_edges ^ predicted_edges)),
        "vf2_isomorphic_unlabelled": bool(nx.is_isomorphic(true_graph, score_graph)),
    }


def evaluate_automaton_recovery(
    learned_assignments: list[np.ndarray] | np.ndarray,
    hidden_states: list[np.ndarray] | np.ndarray,
    *,
    learned_graph: nx.Graph | None = None,
    automaton: HiddenAutomaton | None = None,
    transition_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute diagnostic latent-state and edge-recovery metrics."""

    diagnostics = state_assignment_diagnostics(learned_assignments, hidden_states)
    metrics: dict[str, float] = dict(diagnostics["metrics"])

    if learned_graph is not None and automaton is not None:
        collapsed = collapse_graph_to_states(
            learned_graph,
            diagnostics["majority_state"],
            automaton.n_states,
        )
        metrics.update(
            edge_recovery_diagnostics(
                automaton.transition_graph(threshold=transition_threshold),
                collapsed,
                n_states=automaton.n_states,
            )
        )

    return metrics
