"""Synthetic hidden-automaton benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

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


def evaluate_automaton_recovery(
    learned_assignments: list[np.ndarray] | np.ndarray,
    hidden_states: list[np.ndarray] | np.ndarray,
    *,
    learned_graph: nx.Graph | None = None,
    automaton: HiddenAutomaton | None = None,
    transition_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute diagnostic latent-state and edge-recovery metrics."""

    pred = np.concatenate([np.asarray(x).reshape(-1) for x in learned_assignments]).astype(int)
    truth = np.concatenate([np.asarray(x).reshape(-1) for x in hidden_states]).astype(int)
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

    metrics: dict[str, float] = {
        "clustering_accuracy": float(clustering_accuracy),
        "purity": float(purity),
        "n_learned_nodes": float(len(pred_labels)),
        "n_true_states": float(len(true_labels)),
    }

    if learned_graph is not None and automaton is not None:
        majority_state = {
            int(pred_labels[row]): int(true_labels[int(np.argmax(counts[row]))])
            for row in range(counts.shape[0])
        }
        learned_edges = set()
        for u, v in learned_graph.edges():
            src_proto = int(learned_graph.nodes[u].get("prototype_id", u))
            dst_proto = int(learned_graph.nodes[v].get("prototype_id", v))
            if src_proto in majority_state and dst_proto in majority_state:
                learned_edges.add((majority_state[src_proto], majority_state[dst_proto]))
        true_edges = {
            (src, dst)
            for src in range(automaton.n_states)
            for dst in range(automaton.n_states)
            if float(automaton.transition_probs[src, dst]) > float(transition_threshold)
        }
        overlap = len(learned_edges & true_edges)
        metrics["edge_precision"] = overlap / float(max(1, len(learned_edges)))
        metrics["edge_recall"] = overlap / float(max(1, len(true_edges)))

    return metrics
