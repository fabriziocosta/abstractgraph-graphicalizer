"""Graph interpretation bottleneck graphicalizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abstractgraph_graphicalizer.core import GraphicalizerMixin


@dataclass
class BottleneckOutput:
    """Raw output from a graph interpretation bottleneck forward pass."""

    node_embeddings: torch.Tensor
    adjacency: torch.Tensor
    edge_probabilities: torch.Tensor
    assignments: torch.Tensor
    node_types: torch.Tensor
    token_to_nodes: torch.Tensor
    active_node_mask: torch.Tensor
    losses: dict[str, torch.Tensor]
    token_embeddings: torch.Tensor | None = None
    reconstructed_token_embeddings: torch.Tensor | None = None
    tokens: list[Any] | None = None
    input_id: str | None = None
    metadata: dict[str, Any] | None = None


class _ResidualGraphLayer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        row_mass = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        messages = torch.matmul(adjacency / row_mass, h)
        return self.norm(h + self.ff(messages))


class TinySequenceTransformer(nn.Module):
    """Small sequence encoder for synthetic symbol experiments."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_length: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(int(max_length), self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, tokens: torch.Tensor, return_embeddings: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
        h = self.token_embedding(tokens) + self.position_embedding(positions)
        h = self.encoder(h)
        if return_embeddings:
            return h
        return self.output(h)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens, return_embeddings=True)


class GraphInterpretationBottleneck(nn.Module):
    """Self-supervised neural module that induces sparse predictive graphs."""

    def __init__(
        self,
        d_model: int,
        *,
        num_prototypes: int = 64,
        temperature: float = 0.1,
        node_mask_ratio: float = 0.15,
        token_mask_ratio: float = 0.15,
        top_k_edges: int = 8,
        edge_threshold: float = 0.5,
        hidden_dim: int | None = None,
        gnn_layers: int = 3,
        lambda_node: float = 1.0,
        lambda_token: float = 0.1,
        lambda_sparse: float = 0.01,
        lambda_binary: float = 0.01,
        lambda_entropy: float = 0.01,
        active_mass_threshold: float = 1e-6,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.node_mask_ratio = float(node_mask_ratio)
        self.token_mask_ratio = float(token_mask_ratio)
        self.top_k_edges = int(top_k_edges)
        self.edge_threshold = float(edge_threshold)
        self.lambda_node = float(lambda_node)
        self.lambda_token = float(lambda_token)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_binary = float(lambda_binary)
        self.lambda_entropy = float(lambda_entropy)
        self.active_mass_threshold = float(active_mass_threshold)
        self.eps = float(eps)

        hidden_dim = int(hidden_dim or d_model)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(self.d_model))
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * self.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gnn_layers = nn.ModuleList(
            [_ResidualGraphLayer(self.d_model, hidden_dim) for _ in range(int(gnn_layers))]
        )
        self.token_decoder = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.d_model),
        )

    def _sample_mask(self, length: int, ratio: float, device: torch.device) -> torch.Tensor:
        n_mask = max(1, int(round(float(length) * float(ratio))))
        n_mask = min(length, n_mask)
        order = torch.randperm(length, device=device)
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        mask[order[:n_mask]] = True
        return mask

    def _assign_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(x_norm, proto_norm.transpose(0, 1))
        assignments = F.softmax(sim / max(self.temperature, self.eps), dim=-1)
        mass = assignments.sum(dim=0).clamp_min(self.eps)
        node_embeddings = torch.matmul(assignments.transpose(0, 1), x) / mass.unsqueeze(-1)
        return assignments, node_embeddings, mass

    def _edge_probabilities(self, z: torch.Tensor) -> torch.Tensor:
        k, d_model = z.shape
        zi = z[:, None, :].expand(k, k, d_model)
        zj = z[None, :, :].expand(k, k, d_model)
        pair = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
        logits = self.edge_mlp(pair).squeeze(-1)
        logits = logits.masked_fill(torch.eye(k, dtype=torch.bool, device=z.device), -1e9)
        prob = torch.sigmoid(logits)
        if k <= 1:
            return torch.zeros_like(prob)
        top_k = min(max(1, self.top_k_edges), k - 1)
        _, indices = torch.topk(prob, k=top_k, dim=-1)
        keep = torch.zeros_like(prob)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return prob * keep

    def forward(
        self,
        token_embeddings: torch.Tensor,
        *,
        node_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        tokens: list[Any] | None = None,
        input_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BottleneckOutput:
        if token_embeddings.dim() != 2:
            raise ValueError("token_embeddings must have shape (n_tokens, d_model)")
        if token_embeddings.shape[1] != self.d_model:
            raise ValueError(
                f"Expected embedding dimension {self.d_model}, got {token_embeddings.shape[1]}"
            )

        x = token_embeddings
        device = x.device
        assignments, node_embeddings, mass = self._assign_tokens(x)

        if node_mask is None:
            node_mask = self._sample_mask(self.num_prototypes, self.node_mask_ratio, device)
        else:
            node_mask = node_mask.to(device=device, dtype=torch.bool)
        if token_mask is None:
            token_mask = self._sample_mask(x.shape[0], self.token_mask_ratio, device)
        else:
            token_mask = token_mask.to(device=device, dtype=torch.bool)

        z_masked = node_embeddings.clone()
        z_masked[node_mask] = self.mask_token
        edge_prob = self._edge_probabilities(z_masked)
        edge_hard = (edge_prob > self.edge_threshold).to(edge_prob.dtype)
        adjacency = edge_hard.detach() - edge_prob.detach() + edge_prob

        z_reconstructed = z_masked
        for layer in self.gnn_layers:
            z_reconstructed = layer(z_reconstructed, adjacency)

        token_context = torch.matmul(assignments, z_reconstructed)
        x_hat = self.token_decoder(token_context)

        loss_node = F.mse_loss(z_reconstructed[node_mask], node_embeddings.detach()[node_mask])
        loss_token = F.mse_loss(x_hat[token_mask], x.detach()[token_mask])
        loss_sparse = edge_prob.mean()
        loss_binary = (edge_prob * (1.0 - edge_prob)).mean()
        entropy = -(assignments * torch.log(assignments.clamp_min(self.eps))).sum(dim=-1).mean()
        loss = (
            self.lambda_node * loss_node
            + self.lambda_token * loss_token
            + self.lambda_sparse * loss_sparse
            + self.lambda_binary * loss_binary
            + self.lambda_entropy * entropy
        )

        active = mass > self.active_mass_threshold
        return BottleneckOutput(
            node_embeddings=z_reconstructed,
            adjacency=adjacency,
            edge_probabilities=edge_prob,
            assignments=assignments,
            node_types=torch.arange(self.num_prototypes, device=device),
            token_to_nodes=torch.argmax(assignments, dim=-1),
            active_node_mask=active,
            losses={
                "loss": loss,
                "loss_node": loss_node,
                "loss_token": loss_token,
                "loss_sparse": loss_sparse,
                "loss_binary": loss_binary,
                "loss_entropy": entropy,
            },
            token_embeddings=x,
            reconstructed_token_embeddings=x_hat,
            tokens=tokens,
            input_id=input_id,
            metadata=metadata,
        )


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def bottleneck_output_to_networkx(
    output: BottleneckOutput,
    *,
    output_graph: str = "directed",
    active_only: bool = True,
    edge_label: str = "predictive",
) -> nx.Graph:
    """Convert a bottleneck output into a plain NetworkX graph."""

    if output_graph not in {"directed", "undirected"}:
        raise ValueError("output_graph must be 'directed' or 'undirected'")

    graph: nx.Graph = nx.DiGraph() if output_graph == "directed" else nx.Graph()
    node_embeddings = _to_numpy(output.node_embeddings)
    assignments = _to_numpy(output.assignments)
    adjacency = _to_numpy(output.adjacency)
    probabilities = _to_numpy(output.edge_probabilities)
    active_mask = _to_numpy(output.active_node_mask).astype(bool)
    mass = assignments.sum(axis=0)

    if not active_only:
        active_mask[:] = True
    active_ids = [idx for idx, is_active in enumerate(active_mask) if is_active]
    id_map = {prototype_id: node_idx for node_idx, prototype_id in enumerate(active_ids)}

    for prototype_id in active_ids:
        graph.add_node(
            id_map[prototype_id],
            embedding=node_embeddings[prototype_id],
            node_type=int(prototype_id),
            prototype_id=int(prototype_id),
            assignment_mass=float(mass[prototype_id]),
            label=int(prototype_id),
        )

    if output_graph == "undirected":
        adjacency = np.maximum(adjacency, adjacency.T)
        probabilities = np.maximum(probabilities, probabilities.T)

    for src in active_ids:
        for dst in active_ids:
            if src == dst:
                continue
            weight = float(adjacency[src, dst])
            if weight <= 0.0:
                continue
            u = id_map[src]
            v = id_map[dst]
            if not graph.is_directed() and graph.has_edge(v, u):
                continue
            graph.add_edge(
                u,
                v,
                weight=weight,
                probability=float(probabilities[src, dst]),
                edge_type=edge_label,
                label=edge_label,
            )

    token_to_nodes = _to_numpy(output.token_to_nodes).astype(int)
    graph.graph["source"] = "graph_interpretation_bottleneck"
    graph.graph["assignments"] = assignments
    graph.graph["token_to_nodes"] = token_to_nodes
    graph.graph["active_prototype_ids"] = np.asarray(active_ids, dtype=int)
    if output.tokens is not None:
        graph.graph["tokens"] = output.tokens
    if output.input_id is not None:
        graph.graph["input_id"] = output.input_id
    if output.metadata is not None:
        graph.graph["metadata"] = output.metadata
    return graph


class BottleneckGraphicalizer(GraphicalizerMixin):
    """Scikit-style wrapper around :class:`GraphInterpretationBottleneck`."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        d_in: int | None = None,
        num_prototypes: int = 64,
        use_encoder: bool = True,
        n_epochs: int = 5,
        lr: float = 1e-3,
        device: str = "auto",
        output_graph: str = "directed",
        active_only: bool = True,
        active_mass_threshold: float = 1e-6,
        **bottleneck_kwargs: Any,
    ) -> None:
        self.d_model = d_model
        self.d_in = d_in
        self.num_prototypes = num_prototypes
        self.use_encoder = use_encoder
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.output_graph = output_graph
        self.active_only = active_only
        self.active_mass_threshold = active_mass_threshold
        self.bottleneck_kwargs = bottleneck_kwargs
        self.encoder_: nn.Module | None = None
        self.model_: GraphInterpretationBottleneck | None = None
        self._torch_device: torch.device | None = None

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        return torch.device(self.device)

    def _prepare_instance(self, instance: Any) -> torch.Tensor:
        if isinstance(instance, dict):
            instance = instance.get("token_embeddings", instance.get("x"))
        tensor = instance.detach().clone() if isinstance(instance, torch.Tensor) else torch.as_tensor(instance)
        if tensor.dim() != 2:
            raise ValueError("Each instance must have shape (n_tokens, n_features)")
        return tensor.to(device=self._torch_device, dtype=torch.float32)

    def _ensure_modules(self, first: torch.Tensor) -> None:
        d_in = int(self.d_in or first.shape[1])
        kwargs = dict(self.bottleneck_kwargs)
        kwargs.setdefault("active_mass_threshold", self.active_mass_threshold)
        self.model_ = GraphInterpretationBottleneck(
            d_model=int(self.d_model),
            num_prototypes=int(self.num_prototypes),
            **kwargs,
        ).to(self._torch_device)
        if self.use_encoder:
            self.encoder_ = nn.Linear(d_in, int(self.d_model)).to(self._torch_device)
        elif first.shape[1] != int(self.d_model):
            raise ValueError("use_encoder=False requires input feature dimension to equal d_model")

    def _embed(self, instance: torch.Tensor) -> torch.Tensor:
        if self.use_encoder:
            if self.encoder_ is None:
                raise RuntimeError("Encoder is not initialized")
            return self.encoder_(instance)
        return instance

    def fit(self, X, y=None):
        self._torch_device = self._resolve_device()
        instances = [self._prepare_instance(instance) for instance in X]
        if len(instances) == 0:
            raise ValueError("Empty dataset: X has no instances")
        self._ensure_modules(instances[0])
        params = list(self.model_.parameters())
        if self.encoder_ is not None:
            params += list(self.encoder_.parameters())
        optimizer = torch.optim.Adam(params, lr=float(self.lr))
        self.model_.train()
        if self.encoder_ is not None:
            self.encoder_.train()
        for _ in range(int(self.n_epochs)):
            for instance in instances:
                optimizer.zero_grad(set_to_none=True)
                embeddings = self._embed(instance)
                output = self.model_(embeddings)
                output.losses["loss"].backward()
                optimizer.step()
        return self

    def transform(self, X, y=None) -> list[nx.Graph]:
        if self.model_ is None:
            raise RuntimeError("Model not fit. Call fit() before transform().")
        self.model_.eval()
        if self.encoder_ is not None:
            self.encoder_.eval()
        graphs: list[nx.Graph] = []
        with torch.no_grad():
            for raw in X:
                tokens = raw.get("tokens") if isinstance(raw, dict) else None
                input_id = raw.get("input_id") if isinstance(raw, dict) else None
                metadata = raw.get("metadata") if isinstance(raw, dict) else None
                instance = self._prepare_instance(raw)
                embeddings = self._embed(instance)
                output = self.model_(
                    embeddings,
                    tokens=tokens,
                    input_id=input_id,
                    metadata=metadata,
                )
                graphs.append(
                    bottleneck_output_to_networkx(
                        output,
                        output_graph=self.output_graph,
                        active_only=self.active_only,
                    )
                )
        return graphs

    def fit_transform(self, X, y=None, **fit_params):
        instances = list(X)
        self.fit(instances, y=y)
        return self.transform(instances)
