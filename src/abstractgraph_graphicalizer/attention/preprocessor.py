import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Iterable, Callable, Optional, Tuple

try:
    # Optional import; only needed when ImageNodeClusterer is used
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore


# ============================================================
# 1. Transformer backbone
# ============================================================


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Multi-head self-attention without positional encodings.

        Args:
            d_model: Transformer hidden size.
            n_heads: Number of attention heads.
            dropout: Dropout probability on attention weights.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, n_tokens, d_model).
            return_attn: If True, also returns attention weights.

        Returns:
            out: Tensor of shape (batch, n_tokens, d_model).
            attn: Optional attention tensor of shape (batch, n_heads, n_tokens, n_tokens).
        """
        B, N, D = x.shape
        H = self.n_heads
        d_h = self.head_dim

        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to heads: (B, H, N, d_h)
        def split_heads(t):
            return t.view(B, N, H, d_h).transpose(1, 2)

        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)

        # scaled dot-product attention
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(d_h)  # (B, H, N, N)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, vh)  # (B, H, N, d_h)

        # combine heads
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(context)
        if return_attn:
            return out, attn
        return out, None


class TransformerEncoderLayerCustom(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Single transformer encoder layer with self-attention and feedforward.

        Args:
            d_model: Transformer hidden size.
            n_heads: Number of attention heads.
            dim_feedforward: Size of feedforward intermediate layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: Input tensor of shape (batch, n_tokens, d_model).
            return_attn: If True, returns attention weights.

        Returns:
            out: Tensor of shape (batch, n_tokens, d_model).
            attn: Optional attention tensor (batch, n_heads, n_tokens, n_tokens).
        """
        # Self-attention + residual + norm
        attn_out, attn = self.self_attn(src, return_attn=True)
        src = self.norm1(src + self.dropout1(attn_out))

        # Feedforward + residual + norm
        ff = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        out = self.norm2(src + ff)

        if return_attn:
            return out, attn
        return out, None


class SimpleTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Simple transformer encoder for sets of tokens (no positional encodings).

        Args:
            d_in: Input token embedding dimension.
            d_model: Transformer hidden size.
            n_heads: Number of attention heads.
            num_layers: Number of encoder layers.
            dim_feedforward: Hidden size in feedforward layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerCustom(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Tensor of shape (batch, n_tokens, d_in).
            return_attn: If True, returns attention tensors per layer.

        Returns:
            out: Tensor of shape (batch, n_tokens, d_model).
            attn_list: Optional list of attention tensors, one per layer.
        """
        h = self.dropout(self.input_proj(x))
        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            h, attn = layer(h, return_attn=True)
            if return_attn and attn is not None:
                attn_list.append(attn)
        if return_attn:
            return h, attn_list
        return h, None


# ============================================================
# 2. Graph extraction helpers (MST, DP, preimage edges)
# ============================================================


def maximum_spanning_tree_edges(W: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Compute a maximum spanning tree on a dense symmetric weight matrix.

    Args:
        W: Symmetric weight matrix of shape (N, N).

    Returns:
        List of edges (u, v, weight) for the MST.
    """
    N = W.shape[0]
    if N <= 1:
        return []

    # Build edge list from upper triangle
    edges: List[Tuple[float, int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            w = float(W[i, j])
            if not math.isfinite(w):
                continue
            edges.append((w, i, j))

    # Sort descending by weight for maximum spanning tree
    edges.sort(key=lambda t: t[0], reverse=True)

    parent = list(range(N))
    rank = [0] * N

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst: List[Tuple[int, int, float]] = []
    for w, i, j in edges:
        if union(i, j):
            mst.append((i, j, float(w)))
            if len(mst) == N - 1:
                break

    return mst


def dp_forest_on_tree(
    N: int,
    edges: List[Tuple[int, int, float]],
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Tuple[set, np.ndarray]:
    """
    Run tree-structured DP to cut edges and obtain a forest (clusters).

    Args:
        N: Number of nodes.
        edges: List of tree edges (u, v, w).
        alpha: Weight on edge strength for 'keep' decisions.
        beta: Penalty for starting a new component.

    Returns:
        keep_edges: Set of (u, v) pairs for edges that are kept.
        cluster_ids: Array of shape (N,) with component id per node.
    """
    # This implementation reduces to a per-edge decision rule
    # keep if alpha * w >= beta, then derive components as CCs of kept edges.
    keep_edges: set = set()
    for u, v, w in edges:
        if alpha * float(w) >= beta:
            a, b = (u, v) if u < v else (v, u)
            keep_edges.add((a, b))

    # Build components over N nodes from kept edges
    parent = list(range(N))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in keep_edges:
        union(u, v)

    # Flatten parents and assign compact component ids
    roots = {}
    cluster_ids = np.zeros(N, dtype=int)
    next_id = 0
    for i in range(N):
        r = find(i)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        cluster_ids[i] = roots[r]

    return keep_edges, cluster_ids


# (Legacy image-node grouping from attention removed.)


def build_preimage_edges_from_attention(
    W: np.ndarray,
    K_mst: int = 3,
    alpha: float = 1.0,
    beta: float = 0.5,
    rho: float = 0.6,
) -> Tuple[List[Tuple[int, int, Dict[str, Any]]], np.ndarray]:
    """
    Build robust token-level edges (preimage graph edges) from a symmetric attention matrix.

    Uses the same overall procedure (iterated MST + DP forests + consensus co-clustering)
    to induce robust adjacency between raw tokens rather than grouping them into image nodes.

    Args:
        W: Symmetric attention matrix (N, N).
        K_mst: Number of iterated MSTs.
        alpha: Edge-keep weight for DP on each MST.
        beta: Component-start penalty for DP on each MST.
        rho: Co-clustering threshold for robust adjacency.

    Returns:
        edges: List of (u, v, attr) tuples suitable for adding to a NetworkX Graph.
               attr includes keys like 'consensus', 'mst_count', 'mst_weight_mean', and 'weight'.
        co_cluster_matrix: (N, N) array with co-clustering frequencies.
    """
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    if N <= 1:
        return [], np.eye(N, dtype=float)

    # Working copy to derive disjoint MSTs across iterations
    W_work = W.copy()
    np.fill_diagonal(W_work, -np.inf)

    forests: List[np.ndarray] = []
    # Track DP-kept MST edges across iterations
    dp_kept_counts: Dict[Tuple[int, int], int] = {}
    dp_kept_weights: Dict[Tuple[int, int], List[float]] = {}

    for _ in range(max(1, int(K_mst))):
        mst_edges = maximum_spanning_tree_edges(W_work)
        if len(mst_edges) == 0:
            break
        # Remove MST edges to encourage diversity across iterations
        for u, v, _w in mst_edges:
            W_work[u, v] = -np.inf
            W_work[v, u] = -np.inf

        # DP forest on this MST
        keep_edges, cluster_ids = dp_forest_on_tree(N, mst_edges, alpha=alpha, beta=beta)
        forests.append(cluster_ids)
        # Accumulate kept edges
        for (u, v) in keep_edges:
            a, b = (u, v) if u < v else (v, u)
            w = float(W[u, v])
            dp_kept_counts[(a, b)] = dp_kept_counts.get((a, b), 0) + 1
            dp_kept_weights.setdefault((a, b), []).append(w)

    n_forests = max(1, len(forests))

    # Consensus co-clustering
    co_counts = np.zeros((N, N), dtype=float)
    for cluster_ids in forests:
        for c in np.unique(cluster_ids):
            idx = np.where(cluster_ids == c)[0]
            if idx.size == 0:
                continue
            co_counts[np.ix_(idx, idx)] += 1.0
    co_freq = co_counts / float(n_forests)
    np.fill_diagonal(co_freq, 1.0)

    # Build edges from consensus adjacency and DP-kept MST edges
    edges_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # 1) Consensus edges
    if N > 1:
        for i in range(N):
            for j in range(i + 1, N):
                c = float(co_freq[i, j])
                if c >= float(rho):
                    edges_map[(i, j)] = {
                        "consensus": c,
                        "weight": c,  # use consensus frequency as primary weight
                    }

    # 2) DP-kept MST edges (fill in where consensus didn't connect)
    for (a, b), cnt in dp_kept_counts.items():
        attr = edges_map.get((a, b), {})
        ws = dp_kept_weights.get((a, b), [])
        mean_w = float(np.mean(ws)) if len(ws) > 0 else float(W[a, b])
        attr.update({
            "mst_count": int(cnt),
            "mst_weight_mean": mean_w,
        })
        # If no consensus weight already set, use MST mean as 'weight'
        if "weight" not in attr:
            attr["weight"] = mean_w
        edges_map[(a, b)] = attr

    # Emit edges list
    edges: List[Tuple[int, int, Dict[str, Any]]] = []
    for (u, v), data in edges_map.items():
        edges.append((u, v, data))

    return edges, co_freq


# ============================================================
# 3. Dataset-level node embedding clustering
# ============================================================


class ImageNodeClusterer(BaseEstimator):
    """
    Clusterer for abstract node embeddings across a dataset, to obtain
    discrete type labels for image nodes.
    """

    def __init__(self, n_clusters: int = 32, cluster_method: str = "kmeans", random_state: Optional[int] = None):
        """
        Args:
            n_clusters: Number of clusters (node types).
            cluster_method: Clustering algorithm identifier.
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = int(n_clusters)
        self.cluster_method = cluster_method
        self.random_state = random_state
        self._cluster_model = None

    def fit(self, Z: np.ndarray, y=None):
        """
        Fit the clustering model on node embeddings.

        Args:
            Z: Node embeddings, shape (n_nodes_total, d_model).
            y: Ignored (for compatibility).

        Returns:
            self
        """
        Z = np.asarray(Z, dtype=float)
        n = Z.shape[0]
        if n == 0:
            # No nodes: create a dummy 1-cluster model
            class _Dummy:
                def predict(self, X):
                    return np.zeros(len(X), dtype=int)

            self._cluster_model = _Dummy()
            return self

        if self.cluster_method.lower() == "kmeans":
            if KMeans is None:
                raise ImportError("scikit-learn is required for KMeans clustering")
            k = max(1, min(self.n_clusters, n))
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            model.fit(Z)
            self._cluster_model = model
        else:
            raise ValueError(f"Unsupported cluster_method: {self.cluster_method}")
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Assign cluster ids to node embeddings.

        Args:
            Z: Node embeddings, shape (n_nodes, d_model).

        Returns:
            cluster_ids: Array of shape (n_nodes,) with cluster indices.
        """
        if self._cluster_model is None:
            raise RuntimeError("ImageNodeClusterer must be fit before predict()")
        Z = np.asarray(Z, dtype=float)
        return np.asarray(self._cluster_model.predict(Z))

    def fit_predict(self, Z: np.ndarray, y=None) -> np.ndarray:
        """
        Convenience method: fit clustering model and return cluster ids.

        Args:
            Z: Node embeddings, shape (n_nodes_total, d_model).
            y: Ignored.

        Returns:
            cluster_ids: Array of shape (n_nodes_total,).
        """
        self.fit(Z, y=y)
        return self.predict(Z)


# ============================================================
# 4. Scikit-style wrapper for transformer + abstract graph extraction
# ============================================================


class AbstractGraphPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end wrapper:
      - fit: train transformer encoder on instance-level labels.
      - transform: extract abstract graphs per instance from attention.
      - Optional external clustering model can be used to label image nodes.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        K_mst: int = 3,
        alpha: float = 1.0,
        beta: float = 0.5,
        rho: float = 0.6,
        n_epochs: int = 5,
        lr: float = 1e-3,
        device: str = "auto",
        label_fn: Optional[Callable[[np.ndarray, List[int]], Any]] = None,
        node_clusterer: Optional[ImageNodeClusterer] = None,
    ):
        """
        Args:
            d_model: Transformer hidden size.
            n_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_feedforward: Feedforward hidden size.
            dropout: Dropout probability.
            K_mst: Number of iterated MSTs for graph extraction.
            alpha: Edge-keep weight in DP.
            beta: Component-start penalty in DP.
            rho: Co-clustering threshold.
            n_epochs: Number of training epochs for transformer.
            lr: Learning rate for optimiser.
            device: 'auto', 'cpu', or 'cuda'.
            label_fn: Optional function to label image nodes from tokens.
            node_clusterer: Optional ImageNodeClusterer for dataset-level node types.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.K_mst = K_mst
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.label_fn = label_fn
        self.node_clusterer = node_clusterer

        self.model_: Optional[SimpleTransformerEncoder] = None
        self.classifier_: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.d_in_: Optional[int] = None
        self._torch_device: Optional[torch.device] = None

    def _resolve_device(self) -> None:
        """
        Set internal PyTorch device based on user configuration.
        """
        if self.device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if self.device not in ("cpu", "cuda"):
                raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
            dev = torch.device(self.device)
        self._torch_device = dev

    def _prepare_instance(self, x: Any) -> torch.Tensor:
        """
        Convert one instance into a (1, n_tokens, d_in) tensor on the correct device.

        Args:
            x: Array-like of shape (n_tokens, d_in).

        Returns:
            Tensor of shape (1, n_tokens, d_in).
        """
        if isinstance(x, torch.Tensor):
            t = x.detach()
        else:
            arr = np.asarray(x)
            try:
                # Prefer zero-copy path when NumPy->Torch bridge is available
                t = torch.from_numpy(arr)
            except Exception:
                # Fallback for environments where PyTorch cannot access NumPy
                t = torch.tensor(arr.tolist())
        if t.dim() != 2:
            raise ValueError(f"Each instance must be 2D (n_tokens, d_in), got {t.shape}")
        t = t.to(dtype=torch.float32, device=self._torch_device)
        t = t.unsqueeze(0)
        return t

    def fit(self, X: Iterable[Any], y: Iterable[Any]):
        """
        Train transformer encoder and classifier on instance-level labels.

        Args:
            X: Iterable of instances, each (n_tokens, d_in).
            y: Iterable of instance labels.

        Returns:
            self
        """
        self._resolve_device()
        X_list = list(X)
        y_list = list(y)
        if len(X_list) == 0:
            raise ValueError("Empty dataset: X has no instances")
        # Infer input dimension
        first = np.asarray(X_list[0])
        if first.ndim != 2:
            raise ValueError("Each X[i] must be 2D (n_tokens, d_in)")
        self.d_in_ = int(first.shape[1])

        # Label encoding
        classes, y_idx = np.unique(np.asarray(y_list), return_inverse=True)
        self.classes_ = classes
        n_classes = classes.shape[0]

        # Build model and classifier
        self.model_ = SimpleTransformerEncoder(
            d_in=self.d_in_,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self._torch_device)
        self.classifier_ = nn.Linear(self.d_model, n_classes).to(self._torch_device)

        params = list(self.model_.parameters()) + list(self.classifier_.parameters())
        optimizer = torch.optim.Adam(params, lr=float(self.lr))
        criterion = nn.CrossEntropyLoss()

        # Simple training loop with batch size 1 to avoid padding
        self.model_.train()
        self.classifier_.train()
        for epoch in range(int(self.n_epochs)):
            for xi, yi in zip(X_list, y_idx):
                optimizer.zero_grad(set_to_none=True)
                xt = self._prepare_instance(xi)
                out, _ = self.model_(xt, return_attn=False)
                pooled = out.mean(dim=1)  # (1, d_model)
                logits = self.classifier_(pooled)  # (1, n_classes)
                loss = criterion(logits, torch.tensor([int(yi)], device=self._torch_device))
                loss.backward()
                optimizer.step()

        return self

    def transform(self, X: Iterable[Any]) -> List[nx.Graph]:
        """
        For each instance, process raw data with the transformer and build the
        initial preimage graph as a NetworkX Graph.

        Args:
            X: Iterable of instances, each (n_tokens, d_in).

        Returns:
            List[nx.Graph], each with:
            - node attributes: 'embedding' (np.ndarray), optional 'label'
            - edge attributes: 'weight' (float), optional 'consensus', 'mst_count', 'mst_weight_mean'
            - graph attribute: G.graph['co_cluster'] = np.ndarray (N, N)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fit. Call fit() before transform().")
        self.model_.eval()
        outputs: List[nx.Graph] = []
        for xi in X:
            xt = self._prepare_instance(xi)
            with torch.no_grad():
                out, attn_list = self.model_(xt, return_attn=True)

            token_emb = out[0].detach().cpu().numpy()  # (N, d_model)
            N = token_emb.shape[0]

            # Aggregate attention across heads and layers
            if attn_list is None or len(attn_list) == 0:
                # Fallback: identity if attention not available
                W = np.eye(N, dtype=float)
            else:
                ws = []
                for A in attn_list:
                    # A: (1, H, N, N)
                    A_mean_heads = A.mean(dim=1)  # (1, N, N)
                    ws.append(A_mean_heads[0].detach().cpu().numpy())
                W = np.mean(np.stack(ws, axis=0), axis=0)
                # Symmetrise
                W = 0.5 * (W + W.T)

            # Build preimage graph with robust token-level edges
            edge_list, co_cluster = build_preimage_edges_from_attention(
                W=W,
                K_mst=self.K_mst,
                alpha=self.alpha,
                beta=self.beta,
                rho=self.rho,
            )

            G = nx.Graph()
            # Nodes with embeddings and optional labels
            xi_np = np.asarray(xi)
            for i in range(N):
                emb_i = token_emb[i]
                label_val = None
                if self.label_fn is not None:
                    try:
                        label_val = self.label_fn(xi_np, [i])
                    except Exception:
                        label_val = None
                G.add_node(i, embedding=emb_i, label=label_val)

            # Edges with attributes from the procedure
            for u, v, attr in edge_list:
                G.add_edge(int(u), int(v), **attr)

            # Attach co-cluster matrix as a graph attribute for inspection
            G.graph["co_cluster"] = co_cluster

            outputs.append(G)
        return outputs

    def fit_transform(self, X: Iterable[Any], y: Iterable[Any], **fit_params):
        """
        Fit the model and return transformed abstract graphs.

        Args:
            X: Iterable of instances.
            y: Iterable of labels.
            fit_params: Additional fit parameters.

        Returns:
            graphs: List of graph dicts.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: Iterable[Any]) -> np.ndarray:
        """
        Predict instance-level labels using the trained classifier.

        Args:
            X: Iterable of instances.

        Returns:
            predictions: Array of predicted labels, shape (n_instances,).
        """
        if self.model_ is None or self.classifier_ is None or self.classes_ is None:
            raise RuntimeError("Model not fit. Call fit() before predict().")
        self.model_.eval()
        preds: List[int] = []
        with torch.no_grad():
            for xi in X:
                xt = self._prepare_instance(xi)
                out, _ = self.model_(xt, return_attn=False)
                pooled = out.mean(dim=1)
                logits = self.classifier_(pooled)
                pred = int(torch.argmax(logits, dim=1).item())
                preds.append(pred)
        preds = np.asarray(preds, dtype=int)
        return self.classes_[preds]

    def extract_node_embeddings(
        self,
        graphs: Iterable[nx.Graph]
    ) -> np.ndarray:
        """
        Extract pooled node embeddings from abstract graphs for dataset-level clustering.

        Args:
            graphs: Iterable of NetworkX graphs (nodes carry 'embedding').

        Returns:
            Z: Array of node embeddings, shape (n_nodes_total, d_model).
        """
        emb_list: List[np.ndarray] = []
        for g in graphs:
            for _, data in g.nodes(data=True):
                if "embedding" in data:
                    emb_list.append(np.asarray(data["embedding"]))
        if len(emb_list) == 0:
            return np.zeros((0, int(self.d_model)), dtype=float)
        return np.stack(emb_list, axis=0)

    def assign_node_cluster_labels(
        self,
        graphs: Iterable[nx.Graph]
    ) -> List[nx.Graph]:
        """
        Use the fitted node_clusterer to assign discrete cluster ids to image nodes.

        Args:
            graphs: Iterable of outputs from transform() (with node embeddings).

        Returns:
            graphs_labeled: Same as input, but each node dict gets an extra 'cluster_id'.
        """
        if self.node_clusterer is None:
            return graphs

        graphs = list(graphs)
        Z = self.extract_node_embeddings(graphs)
        if getattr(self.node_clusterer, "_cluster_model", None) is None:
            self.node_clusterer.fit(Z)
        cluster_ids = self.node_clusterer.predict(Z)

        # Assign sequentially to nodes across graphs
        idx = 0
        out_list: List[nx.Graph] = []
        for g in graphs:
            G = g.copy()
            for node in G.nodes():
                cid = int(cluster_ids[idx]) if idx < len(cluster_ids) else -1
                idx += 1
                G.nodes[node]["cluster_id"] = cid
            out_list.append(G)
        return out_list
