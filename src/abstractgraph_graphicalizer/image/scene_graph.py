"""Image scene-graph graphicalizers based on supplied segments."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean

from abstractgraph_graphicalizer.core import GraphicalizerMixin


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    return float(inter / union) if union > 0 else 0.0


def filter_by_size(
    segments: list[dict[str, Any]],
    image_size: tuple[int, int],
    *,
    min_size: float | None = None,
    max_size: float | None = None,
    mask_key: str = "mask",
) -> list[dict[str, Any]]:
    if min_size is None and max_size is None:
        return segments
    total_pixels = image_size[0] * image_size[1]
    kept: list[dict[str, Any]] = []
    for seg in segments:
        if mask_key in seg and seg[mask_key] is not None:
            area = int(np.count_nonzero(seg[mask_key]))
        else:
            x0, y0, x1, y1 = seg["bbox"]
            area = int((x1 - x0) * (y1 - y0))
        if min_size is not None:
            threshold = min_size * total_pixels if min_size <= 1 else min_size
            if area < threshold:
                continue
        if max_size is not None:
            threshold = max_size * total_pixels if max_size <= 1 else max_size
            if area > threshold:
                continue
        kept.append(seg)
    return kept


def filter_overlapping_by_iou(
    segments: list[dict[str, Any]],
    *,
    iou_threshold: float | None,
    conf_key: str = "score",
) -> list[dict[str, Any]]:
    if iou_threshold is None:
        return segments

    def compute_bbox_iou(box1, box2) -> float:
        x1, y1, x2, y2 = box1
        xa, ya, xb, yb = box2
        xi1, yi1 = max(x1, xa), max(y1, ya)
        xi2, yi2 = min(x2, xb), min(y2, yb)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (xb - xa) * (yb - ya)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    sorted_segments = sorted(segments, key=lambda seg: seg.get(conf_key, 0.0), reverse=True)
    keep: list[dict[str, Any]] = []
    for seg in sorted_segments:
        mask = seg.get("mask")
        should_keep = True
        for other in keep:
            other_mask = other.get("mask")
            if mask is not None and other_mask is not None:
                overlap = mask_iou(mask, other_mask)
            else:
                overlap = compute_bbox_iou(seg["bbox"], other["bbox"])
            if overlap > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(seg)
    return keep


def _compute_bbox_centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _compute_mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean()))


def _mask_extents(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_geometric_relations_graph(
    segments: list[dict[str, Any]],
    *,
    selected_labels: list[str] | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
    use_masks: bool = True,
    near_threshold: float = 0.05,
    overlap_area_threshold: float = 0.0,
    containment_area_threshold: float = 1.0,
    include_overlapping: bool = True,
    include_contained: bool = True,
    include_near: bool = True,
    include_left_of: bool = True,
    include_above: bool = True,
    n_iter: int = 1,
) -> nx.MultiDiGraph:
    """Build an MST-pruned scene graph from segment geometry."""
    graph = nx.MultiDiGraph()
    indexed = list(enumerate(segments))
    if selected_labels is not None:
        allowed = set(selected_labels)
        indexed = [
            (idx, seg)
            for idx, seg in indexed
            if (seg.get("semantic_label") or seg.get("label")) in allowed
        ]
    if not indexed:
        return graph

    h = w = None
    if use_masks:
        for _, seg in indexed:
            mask = seg.get("mask")
            if mask is not None:
                h, w = mask.shape[:2]
                break
    if h is None or w is None:
        w = max(seg["bbox"][2] for _, seg in indexed) + 1
        h = max(seg["bbox"][3] for _, seg in indexed) + 1

    indexed = [
        (idx, seg)
        for idx, seg in indexed
        if seg in filter_by_size([seg], (h, w), min_size=min_size, max_size=max_size)
    ]
    if not indexed:
        return graph

    min_dim = min(w, h)
    abs_near = near_threshold * min_dim if near_threshold <= 1 else near_threshold
    geoms: list[dict[str, Any]] = []
    orig_idxs = [idx for idx, _ in indexed]
    for _, seg in indexed:
        if use_masks and seg.get("mask") is not None:
            mask = seg["mask"]
            area = int(mask.sum())
            centroid = _compute_mask_centroid(mask)
            min_x, min_y, max_x, max_y = _mask_extents(mask)
        else:
            mask = None
            x0, y0, x1, y1 = seg["bbox"]
            area = int((x1 - x0 + 1) * (y1 - y0 + 1))
            centroid = _compute_bbox_centroid((x0, y0, x1, y1))
            min_x, min_y, max_x, max_y = x0, y0, x1, y1
        geoms.append(
            {
                "mask": mask,
                "area": area,
                "centroid": centroid,
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            }
        )

    relations: list[tuple[int, int, str]] = []
    n = len(geoms)
    for a_idx, ga in enumerate(geoms):
        for b_idx, gb in enumerate(geoms):
            if a_idx == b_idx:
                continue
            overlap = 0
            if include_overlapping or include_contained:
                if ga["mask"] is not None and gb["mask"] is not None:
                    overlap = int(np.sum(ga["mask"] & gb["mask"]))
                else:
                    xi0 = max(ga["min_x"], gb["min_x"])
                    yi0 = max(ga["min_y"], gb["min_y"])
                    xi1 = min(ga["max_x"], gb["max_x"])
                    yi1 = min(ga["max_y"], gb["max_y"])
                    overlap = max(0, xi1 - xi0 + 1) * max(0, yi1 - yi0 + 1)
            if include_overlapping:
                threshold = (
                    overlap_area_threshold * ga["area"]
                    if overlap_area_threshold <= 1
                    else overlap_area_threshold
                )
                if overlap >= threshold:
                    relations.append((a_idx, b_idx, "is_overlapping"))
            if include_contained:
                threshold = (
                    containment_area_threshold * ga["area"]
                    if containment_area_threshold <= 1
                    else containment_area_threshold
                )
                if overlap >= threshold:
                    relations.append((a_idx, b_idx, "is_contained"))
            if include_near and euclidean(ga["centroid"], gb["centroid"]) < abs_near:
                relations.append((a_idx, b_idx, "is_near"))
            if include_left_of and ga["max_x"] < gb["min_x"]:
                relations.append((a_idx, b_idx, "is_left_of"))
            if include_above and ga["max_y"] < gb["min_y"]:
                relations.append((a_idx, b_idx, "is_above"))

    base_edges: dict[tuple[int, int], float] = {}
    for a_idx, b_idx, _ in relations:
        u, v = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
        distance = euclidean(geoms[u]["centroid"], geoms[v]["centroid"])
        if (u, v) not in base_edges or distance < base_edges[(u, v)]:
            base_edges[(u, v)] = distance

    def kruskal(edges: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def unite(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        mst: list[tuple[int, int]] = []
        for (u, v), _ in sorted(edges.items(), key=lambda item: item[1]):
            if find(u) != find(v):
                unite(u, v)
                mst.append((u, v))
                if len(mst) == n - 1:
                    break
        return mst

    remaining = dict(base_edges)
    layered: list[tuple[int, int]] = []
    for _ in range(n_iter):
        if not remaining:
            break
        layer = kruskal(remaining)
        if not layer:
            break
        for edge in layer:
            layered.append(edge)
            remaining.pop(edge, None)
    pair_set = {tuple(sorted(edge)) for edge in layered}

    for orig_i, seg in indexed:
        label = seg.get("semantic_label") or seg.get("label") or "object"
        bbox = seg.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            warnings.warn(f"Segment {orig_i} missing valid bbox, setting None")
            bbox = None
        position = geoms[orig_idxs.index(orig_i)]["centroid"]
        confidence = seg.get("semantic_confidence") or seg.get("score") or seg.get("mask_score") or 0.0
        graph.add_node(
            orig_i,
            label=label,
            caption=seg.get("caption"),
            bbox=bbox,
            confidence=confidence,
            pos=position,
        )

    for a_idx, b_idx, relation in relations:
        u, v = orig_idxs[a_idx], orig_idxs[b_idx]
        if tuple(sorted((a_idx, b_idx))) in pair_set and graph.has_node(u) and graph.has_node(v):
            graph.add_edge(u, v, relation=relation)

    return graph


def visualize_scene_graph_on_image(
    image: np.ndarray,
    segments: list[dict[str, Any]],
    graph: nx.MultiDiGraph,
    *,
    show_image: bool = True,
    show_masks: bool = True,
    show_bbox: bool = True,
    show_graph: bool = True,
    alpha: float = 0.25,
    offset: float = 10.0,
    ax=None,
):
    """Visualize a scene graph overlaid on an image and return the axis."""
    height, width, _ = image.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    if show_image:
        ax.imshow(image)
    else:
        ax.set_facecolor("white")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    if show_masks:
        overlay = image.astype(float).copy()
        for idx, seg in enumerate(segments):
            if idx not in graph.nodes:
                continue
            mask = seg.get("mask")
            if mask is None:
                x0, y0, x1, y1 = map(int, seg["bbox"])
                mask = np.zeros((height, width), dtype=bool)
                mask[y0:y1, x0:x1] = True
            color = (np.random.rand(3) * 255).astype(float)
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
        ax.imshow(overlay.astype(np.uint8), extent=[0, width, height, 0])

    if show_bbox:
        for idx, seg in enumerate(segments):
            if idx not in graph.nodes:
                continue
            x0, y0, x1, y1 = seg["bbox"]
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                    zorder=2,
                )
            )

    if show_graph:
        pos = {
            node: (
                (segments[node]["bbox"][0] + segments[node]["bbox"][2]) / 2.0,
                (segments[node]["bbox"][1] + segments[node]["bbox"][3]) / 2.0,
            )
            for node in graph.nodes
        }
        for node, (x, y) in pos.items():
            ax.scatter(x, y, s=80, color="#f4a261", edgecolors="black", zorder=3)
            ax.text(
                x,
                y,
                graph.nodes[node].get("label", str(node)),
                ha="center",
                va="center_baseline",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
                zorder=4,
            )
        for source, target, data in graph.edges(data=True):
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->"))
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            perp = np.array([-(y1 - y0), x1 - x0], dtype=float)
            norm = np.linalg.norm(perp)
            if norm != 0:
                perp = perp / norm
            mx_off, my_off = mx + perp[0] * offset, my + perp[1] * offset
            ax.text(mx_off, my_off, data.get("relation", ""), fontsize=8, ha="center", va="center")

    ax.axis("off")
    return ax


def load_images(directory: str | Path, *, suffix: str = ".jpg", return_names: bool = False):
    """Load images from a directory in lexicographic order."""
    image_files = sorted(
        filename for filename in os.listdir(directory) if filename.lower().endswith(suffix.lower())
    )
    if not image_files:
        warnings.warn(f"No images found with suffix {suffix} in {directory}")
        return ([], []) if return_names else []

    images: list[np.ndarray] = []
    names: list[str] = []
    for filename in image_files:
        path = Path(directory) / filename
        images.append(np.array(Image.open(path).convert("RGB")))
        names.append(filename)
    return (images, names) if return_names else images


class ImageSegmentGraphicalizer(GraphicalizerMixin):
    """Build scene graphs from images plus precomputed segment dictionaries."""

    def __init__(self, **graphicalize_kwargs) -> None:
        self.graphicalize_kwargs = graphicalize_kwargs

    def transform(self, X, y=None) -> list[nx.MultiDiGraph]:
        if y is None:
            raise ValueError("ImageSegmentGraphicalizer.transform requires segment annotations in y")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        graphs: list[nx.MultiDiGraph] = []
        for image_np, segments in zip(X, y):
            graph = extract_geometric_relations_graph(segments, **self.graphicalize_kwargs)
            graph.graph["image"] = np.asarray(image_np).copy()
            graph.graph["segments"] = list(segments)
            graphs.append(graph)
        return graphs
