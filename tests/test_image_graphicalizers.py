from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image

from abstractgraph_graphicalizer.image import (
    ImageSegmentGraphicalizer,
    extract_geometric_relations_graph,
    filter_by_size,
    filter_overlapping_by_iou,
    load_images,
    visualize_scene_graph_on_image,
)


class ImageGraphicalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)
        self.segments = [
            {"bbox": [1, 1, 6, 6], "label": "left", "score": 0.9},
            {"bbox": [10, 1, 16, 6], "label": "right", "score": 0.8},
        ]

    def test_filter_helpers(self) -> None:
        filtered = filter_by_size(self.segments, (20, 20), min_size=0.01)
        self.assertEqual(len(filtered), 2)
        overlap = filter_overlapping_by_iou(
            self.segments + [{"bbox": [1, 1, 6, 6], "label": "dup", "score": 0.1}],
            iou_threshold=0.5,
        )
        self.assertEqual(len(overlap), 2)

    def test_extract_geometric_relations_graph(self) -> None:
        graph = extract_geometric_relations_graph(self.segments, include_left_of=True, include_above=False)
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertTrue(any(data["relation"] == "is_left_of" for _, _, data in graph.edges(data=True)))

    def test_visualize_scene_graph_on_image(self) -> None:
        graph = extract_geometric_relations_graph(self.segments, include_left_of=True, include_above=False)
        ax = visualize_scene_graph_on_image(self.image, self.segments, graph)
        self.assertTrue(hasattr(ax, "imshow"))

    def test_image_segment_graphicalizer(self) -> None:
        graphs = ImageSegmentGraphicalizer(include_left_of=True, include_above=False).transform(
            [self.image], [self.segments]
        )
        self.assertEqual(len(graphs), 1)
        self.assertIn("image", graphs[0].graph)
        self.assertIn("segments", graphs[0].graph)

    def test_load_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            array = np.ones((5, 5, 3), dtype=np.uint8) * 255
            Image.fromarray(array).save(Path(tmpdir) / "a.jpg")
            Image.fromarray(array).save(Path(tmpdir) / "b.jpg")
            images, names = load_images(tmpdir, return_names=True)
        self.assertEqual(len(images), 2)
        self.assertEqual(names, ["a.jpg", "b.jpg"])


if __name__ == "__main__":
    unittest.main()
