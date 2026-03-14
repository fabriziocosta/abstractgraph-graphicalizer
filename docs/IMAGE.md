# Image Graphicalizers

The image backend builds scene graphs from images plus precomputed segment
descriptions such as bounding boxes or masks.

## Intended input

Use this backend when you already have:

- RGB images as NumPy arrays
- segment dictionaries with bounding boxes
- optional masks, labels, captions, or confidence scores

This backend does not currently own the full object-detection or semantic
classification pipeline. It assumes those segment proposals already exist and
focuses on turning them into a graph.

## Main entrypoints

- `extract_geometric_relations_graph`
- `ImageSegmentGraphicalizer`
- `visualize_scene_graph_on_image`
- `load_images`

## Output idea

The output is a `networkx.MultiDiGraph` whose nodes represent segmented image
objects. Edges encode geometric relations such as overlap, containment,
proximity, left-of, and above. Graph metadata can retain the original image and
segment list for downstream visualization or further processing.

## When to use it

Use this backend when image structure should be represented relationally after
segmentation, especially in workflows where object proposals come from an
external detector or are curated manually.
