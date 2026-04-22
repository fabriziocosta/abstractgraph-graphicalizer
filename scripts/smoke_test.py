"""Smoke test for abstractgraph-graphicalizer."""

from __future__ import annotations

import numpy as np

from abstractgraph_graphicalizer.attention import AbstractGraphPreprocessor
from abstractgraph_graphicalizer.protein import (
    ProteinChainRecord,
    ProteinContactNetworkGraphicalizer,
    ProteinLabelGraphicalizer,
    ResidueCA,
)


def main() -> None:
    X = [np.random.randn(8, 4), np.random.randn(6, 4)]
    y = [0, 1]
    preprocessor = AbstractGraphPreprocessor(d_model=8, n_heads=2, num_layers=1, n_epochs=1)
    preprocessor.fit(X, y)
    graphs = preprocessor.transform(X)
    print("attention_graphs", len(graphs), len(graphs[0]))

    record = ProteinChainRecord(
        sample_id="demo.A",
        pdb_id="demo",
        label_asym_id="A",
        auth_asym_id="A",
        residues=(
            ResidueCA("ALA", "A", "A", "1", "1", "", (0.0, 0.0, 0.0)),
            ResidueCA("GLY", "A", "A", "2", "2", "", (5.0, 0.0, 0.0)),
        ),
    )
    protein_graph = ProteinContactNetworkGraphicalizer().transform([record])[0]
    print("protein_graph", protein_graph.number_of_nodes(), protein_graph.number_of_edges())
    labeled_graph = ProteinLabelGraphicalizer(alphabet="hp2").transform([protein_graph])[0]
    print("protein_labels", labeled_graph.nodes[0]["label"], labeled_graph.edges[0, 1]["label"])


if __name__ == "__main__":
    main()
