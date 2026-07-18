"""Graphicalizers for converting raw data into labeled NetworkX graphs."""
from abstractgraph_graphicalizer.chem import (
    CHEM_EDGE_SCHEMA,
    CHEM_NODE_SCHEMA,
    MoleculeGraphicalizer,
    draw_graph,
    draw_molecule,
    graph_to_rdmol,
    rdmol_to_graph,
    sdf_to_graphs,
    smi_to_graphs,
    smiles_list_to_graphs,
    smiles_to_graph,
)
from abstractgraph_graphicalizer.data import (
    DataMatrixGraphicalizer,
    FeatureCorrelationGraphicalizer,
    data_matrix_to_feature_graph,
    data_to_graph,
)
from abstractgraph_graphicalizer.image import (
    ImageSegmentGraphicalizer,
    extract_geometric_relations_graph,
    load_images,
    visualize_scene_graph_on_image,
)
from abstractgraph_graphicalizer.protein import (
    AMINO_ACID_ALPHABETS,
    ProteinChainRecord,
    ProteinContactNetworkGraphicalizer,
    ProteinContactNetworkLoader,
    ProteinLabelGraphicalizer,
    ResidueCA,
    download_mmcif,
    extract_chain_record,
    label_protein_contact_graph,
    protein_chain_record_to_pcn,
)
from abstractgraph_graphicalizer.graph import (
    NodeEmbedderGraphGraphicalizer,
    NormalizedLaplacianSVDGraphGraphicalizer,
    ProductGraphGraphicalizer,
    MutualNearestNeighbourGraphicalizer,
    NearestNeighborVectorGraphicalizer,
    SequenceGraphicalizer,
    StringGraphicalizer,
    annotate_normalized_laplacian_svd,
    mutual_nearest_neighbour_graph,
    normalized_laplacian_svd,
    product_graph,
    sequence_to_graph,
    string_to_graph,
)
from abstractgraph_graphicalizer.rna import (
    RNAFoldGraphicalizer,
    RNASequenceGraphicalizer,
    SequenceReverseComplementGraphicalizer,
    make_reverse_complement_graph,
    read_fasta,
    rnafold_to_graphs,
    seq_struct_to_graph,
    seq_to_graph,
    sequence_dotbracket_to_graph,
)
from abstractgraph_graphicalizer.bottleneck import (
    BottleneckGraphicalizer,
    BottleneckOutput,
    GraphInterpretationBottleneck,
    HiddenAutomaton,
    TinySequenceTransformer,
    aggregate_bottleneck_graphs,
    bottleneck_output_to_networkx,
    collapse_graph_to_states,
    edge_recovery_diagnostics,
    evaluate_automaton_recovery,
    extract_sequence_embeddings,
    generate_automaton_sequences,
    sample_hidden_automaton,
    state_assignment_diagnostics,
    train_tiny_sequence_transformer,
    transition_graph_from_assignments,
)

_ATTENTION_EXPORTS = {
    "AbstractGraphPreprocessor",
    "ImageNodeClusterer",
}

_attention_import_error = None
try:
    from abstractgraph_graphicalizer.attention import AbstractGraphPreprocessor, ImageNodeClusterer
except (ImportError, OSError) as exc:
    _attention_import_error = exc

__all__ = [
    "AbstractGraphPreprocessor",
    "BottleneckGraphicalizer",
    "BottleneckOutput",
    "ImageNodeClusterer",
    "GraphInterpretationBottleneck",
    "HiddenAutomaton",
    "TinySequenceTransformer",
    "aggregate_bottleneck_graphs",
    "bottleneck_output_to_networkx",
    "collapse_graph_to_states",
    "edge_recovery_diagnostics",
    "evaluate_automaton_recovery",
    "extract_sequence_embeddings",
    "generate_automaton_sequences",
    "sample_hidden_automaton",
    "state_assignment_diagnostics",
    "train_tiny_sequence_transformer",
    "transition_graph_from_assignments",
    "MoleculeGraphicalizer",
    "CHEM_NODE_SCHEMA",
    "CHEM_EDGE_SCHEMA",
    "smiles_to_graph",
    "smiles_list_to_graphs",
    "sdf_to_graphs",
    "smi_to_graphs",
    "rdmol_to_graph",
    "graph_to_rdmol",
    "draw_molecule",
    "draw_graph",
    "DataMatrixGraphicalizer",
    "FeatureCorrelationGraphicalizer",
    "data_matrix_to_feature_graph",
    "data_to_graph",
    "SequenceGraphicalizer",
    "StringGraphicalizer",
    "sequence_to_graph",
    "string_to_graph",
    "MutualNearestNeighbourGraphicalizer",
    "NearestNeighborVectorGraphicalizer",
    "mutual_nearest_neighbour_graph",
    "normalized_laplacian_svd",
    "annotate_normalized_laplacian_svd",
    "NormalizedLaplacianSVDGraphGraphicalizer",
    "NodeEmbedderGraphGraphicalizer",
    "product_graph",
    "ProductGraphGraphicalizer",
    "RNASequenceGraphicalizer",
    "RNAFoldGraphicalizer",
    "SequenceReverseComplementGraphicalizer",
    "sequence_dotbracket_to_graph",
    "seq_struct_to_graph",
    "seq_to_graph",
    "rnafold_to_graphs",
    "make_reverse_complement_graph",
    "read_fasta",
    "ImageSegmentGraphicalizer",
    "extract_geometric_relations_graph",
    "visualize_scene_graph_on_image",
    "load_images",
    "ProteinChainRecord",
    "ProteinContactNetworkGraphicalizer",
    "ProteinContactNetworkLoader",
    "ProteinLabelGraphicalizer",
    "ResidueCA",
    "AMINO_ACID_ALPHABETS",
    "download_mmcif",
    "extract_chain_record",
    "label_protein_contact_graph",
    "protein_chain_record_to_pcn",
]


def __getattr__(name: str):
    if name in _ATTENTION_EXPORTS and _attention_import_error is not None:
        raise ImportError(
            "Attention components require optional torch dependencies that could not be imported."
        ) from _attention_import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
