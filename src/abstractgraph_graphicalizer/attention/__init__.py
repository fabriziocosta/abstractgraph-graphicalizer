"""Attention-driven graph induction backends."""

_attention_import_error = None
try:
    from abstractgraph_graphicalizer.attention.preprocessor import AbstractGraphPreprocessor, ImageNodeClusterer
except (ImportError, OSError) as exc:
    _attention_import_error = exc

__all__ = ["AbstractGraphPreprocessor", "ImageNodeClusterer"]


def __getattr__(name: str):
    if name in __all__ and _attention_import_error is not None:
        raise ImportError(
            "Attention components require optional torch dependencies that could not be imported."
        ) from _attention_import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
