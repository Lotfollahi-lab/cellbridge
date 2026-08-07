"""Lorentz alignment components for CellBridge."""

__all__ = ["LorentzOps", "OntologyEmbedding"]


def __getattr__(name):
    """Keep the Euclidean baseline importable when the geoopt extra is absent."""

    if name in __all__:
        from .manifold import LorentzOps, OntologyEmbedding

        return {"LorentzOps": LorentzOps, "OntologyEmbedding": OntologyEmbedding}[name]
    raise AttributeError(name)
