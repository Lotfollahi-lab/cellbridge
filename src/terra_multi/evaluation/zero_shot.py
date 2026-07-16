"""Non-parametric kNN evaluation against labelled reference embeddings.

This is called "zero-shot" in experiment configs because no model parameters
are trained. It is still label-reference evaluation: neighbours come from the
labelled training split.
"""

from sklearn.neighbors import KNeighborsClassifier


def fit_knn_classifier(
    features,
    labels,
    n_neighbors=20,
    metric="cosine",
    weights="uniform",
):
    """Index labelled embeddings for deterministic nearest-neighbour voting."""
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors > len(features):
        raise ValueError(
            f"n_neighbors={n_neighbors} exceeds {len(features)} reference rows"
        )

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        algorithm="brute",
        # Brute-force neighbour search is O(n_test x n_train x d); parallelise the
        # query across cores so a large sample is not left on a single thread.
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model
