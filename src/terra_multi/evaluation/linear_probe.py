"""Linear estimators shared by every downstream evaluation."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def reduce_features(features, train_indices, n_components, seed=0):
    """Fit PCA on training rows and transform every row."""
    features = np.asarray(features, dtype=np.float32)
    if not n_components:
        return features, None

    train_indices = np.asarray(train_indices, dtype=int)
    n_components = int(n_components)
    maximum = min(len(train_indices), features.shape[1])
    if n_components > maximum:
        raise ValueError(
            f"cannot fit {n_components} PCA components from {maximum} available"
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=int(seed),
    )
    pca.fit(features[train_indices])
    transformed = pca.transform(features).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    return transformed, explained


def _classifier(C, class_weight, max_iter):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    max_iter=int(max_iter),
                ),
            ),
        ]
    )


def fit_linear_classifier(
    features,
    labels,
    Cs,
    class_weight="balanced",
    max_iter=2000,
):
    """Fit the historical three-fold HEST linear probe."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=list(Cs),
                    cv=3,
                    scoring="balanced_accuracy",
                    class_weight=class_weight,
                    max_iter=int(max_iter),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def fit_validated_linear_classifier(
    features,
    labels,
    train_indices,
    val_indices,
    Cs,
    class_weight="balanced",
    max_iter=2000,
):
    """Select C on validation macro-F1, then refit on train and validation."""
    best_C, best_score = None, -np.inf
    for C in Cs:
        model = _classifier(C, class_weight, max_iter)
        model.fit(features[train_indices], labels[train_indices])
        predicted = model.predict(features[val_indices])
        score = f1_score(
            labels[val_indices], predicted, average="macro", zero_division=0
        )
        if score > best_score:
            best_C, best_score = float(C), float(score)

    fit_indices = np.concatenate([train_indices, val_indices])
    model = _classifier(best_C, class_weight, max_iter)
    model.fit(features[fit_indices], labels[fit_indices])
    return model, best_C, best_score


def fit_linear_regressor(features, targets, alphas):
    """Fit the historical multi-target expression probe."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=list(alphas), alpha_per_target=True)),
        ]
    )
    model.fit(features, targets)
    return model
