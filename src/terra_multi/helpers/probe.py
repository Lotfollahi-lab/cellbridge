"""Backward-compatible imports for the shared evaluation estimators."""

from terra_multi.evaluation.linear_probe import (
    fit_linear_classifier,
    fit_linear_regressor,
)


def fit_probe(X_train, Y_train, alphas):
    return fit_linear_regressor(X_train, Y_train, alphas)


def predict(model, X):
    return model.predict(X)


def fit_classifier(X_train, y_train, Cs, class_weight="balanced"):
    return fit_linear_classifier(X_train, y_train, Cs, class_weight)
