"""Ridge linear probe: standardize on train, then RidgeCV."""

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(X_train, Y_train, alphas):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=list(alphas), alpha_per_target=True)),
        ]
    )
    model.fit(X_train, Y_train)
    return model


def predict(model, X):
    return model.predict(X)
