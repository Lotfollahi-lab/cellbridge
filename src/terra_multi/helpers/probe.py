"""Linear probes: StandardScaler on train, then RidgeCV (regression) or
LogisticRegressionCV (classification)."""

from sklearn.linear_model import LogisticRegressionCV, RidgeCV
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


def fit_classifier(X_train, y_train, Cs, class_weight="balanced"):
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
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model
